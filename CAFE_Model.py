"""
Motto  : To Advance Infinitely
Time   : 2025/5/12 13:32
Author : LingQi Wang
"""
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
import random
from models.CLIP.ResNet import ResNet

from torch.autograd import Variable
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _single, _pair, _triple

class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.no_spatial = no_spatial
        self.channel_pool = lambda x: torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def spatial_attention(self, x):
        x_compress = self.channel_pool(x)
        x_out = self.spatial_conv(x_compress)
        return x * x_out

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.spatial_attention(x_perm1)
        x_out1 = x_out1.permute(0, 2, 1, 3).contiguous()

        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.spatial_attention(x_perm2)
        x_out2 = x_out2.permute(0, 3, 2, 1).contiguous()

        if not self.no_spatial:
            x_out3 = self.spatial_attention(x)
            out = (1/3) * (x_out1 + x_out2 + x_out3)
        else:
            out = 0.5 * (x_out1 + x_out2)

        return out

# ==================== Attention Modules ====================
class SEBlock(nn.Module):
    """Squeeze-and-Excitation block"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ECA(nn.Module):
    """Efficient Channel Attention"""
    def __init__(self, channels, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        b, c, _, _ = x.size()
        y = self.avg_pool(x)           # [B, C, 1, 1]
        y = y.view(b, 1, c)            # [B, 1, C]
        y = self.conv(y)               # [B, 1, C]
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ChannelAttention(nn.Module):
    """Channel Attention for CBAM"""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels//reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x))
        max_ = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg + max_)

class SpatialAttention(nn.Module):
    """Spatial Attention for CBAM"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        max_, _ = x.max(dim=1, keepdim=True)
        y = torch.cat([avg, max_], dim=1)
        return x * self.sigmoid(self.conv(y))

class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

class MultiHeadChannelAttention(nn.Module):
    """Channel-wise Multi-Head Self-Attention for mask generation"""
    def __init__(self, channels, num_heads=8, embed_dim=None):
        super().__init__()
        self.channels = channels
        self.embed_dim = embed_dim or (channels // num_heads)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.embed = nn.Linear(1, self.embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=num_heads)
        self.fc = nn.Linear(self.embed_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        b, c, _, _ = x.shape
        # scalar descriptor per channel
        y = self.avg_pool(x).view(b, c, 1)        # [B, C, 1]
        y = y.permute(1, 0, 2)                     # [C, B, 1]
        # embed to d_model
        y_embed = self.embed(y)                   # [C, B, embed_dim]
        # self-attention across channels
        attn_out, _ = self.mha(y_embed, y_embed, y_embed)  # [C, B, embed_dim]
        # project to channel weight
        w = self.fc(attn_out)                     # [C, B, 1]
        w = w.permute(1, 0, 2).view(b, c, 1, 1)    # [B, C, 1, 1]
        return self.sigmoid(w)


# 自定义的池化模块，用于对特征进行池化操作，同时通过维度转换确保在通道上能进行适当的操作
# 对输入张量做 最大池化 前先将数据维度做转置（将第3维和第1维交换），池化后再转置回来
class my_MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super(my_MaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    # 对输入张量做最大池化前先将数据维度做转置（将第3维和第1维交换），池化后再转置回来
    # 这种操作保证了在特定维度上（例如通道维度）做池化时数据排列符合预期
    def forward(self, input):
        input = input.transpose(3, 1)

        input = F.max_pool2d(input, self.kernel_size, self.stride,
                             self.padding, self.dilation, self.ceil_mode,
                             self.return_indices)
        input = input.transpose(3, 1).contiguous()

        return input

    def __repr__(self):
        kh, kw = _pair(self.kernel_size)
        dh, dw = _pair(self.stride)
        padh, padw = _pair(self.padding)
        dilh, dilw = _pair(self.dilation)
        padding_str = ', padding=(' + str(padh) + ', ' + str(padw) + ')' \
            if padh != 0 or padw != 0 else ''
        dilation_str = (', dilation=(' + str(dilh) + ', ' + str(dilw) + ')'
                        if dilh != 0 and dilw != 0 else '')
        ceil_str = ', ceil_mode=' + str(self.ceil_mode)
        return self.__class__.__name__ + '(' \
            + 'kernel_size=(' + str(kh) + ', ' + str(kw) + ')' \
            + ', stride=(' + str(dh) + ', ' + str(dw) + ')' \
            + padding_str + dilation_str + ceil_str + ')'


##### channel dropping
# 生成一个用于通道 dropping 的随机掩码，用以实现通道分离s
# 遍历 7 个基本表情类别（0～6），前 6 类生成 63 个 1 和 10 个 0
# 最后一类生成 64 个 1 和 10 个 0，共组成 512 个数（对应 ResNet-18 输出的通道数）
# 每一类内通过 random.shuffle 打乱顺序
# 对整个 batch 复制该掩码，reshape 成 (batch_size, 512, 1, 1)，并转换为 CUDA 张量
# 生成的掩码将在后续与固定人脸特征相乘，起到屏蔽部分通道的作用，从而迫使模型在各个通道间学习到多样化的表达信息
def Mask(nb_batch):
    bar = []
    for i in range(7):
        foo = [1] * 63 + [0] * 10
        if i == 6:
            foo = [1] * 64 + [0] * 10
        random.shuffle(foo)  #### generate mask
        bar += foo
    bar = [bar for i in range(nb_batch)]
    bar = np.array(bar).astype("float32")
    bar = bar.reshape(nb_batch, 512, 1, 1)
    bar = torch.from_numpy(bar)
    bar = bar.cuda()
    bar = Variable(bar)
    return bar


# 计算两部分损失
# Separation loss（loss_2）：对特征进行池化后，计算每个分组中特征的总和，利用“1 - 平均总和/cnum”作为惩罚项，鼓励每个分组内的激活分布更加均衡
# Cls loss（分类损失）（loss_1）：先利用随机掩码（由 Mask 函数生成）对输入特征进行通道 dropping，
# 再经过自定义最大池化（my_MaxPool2d）得到一个向量，之后利用交叉熵损失与真实标签对比
def supervisor(x, targets, cnum):
    branch = x
    branch = branch.reshape(branch.size(0), branch.size(1), 1, 1)
    branch = my_MaxPool2d(kernel_size=(1, cnum), stride=(1, cnum))(branch)
    branch = branch.reshape(branch.size(0), branch.size(1), branch.size(2) * branch.size(3))
    loss_2 = 1.0 - 1.0 * torch.mean(torch.sum(branch, 2)) / cnum  # set margin = 3.0

    mask = Mask(x.size(0))
    branch_1 = x.reshape(x.size(0), x.size(1), 1, 1) * mask
    branch_1 = my_MaxPool2d(kernel_size=(1, cnum), stride=(1, cnum))(branch_1)
    branch_1 = branch_1.view(branch_1.size(0), -1)
    loss_1 = nn.CrossEntropyLoss()(branch_1, targets)
    return [loss_1, loss_2]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # 下采样使用了一个 1x1 卷积核和相应的批归一化（BatchNorm），用于匹配输入输出的通道数，并确保残差连接的维度一致
        if downsample:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                             stride=stride, bias=False)
            bn = nn.BatchNorm2d(out_channels)
            downsample = nn.Sequential(conv, bn)
        else:
            downsample = None

        self.downsample = downsample

    def forward(self, x):

        i = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            i = self.downsample(i)

        x += i
        x = self.relu(x)

        return x


from models.CLIP.clip import clip

# 设置cuda设备、加载预训练的clip模型
device = torch.device('cuda:0')
# Vision Transformer，B表示基础规模，32指图像被切分为32 x 32的patch，
# 输入图像会被划分为固定大小的图像块，每个块大小为 32×32 像素，然后这些小块被嵌入到模型中
# clip.load返回一个元组，第一个元素是加载好的 CLIP 模型，第二个元素就是对应的预处理函数
clip_model, preprocess = clip.load("ViT-B/32", device=device)
import os


class Model(nn.Module):

    # 测试也需要修改
    def __init__(self, pretrained=True, num_classes=7, drop_rate=0, attention_type='triplet'):
        super(Model, self).__init__()

        # 加载在 MS-Celeb 数据集上预训练的 ResNet-18 权重，以便获得较强的人脸特征提取能力
        res18 = ResNet(block=BasicBlock, n_blocks=[2, 2, 2, 2], channels=[64, 128, 256, 512], output_dim=1000)
        # 使用动态、跨平台兼容的 相对路径解析
        current_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(current_dir, '..', '..', 'checkpoints', 'resnet18_msceleb.pth')
        weights_path = os.path.normpath(weights_path)  # 标准化路径（跨平台）

        msceleb_model = torch.load(weights_path)  # train的路径，但是test时路径就是错的
        state_dict = msceleb_model['state_dict']
        res18.load_state_dict(state_dict,
                              strict=False)  # strict=False 来允许加载部分模型权重，这意味着如果某些层的权重不匹配（例如，模型结构的轻微调整），可以忽略这些不匹配的部分

        # 提取特征模块拆分
        # self.features：由ResNet除最后两个层之外的所有层构成，用于提取较底层的特征
        # self.features2：取ResNet中倒数第二层，目的是进一步得到一个用于生成掩码的特征表示
        # self.fc：新建一个全连接层，将经过掩码筛选后的512维特征映射到7个表情类别上
        self.drop_rate = drop_rate
        self.features = nn.Sequential(
            *list(res18.children())[:-2])  # Sequential((0): AdaptiveAvgPool2d(output_size=(1, 1)))
        self.features2 = nn.Sequential(
            *list(res18.children())[-2:-1])  # Sequential((0): Linear(in_features=512, out_features=1000, bias=True))

        fc_in_dim = list(res18.children())[-1].in_features  # original fc layer's in dimention 512
        self.fc = nn.Linear(fc_in_dim, num_classes)  # new fc layer 512x7

        # Attention module
        self.attention_type = attention_type.lower()
        if self.attention_type == 'se':
            self.attention = SEBlock(fc_in_dim)
        elif self.attention_type == 'eca':
            self.attention = ECA(fc_in_dim)
        elif self.attention_type == 'cbam':
            self.attention = CBAM(fc_in_dim)
        elif self.attention_type == 'triplet':
            self.attention = TripletAttention()
        elif self.attention_type == 'senetv2':
            from module.dream_code.SENet_v2 import SEAttention
            self.attention = SEAttention(channel=fc_in_dim)
        elif self.attention_type == 'mhsa':
            self.attention = MultiHeadChannelAttention(fc_in_dim, num_heads=8, embed_dim=fc_in_dim // 8)
        else:
            self.attention = nn.Identity()

        self.parm = {}
        for name, parameters in self.fc.named_parameters():
            print(name, ':', parameters.size())
            self.parm[name] = parameters

    # 前向传播流程
    def forward(self, x, clip_model, targets, phase='train'):

        # 固定人脸特征提取：利用预训练的 CLIP 模型（在外部固定住参数）对输入图像x提取出固定的通用人脸特征（Fixed face feature）
        with torch.no_grad():
            image_features = clip_model.encode_image(x)

        # FER特征提取与掩码生成：将输入图像通过预训练 ResNet 模型的部分层得到特征x，
        # 这部分特征用于学习一个掩码。接下来，通过 torch.sigmoid(x) 将该特征归一化到 [0,1] 区间，作为 Sigmoid mask
        x = self.features(x)  # 由ResNet除最后两个层之外的所有层去提取图片的底层特征
        feat = x

        # apply attention
        if self.attention_type == 'senetv2':
            assert feat.shape[1] == 512, f"SEAttention expects 512 channels, got {feat.shape[1]}"
            feat = self.attention(feat, feat, feat)
        else:
            feat = self.attention(feat)

        # apply attention
        # feat = self.attention(feat)

        x = self.features2(feat)
        x = x.view(x.size(0), -1)

        ################### sigmoid mask (important)
        # 利用 Sigmoid 掩码筛选固定人脸特征
        # 将CLIP提取的固定人脸特征与ResNet生成的Sigmoidmask相乘，实现“特征筛选”，得到Selected face feature。
        # 在训练阶段，调用supervisor函数计算两个额外的损失（MC_loss），包括分离和多样性损失
        if phase == 'train':
            MC_loss = supervisor(image_features * torch.sigmoid(x), targets, cnum=73)

        x = image_features * torch.sigmoid(x)
        # 分类预测：将筛选后的特征输入全连接层进行最终的 7 类表情分类
        out = self.fc(x)

        # 输出：在训练阶段，返回分类结果和 MC_loss（用于反向传播）；在测试阶段，直接返回预测结果
        if phase == 'train':
            return out, MC_loss
        else:
            return out, out
