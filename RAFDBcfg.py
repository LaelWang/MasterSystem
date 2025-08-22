"""
Motto  : To Advance Infinitely
Time   : 2025/4/27 15:30
Author : LingQi Wang
"""


class RAFDBConfig:
    name = "RAFDB"
    img_size = 224
    raf_path = 'W:/PyCharm Project/FER/DATASETS/RAF-DB (basic)'
    num_classes = 7
    seed = 3407
    device = 0
    batch_size = 32
    workers = 4
    # 设置模型选择
    # 使用时只需要调整这里！
    model_config = 'CLIP_CBAM_FocalLoss'  # 可选: ResNet18, ResNet34, ResNet50, WTConvNeXt, CLIP, ConvNeXt

    # FAILED
    # model4 = 'wtconv2d'
    # pretrained_path = '' # ResNet-18 ImageNet 权重Conv2d和WTConv2d参数完全不同
    # backbone = "res18"
    # modules = 'wtconv2d'
    # epochs = 50

    # 模型配置列表
    configs = {
        'ViT': {
            'model': 'ViT',
            'backbone': 'Transformer',
            'modules': '',
            'size': 'Base',
            'epochs': 50,
            'optimizer': 'AdamW',
            'lr': 0.0002,
            'weight_decay': 0.05,
            'scheduler': 'CosineAnnealingLR',
            'gamma': 0.9,
            'T_max': 50,
            'pretrained_path': 'checkpoints/vit_base_patch16_224.pth'
        },
        'CLIP': {
            'model': 'CLIP',
            'backbone': 'ResNet18',
            'modules': '',
            'size': 'Base',
            'epochs': 50,
            'optimizer': 'Adam',
            'lr': 0.0002,
            'weight_decay': 1e-4,
            'scheduler': 'ExponentialLR',
            'gamma': 0.9,
            'T_max': 50,
            'pretrained_path': 'checkpoints/resnet18_msceleb.pth'
        },
        'CLIPNoAug': {
            'model': 'CLIP',
            'backbone': 'ResNet18',
            'modules': '',
            'size': 'Base',
            'epochs': 50,
            'optimizer': 'Adam',
            'lr': 0.0002,
            'weight_decay': 1e-4,
            'scheduler': 'ExponentialLR',
            'gamma': 0.9,
            'T_max': 50,
            'pretrained_path': 'checkpoints/resnet18_msceleb.pth'
        },
        'CLIP_SE': {
            'model': 'CLIP',
            'backbone': 'ResNet18',
            'modules': 'SE',
            'size': 'Base',
            'epochs': 50,
            'optimizer': 'Adam',
            'lr': 0.0002,
            'weight_decay': 1e-4,
            'scheduler': 'ExponentialLR',
            'gamma': 0.9,
            'T_max': 50,
            'pretrained_path': 'checkpoints/resnet18_msceleb.pth'
        },
        'CLIP_ECA': {
            'model': 'CLIP',
            'backbone': 'ResNet18',
            'modules': 'ECA',
            'size': 'Base',
            'epochs': 50,
            'optimizer': 'Adam',
            'lr': 0.0002,
            'weight_decay': 1e-4,
            'scheduler': 'ExponentialLR',
            'gamma': 0.9,
            'T_max': 50,
            'pretrained_path': 'checkpoints/resnet18_msceleb.pth'
        },
        'CLIP_Center': {
            'model': 'CLIP',
            'backbone': 'ResNet18',
            'modules': 'None',
            'size': 'Base',
            'epochs': 50,
            'optimizer': 'Adam',
            'lr': 0.0002,
            'weight_decay': 1e-4,
            'scheduler': 'ExponentialLR',
            'gamma': 0.9,
            'T_max': 50,
            'pretrained_path': 'checkpoints/resnet18_msceleb.pth',
            'loss': 'CenterLoss'
        },
        'CLIP_Focal': {
            'model': 'CLIP',
            'backbone': 'ResNet18',
            'modules': 'None',
            'size': 'Base',
            'epochs': 50,
            'optimizer': 'Adam',
            'lr': 0.0002,
            'weight_decay': 1e-4,
            'scheduler': 'ExponentialLR',
            'gamma': 0.9,
            'T_max': 50,
            'pretrained_path': 'checkpoints/resnet18_msceleb.pth',
            'loss': 'FocalLoss'
        },
        'CLIP_CBAM_FocalLoss': {
            'model': 'CLIP',
            'backbone': 'ResNet18',
            'modules': 'CBAM',
            'size': 'Base',
            'epochs': 50,
            'optimizer': 'Adam',
            'lr': 0.0002,
            'weight_decay': 1e-4,
            'scheduler': 'ExponentialLR',
            'gamma': 0.9,
            'T_max': 50,
            'pretrained_path': 'checkpoints/resnet18_msceleb.pth',
            'loss': 'FocalLoss'
        },
        'CLIP_CBAM_CenterLoss': {
            'model': 'CLIP',
            'backbone': 'ResNet18',
            'modules': 'CBAM',
            'size': 'Base',
            'epochs': 50,
            'optimizer': 'Adam',
            'lr': 0.0002,
            'weight_decay': 1e-4,
            'scheduler': 'ExponentialLR',
            'gamma': 0.9,
            'T_max': 50,
            'pretrained_path': 'checkpoints/resnet18_msceleb.pth',
            'loss': 'CenterLoss'
        },
        'CLIP_TripletA': {
            'model': 'CLIP',
            'backbone': 'ResNet18',
            'modules': 'TripletA',
            'size': 'Base',
            'epochs': 50,
            'optimizer': 'Adam',
            'lr': 0.0002,
            'weight_decay': 1e-4,
            'scheduler': 'ExponentialLR',
            'gamma': 0.9,
            'T_max': 50,
            'pretrained_path': 'checkpoints/resnet18_msceleb.pth'
        },
        'ResNet18': {
            'model': 'ResNet18',
            'backbone': 'ResNet',
            'modules': '',
            'size': 'Base',
            'epochs': 50,
            'optimizer': 'SGD',
            'lr': 0.01,
            'weight_decay': 5e-4,
            'momentum': 0.9,
            'scheduler': 'StepLR',
            'step_size': 10,
            'gamma': 0.1,
            'T_max': 50
        },
        'ResNet34': {
            'model': 'ResNet34',
            'backbone': 'ResNet',
            'modules': '',
            'size': 'Base',
            'epochs': 50,
            'optimizer': 'SGD',
            'lr': 0.01,
            'weight_decay': 5e-4,
            'momentum': 0.9,
            'scheduler': 'StepLR',
            'step_size': 10,
            'gamma': 0.1,
            'T_max': 50
        },
        'ResNet50': {
            'model': 'ResNet50',
            'backbone': 'ResNet',
            'modules': '',
            'size': 'Base',
            'epochs': 50,
            'optimizer': 'SGD',
            'lr': 0.01,
            'weight_decay': 5e-4,
            'momentum': 0.9,
            'scheduler': 'StepLR',
            'step_size': 10,
            'gamma': 0.1,
            'T_max': 50
        },
        'WTConvNeXt': {
            'model': 'WTConvNeXt',
            'backbone': 'ConvNeXt',
            'modules': '',
            'size': 's',
            'epochs': 50,
            'optimizer': 'AdamW',
            'lr': 0.025,
            'weight_decay': 5e-2,
            'momentum': 0.9,
            'gamma': 0.9,
            'scheduler': 'ExponentialLR',
            'T_max': 50,
            'pretrained_path': 'checkpoints/WTConvNeXt_small_5_300e_ema.pth'
        },
        'WTConvNeXt_SGD': {
            'model': 'WTConvNeXt',
            'backbone': 'ConvNeXt',
            'modules': '',
            'size': 's',
            'epochs': 50,
            'optimizer': 'SGD',
            'lr': 0.01,
            'weight_decay': 5e-4,
            'momentum': 0.9,
            'gamma': 0.1,
            'scheduler': 'ExponentialLR',
            'step_size': 10,
            'T_max': 50,
            'pretrained_path': 'checkpoints/WTConvNeXt_small_5_300e_ema.pth'
        },
        'WTConvNeXt_AdamW': {
            'model': 'WTConvNeXt',
            'backbone': 'ConvNeXt',
            'modules': '',
            'size': 's',
            'epochs': 50,
            'optimizer': 'AdamW',
            'lr': 0.0002,
            'weight_decay': 1e-4,
            'momentum': 0.9,
            'gamma': 0.9,
            'scheduler': 'CosineAnnealing',
            'T_max': 50,
            'pretrained_path': 'checkpoints/WTConvNeXt_small_5_300e_ema.pth'
        },
        'WTConvNeXt_Adam': {
            'model': 'WTConvNeXt',
            'backbone': 'ConvNeXt',
            'modules': '',
            'size': 's',
            'epochs': 50,
            'optimizer': 'Adam',
            'lr': 0.0002,
            'weight_decay': 1e-4,
            'momentum': 0.9,
            'gamma': 0.9,
            'scheduler': 'ExponentialLR',
            'T_max': 50,
            'pretrained_path': 'checkpoints/WTConvNeXt_small_5_300e_ema.pth'
        },
        'ConvNeXt': {
            'model': 'ConvNeXt',
            'backbone': 'ConvNeXt',
            'modules': '',
            'size': 'Base',
            'epochs': 50,
            'optimizer': 'AdamW',
            'lr': 0.0002,
            'weight_decay': 1e-4,
            'scheduler': 'CosineAnnealingLR',
            'T_max': 50
        }
    }

    # 依据选择动态应用配置
    selected = configs[model_config]
    model = selected.get('model')
    backbone = selected.get('backbone')
    modules = selected.get('modules', '')
    size = selected.get('size', '')
    epochs = selected.get('epochs')
    optimizer = selected.get('optimizer')
    lr = selected.get('lr')
    weight_decay = selected.get('weight_decay')
    momentum = selected.get('momentum', 0)
    scheduler = selected.get('scheduler')
    step_size = selected.get('step_size', 10)
    gamma = selected.get('gamma', 0.1)
    T_max = selected.get('T_max', 50)
    pretrained_path = selected.get('pretrained_path', '')
    loss = selected.get('loss', 'CrossEntropyLoss')
