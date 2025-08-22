from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.RAFDB_aug import RafDataset

from utils.path_manager import create_output_dirs
from utils.predict_utils import save_class_mapping


def build_dataloader_rafdb(cfg):
    # RAFDB/SFEW 里 Dataset返回的是PIL → Transform统一ToTensor
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),  # 对图像进行归一化
        transforms.RandomHorizontalFlip(),
        transforms.RandomErasing(scale=(0.02, 0.25))
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = RafDataset(cfg, phase='train', transform=train_transforms)
    print(f"[INFO] Number of training samples: {len(train_dataset)}")
    # img, label, idx, img1 = train_dataset[0]
    # print(img.shape)  # torch.Size([3, 224, 224])
    # print(label)  # 类别标签，int 类型
    # print(idx)  # 样本索引
    # print(img1.shape)  # torch.Size([3, 224, 224])

    # 保存类别映射
    outputs_dir, _ = create_output_dirs(cfg)
    save_class_mapping(train_dataset, outputs_dir)

    val_dataset = RafDataset(cfg, phase='val', transform=test_transforms)
    print(f"[INFO] Number of validation samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.batch_size,
                              shuffle=True,  # 打乱顺序
                              num_workers=cfg.workers,
                              pin_memory=True)

    test_loader = DataLoader(val_dataset,
                             batch_size=cfg.batch_size,
                             shuffle=False,  # 不打乱顺序
                             num_workers=cfg.workers,
                             pin_memory=True)
    return train_loader, test_loader
