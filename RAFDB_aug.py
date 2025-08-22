"""
Motto  : To Advance Infinitely
Time   : 2025/5/16 19:18
Author : LingQi Wang
"""

import os
import random
import numpy as np
from PIL import Image
import torch.utils.data as data
from torchvision import transforms
import cv2


def add_g(image_array, mean=0.0, var=30):
    std = var ** 0.5
    image_add = image_array + np.random.normal(mean, std, image_array.shape)
    image_add = np.clip(image_add, 0, 255).astype(np.uint8)
    return image_add

def flip_image(image_array):
    return cv2.flip(image_array, 1)


class RafDataset(data.Dataset):
    def __init__(self, args, phase, basic_aug=True, transform=None):
        self.data_root = args.raf_path
        self.phase = phase
        self.basic_aug = basic_aug
        self.transform = transform

        if self.phase not in ['train', 'val']:
            raise ValueError(f"Invalid phase {self.phase}")

        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}

        phase_dir = os.path.join(self.data_root, self.phase)
        classes = sorted(os.listdir(phase_dir))

        for idx, class_name in enumerate(classes):
            class_dir = os.path.join(phase_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            self.class_to_idx[class_name] = idx
            for img_name in os.listdir(class_dir):
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                img_path = os.path.join(class_dir, img_name)
                self.image_paths.append(img_path)
                self.labels.append(idx)

        self.aug_func = [flip_image, add_g]
        print(f"[INFO] RAFDB_aug loaded {len(self.image_paths)} samples with {len(self.class_to_idx)} classes")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = cv2.imread(img_path)[:, :, ::-1]  # ← np.ndarray，BGR to RGB

        # if self.phase == 'train':
        #     if self.basic_aug:
        #         if random.random() > 0.5:
        #             image = self.aug_func[0](image)
        #         if random.random() > 0.5:
        #             image = self.aug_func[1](image)

        if self.phase == 'train':
            if self.basic_aug and random.random() > 0.5:
                image = self.aug_func[1](image)  # add_g only

        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)

        return image, label, idx, image
