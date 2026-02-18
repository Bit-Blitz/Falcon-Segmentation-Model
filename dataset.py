import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class FalconDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = [f for f in os.listdir(images_dir) if f.endswith('.png')]
        # Mapping based on dataset analysis: 
        # [100, 200, 300, 500, 550, 600, 700, 800, 7100, 10000] found
        # All must be mapped to 0-9.
        # 100 -> 0
        # 200 -> 1
        # 300 -> 2
        # 500 -> 3
        # 550 -> 4
        # 600 -> 5
        # 700 -> 6
        # 800 -> 7
        # 7100 -> 8
        # 10000 -> 9
        
        self.mapping = np.ones(65536, dtype=np.int64) * 255
        self.mapping[100] = 0
        self.mapping[200] = 1
        self.mapping[300] = 2
        self.mapping[500] = 3
        self.mapping[550] = 4
        self.mapping[600] = 5
        self.mapping[700] = 6
        self.mapping[800] = 7
        self.mapping[7100] = 8
        self.mapping[10000] = 9
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        
        # Apply mapping using lookup table
        mask_mapped = self.mapping[mask]
        
        # Convert to uint8 for Albumentations (it expects uint8 for masks usually, or we can keep it as is if values are < 255)
        # But our values are 0-5 and 255. So uint8 is fine.
        mask_mapped = mask_mapped.astype(np.uint8)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask_mapped)
            image = augmented['image']
            mask_mapped = augmented['mask']
            
        return image, mask_mapped.long()

def get_transforms(phase):
    if phase == 'train':
        return A.Compose([
            A.Resize(512, 512),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.2),
            A.GaussNoise(p=0.2),
            A.RandomResizedCrop(size=(512, 512), scale=(0.7, 1.0)),
            A.Rotate(limit=10, p=0.5, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=255),
            A.Normalize(),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(512, 512),
            A.Normalize(),
            ToTensorV2()
        ])
