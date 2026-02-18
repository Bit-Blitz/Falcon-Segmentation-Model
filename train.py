import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
import random
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from model_hrnet import HRNetV2_Segmentation
from loss import CombinedLoss
from dataset import FalconDataset
from utils import compute_intersection_union

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True # Optimized for fixed input size

def freeze_bn(m):
    if isinstance(m, nn.BatchNorm2d):
        m.eval()
        for p in m.parameters():
            p.requires_grad = False

def get_hrnet_transforms(phase):
    # HRNet input size 384x384
    # Improved augmentation for better generalization
    if phase == 'train':
        return A.Compose([
            # RandomResizedCrop is better than simple Resize
            A.RandomResizedCrop(size=(384, 384), scale=(0.5, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=255), 
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(384, 384),
            A.Normalize(),
            ToTensorV2()
        ])

def train(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataset
    train_dir = os.path.join(args.data_root, "train")
    val_dir = os.path.join(args.data_root, "val")
    
    train_dataset = FalconDataset(
        os.path.join(train_dir, "Color_Images"),
        os.path.join(train_dir, "Segmentation"),
        transform=get_hrnet_transforms('train')
    )
    val_dataset = FalconDataset(
        os.path.join(val_dir, "Color_Images"),
        os.path.join(val_dir, "Segmentation"),
        transform=get_hrnet_transforms('val')
    )
    
    if args.batch_size < 2:
        print("Warning: Batch size must be at least 2 for BatchNorm. Setting batch_size to 2.")
        args.batch_size = 2

    # i5-13420H (8 Cores: 4P + 4E, 12 Threads) -> Use 6 workers to balance load without overhead
    # RTX 4050 6GB -> Keep batch_size modest (4-6) to avoid OOM
    # persistent_workers=True keeps workers alive between epochs
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=6, 
        pin_memory=True, 
        drop_last=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=6, 
        pin_memory=True,
        persistent_workers=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Model
    model = HRNetV2_Segmentation(num_classes=10).to(device)
    
    # Freeze early layers (Stem + Stage 1)
    model.freeze_early_layers()
    
    # Freeze BN layers for stability with small batch size
    model.apply(freeze_bn)
    
    # Optimizer & Scheduler
    # Only optimize parameters that require gradients
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Loss: Weighted CE + Dice with Dynamic Schedule
    # Initial Weights (Aggressive for Epochs 1-5):
    # Ground Clutter (4): 3.0, Dry Bushes (3): 1.5
    class_weights = torch.tensor([
        1.0, # 0
        1.0, # 1
        1.0, # 2
        1.5, # 3: Dry Bushes
        3.0, # 4: Ground Clutter
        1.0, # 5
        1.3, # 6: Logs
        2.6, # 7: Rocks
        1.0, # 8
        0.25 # 9: Sky
    ]).to(device)
    
    criterion = CombinedLoss(num_classes=10, weight=class_weights, ignore_index=255).to(device)
    
    # Load best checkpoint if available to resume/finetune
    best_model_path = "best_hrnet_model.pth"
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path} for fine-tuning...")
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint)
        # Reset optimizer for fine-tuning
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # AMP
    scaler = GradScaler()
    
    best_iou = 0.0
    
    for epoch in range(args.epochs):
        # Dynamic Loss Re-weighting Schedule
        if epoch == 5:
            print("\n>>> SCHEDULE UPDATE: Reducing class weights to prevent overfitting <<<")
            new_weights = torch.tensor([
                1.0, 1.0, 1.0, 
                1.2, # Dry Bushes: 1.5 -> 1.2
                2.0, # Ground Clutter: 3.0 -> 2.0
                1.0, 1.3, 2.6, 1.0, 0.25
            ]).to(device)
            # Update criterion weights
            criterion.weight = new_weights
            criterion.ce.weight = new_weights
            
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            with autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        scheduler.step()
        
        # Validation
        model.eval()
        total_intersections = np.zeros(10)
        total_unions = np.zeros(10)
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                images = images.to(device)
                masks = masks.to(device)
                
                with autocast(device_type="cuda"):
                    outputs = model(images)
                
                # Validation mIoU
                # Using argmax for metric computation
                preds = torch.argmax(outputs, dim=1)
                
                # Move to CPU for metric calculation, keep as tensor
                preds_cpu = preds.cpu()
                masks_cpu = masks.cpu()
                
                intersection, union = compute_intersection_union(preds_cpu, masks_cpu, num_classes=10, ignore_index=255)
                total_intersections += np.array(intersection)
                total_unions += np.array(union)
        
        iou_per_class = []
        for i in range(10):
            if total_unions[i] == 0:
                iou_per_class.append(np.nan)
            else:
                iou_per_class.append(total_intersections[i] / total_unions[i])
        
        iou_per_class = np.array(iou_per_class)
        mean_iou = np.nanmean(iou_per_class)
        
        print(f"Epoch {epoch+1} Mean IoU: {mean_iou:.4f}")
        print(f"Class IoUs: {iou_per_class}")
        
        if mean_iou > best_iou:
            best_iou = mean_iou
            torch.save(model.state_dict(), "best_hrnet_model.pth")
            print(f"Saved best model with mIoU: {best_iou:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=r"C:\Users\Mohammad\OneDrive\Desktop\Startathon\Offroad_Segmentation_Training_Dataset")
    parser.add_argument("--epochs", type=int, default=30)
    # Default batch_size=6 for 6GB VRAM + HRNet-W18 (384x384)
    # If OOM, reduce to 4
    parser.add_argument("--batch_size", type=int, default=10)
    args = parser.parse_args()
    
    train(args)
