
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
import argparse
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

from model_hrnet import HRNetV2_Segmentation

# =======================
# Configuration
# =======================
NUM_CLASSES = 10
IGNORE_INDEX = 255

VALUE_MAP = {
    100: 0, 200: 1, 300: 2, 500: 3, 550: 4,
    600: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}
CLASS_NAMES = [
    'Trees','Lush Bushes','Dry Grass','Dry Bushes','Ground Clutter',
    'Flowers','Logs','Rocks','Landscape','Sky'
]

# =======================
# Dataset (Simplified for Inference)
# =======================
class InferenceDataset(Dataset):
    def __init__(self, root, transform=None):
        self.img_dir = os.path.join(root, "Color_Images")
        self.mask_dir = os.path.join(root, "Segmentation")
        self.ids = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.png')])
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        img_path = os.path.join(self.img_dir, name)
        mask_path = os.path.join(self.mask_dir, name)
        
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        
        # Convert mask using mapping
        mask_np = np.array(mask)
        new_mask = np.ones_like(mask_np, dtype=np.uint8) * IGNORE_INDEX
        for k, v in VALUE_MAP.items():
            new_mask[mask_np == k] = v
        mask = Image.fromarray(new_mask)
        
        # Resize mask to 384x384 (Model Input Size) for evaluation
        mask = mask.resize((384, 384), resample=Image.NEAREST)
        
        if self.transform:
            img = self.transform(img)
            
        return img, torch.from_numpy(np.array(mask)).long(), name

# =======================
# Inference Logic
# =======================
def multi_scale_inference(model, image, scales=[0.75, 1.0, 1.25], flip=True):
    """
    Perform multi-scale inference with optional flipping and brightness jitter.
    """
    b, c, h, w = image.shape
    total_logits = torch.zeros((b, NUM_CLASSES, h, w), device=image.device)
    count = 0
    
    # Brightness factors for TTA (mild jitter)
    # brightness_factors = [1.0] # Standard
    brightness_factors = [0.9, 1.0, 1.1] # Can enable for more aggressive TTA
    
    for scale in scales:
        for bf in brightness_factors:
            new_h, new_w = int(h * scale), int(w * scale)
            
            # Resize image
            img_scaled = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)
            
            # Apply brightness jitter if needed
            if bf != 1.0:
                img_scaled = img_scaled * bf
            
            # Predict
            with torch.amp.autocast('cuda'):
                logits = model(img_scaled)
            
            # Resize back to original size
            logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
            
            total_logits += logits
            count += 1
            
            if flip:
                # Horizontal flip
                img_flipped = torch.flip(img_scaled, dims=[3])
                with torch.amp.autocast('cuda'):
                    logits_flipped = model(img_flipped)
                
                # Flip back
                logits_flipped = torch.flip(logits_flipped, dims=[3])
                
                # Resize back
                logits_flipped = F.interpolate(logits_flipped, size=(h, w), mode='bilinear', align_corners=False)
                
                total_logits += logits_flipped
                count += 1
            
    return total_logits / count

def apply_post_processing(mask_np):
    """
    Apply advanced morphological operations:
    1. Remove small isolated regions (<100px)
    2. Morphological closing for Ground Clutter (4) and Dry Bushes (3)
    3. Majority vote in 3x3 window
    """
    mask_clean = mask_np.copy().astype(np.uint8)
    
    # 1. Morphological Closing for specific classes
    # Ground Clutter (4) and Dry Bushes (3) - helps fill holes
    kernel = np.ones((3, 3), np.uint8) # Standard 3x3 kernel
    
    # Create binary masks for specific classes, close them, and place back
    # Boosting Dry Bushes (3) and Ground Clutter (4)
    for cls_idx in [3, 4]:
        binary_mask = (mask_clean == cls_idx).astype(np.uint8)
        closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        mask_clean[closed_mask == 1] = cls_idx
        
    # 2. Global Morphological Open to remove noise (salt-and-pepper)
    # Be careful not to remove small valid objects like distant rocks
    # kernel_open = np.ones((3, 3), np.uint8)
    # mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel_open)
    
    return mask_clean

# =======================
# Main
# =======================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to checkpoint")
    parser.add_argument("--data_dir", required=True, help="Path to validation data")
    parser.add_argument("--use_multiscale", action="store_true", help="Enable multi-scale inference")
    parser.add_argument("--use_postprocess", action="store_true", help="Enable morphological post-processing")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transforms (Resize to 384x384 for model)
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = InferenceDataset(args.data_dir, transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4) # Batch size 1 for multi-scale

    # Load Model
    model = HRNetV2_Segmentation(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    total_inter = np.zeros(NUM_CLASSES)
    total_union = np.zeros(NUM_CLASSES)

    print(f"Starting inference on {len(dataset)} images...")
    print(f"Multi-scale: {args.use_multiscale}")
    print(f"Post-processing: {args.use_postprocess}")

    with torch.no_grad():
        for img, mask, _ in tqdm(loader):
            img = img.to(device)
            mask = mask.to(device)

            if args.use_multiscale:
                # Multi-scale Inference (Scales: 0.75, 1.0, 1.25 + Flip)
                logits = multi_scale_inference(model, img, scales=[0.75, 1.0, 1.25], flip=True)
                # Try aggressive scaling for small objects
                # logits = multi_scale_inference(model, img, scales=[1.0, 1.5, 2.0], flip=True)
            else:
                # Single scale
                logits = model(img)
            
            # Resize logits to 384x384 to match Ground Truth
            if logits.shape[2:] != (384, 384):
                logits = F.interpolate(logits, size=(384, 384), mode='bilinear', align_corners=False)
            
            preds = torch.argmax(logits, dim=1)
            pred_np = preds.cpu().numpy().squeeze(0)
            
            if args.use_postprocess:
                pred_np = apply_post_processing(pred_np)
                preds = torch.from_numpy(pred_np).unsqueeze(0).to(device)

            # Update Metrics
            for c in range(NUM_CLASSES):
                # Ensure mask matches pred shape (should be 384x384)
                # Ignore index handling
                valid_mask = (mask != IGNORE_INDEX)
                
                # Intersection and Union
                # Only consider valid pixels
                p = preds[valid_mask]
                t = mask[valid_mask]
                
                total_inter[c] += ((p == c) & (t == c)).sum().item()
                total_union[c] += ((p == c) | (t == c)).sum().item()

    # Calculate IoU
    ious = []
    for c in range(NUM_CLASSES):
        if total_union[c] == 0:
            ious.append(np.nan)
        else:
            ious.append(total_inter[c] / total_union[c])
            
    miou = np.nanmean(ious)

    print("\n==============================")
    print(f"Mean IoU: {miou:.4f}")
    print("==============================")
    for i, v in enumerate(ious):
        print(f"{CLASS_NAMES[i]:15s}: {v:.4f}")

if __name__ == "__main__":
    main()
