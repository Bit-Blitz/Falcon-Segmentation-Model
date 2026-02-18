import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
import os

def visualize_prediction(image, mask, pred, save_path=None):
    """
    image: (C, H, W) tensor or (H, W, C) numpy
    mask: (H, W) tensor or numpy
    pred: (H, W) tensor or numpy (class indices)
    """
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
        # Denormalize if normalized
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)
        
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
        
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
        
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(image)
    ax[0].set_title("Input Image")
    ax[0].axis('off')
    
    ax[1].imshow(mask, cmap='jet', vmin=0, vmax=9)
    ax[1].set_title("Ground Truth")
    ax[1].axis('off')
    
    ax[2].imshow(pred, cmap='jet', vmin=0, vmax=9)
    ax[2].set_title("Prediction")
    ax[2].axis('off')
    
    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)

def compute_intersection_union(pred, target, num_classes, ignore_index=255):
    pred = pred.view(-1)
    target = target.view(-1)
    
    # Filter out ignored pixels
    mask = target != ignore_index
    pred = pred[mask]
    target = target[mask]
    
    intersections = []
    unions = []
    
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        
        intersections.append(intersection)
        unions.append(union)
            
    return intersections, unions
