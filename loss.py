import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    def __init__(self, num_classes=10, weight=None, ignore_index=255):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.weight = weight
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def dice_loss(self, pred, target):
        # pred: (B, C, H, W) logits
        # target: (B, H, W)
        
        pred_softmax = F.softmax(pred, dim=1)
        
        # Create one-hot target
        # Handle ignore_index by clamping/masking
        target_clamped = target.clone()
        if self.ignore_index is not None:
            target_clamped[target == self.ignore_index] = 0
            
        target_one_hot = F.one_hot(target_clamped, num_classes=self.num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        
        # Mask out ignore_index
        if self.ignore_index is not None:
            mask = (target != self.ignore_index).float().unsqueeze(1)
            target_one_hot = target_one_hot * mask
            pred_softmax = pred_softmax * mask
            
        smooth = 1e-5
        intersection = (pred_softmax * target_one_hot).sum(dim=(2, 3))
        union = pred_softmax.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice_score = (2. * intersection + smooth) / (union + smooth)
        
        # Apply class weights to Dice score if provided
        if self.weight is not None:
            # Normalize weights to sum to num_classes so the scale of loss is similar
            # We want higher weight -> higher loss -> lower dice score contribution?
            # Dice loss is 1 - dice. We want to penalize low dice more for weighted classes.
            # Loss = weight * (1 - dice)
            w = self.weight / self.weight.mean() # Normalize around 1
            dice_loss = 1 - dice_score
            weighted_dice_loss = (dice_loss * w).mean()
            return weighted_dice_loss
            
        return 1 - dice_score.mean()

    def forward(self, pred, target):
        loss_ce = self.ce(pred, target)
        loss_dice = self.dice_loss(pred, target)
        return loss_ce + loss_dice
