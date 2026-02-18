import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class HRNetV2_Segmentation(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()
        # Load HRNet-W18 from timm
        # features_only=True returns features at strides 2, 4, 8, 16, 32
        self.backbone = timm.create_model('hrnet_w18', pretrained=pretrained, features_only=True)
        
        # Based on timm hrnet_w18 output shapes: [64, 128, 256, 512, 1024]
        # We use strides 4, 8, 16, 32 (indices 1, 2, 3, 4)
        self.in_channels = [128, 256, 512, 1024] 
        self.fusion_channels = sum(self.in_channels)
        
        self.cls_head = nn.Sequential(
            nn.Conv2d(self.fusion_channels, self.fusion_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.fusion_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.fusion_channels, num_classes, kernel_size=1)
        )
        
    def freeze_early_layers(self):
        # Freeze Stem, Stage 1 ONLY
        # Stage 2 must be trainable for better features
        print("Freezing Stem and Stage 1...")
        
        for name, param in self.backbone.named_parameters():
            # Freeze Stem (conv1, bn1, conv2, bn2)
            if name.startswith('conv1') or name.startswith('bn1') or \
               name.startswith('conv2') or name.startswith('bn2'):
                param.requires_grad = False
            
            # Freeze Stage 1 (layer1)
            if name.startswith('layer1'):
                param.requires_grad = False
                
            # DO NOT FREEZE STAGE 2 (stage2, transition1)
            # DO NOT FREEZE TRANSITION 2
                
    def forward(self, x):
        features = self.backbone(x)
        # features list: [stride2, stride4, stride8, stride16, stride32]
        
        # We use stride 4, 8, 16, 32 (indices 1, 2, 3, 4)
        x0 = features[1] # Stride 4 (128 ch)
        x1 = features[2] # Stride 8 (256 ch)
        x2 = features[3] # Stride 16 (512 ch)
        x3 = features[4] # Stride 32 (1024 ch)
        
        # Upsample all to x0 size
        h, w = x0.size(2), x0.size(3)
        
        x1 = F.interpolate(x1, size=(h, w), mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, size=(h, w), mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, size=(h, w), mode='bilinear', align_corners=True)
        
        # Concatenate
        out = torch.cat([x0, x1, x2, x3], dim=1)
        
        # Classification head
        out = self.cls_head(out)
        
        # Upsample to original input size (Stride 4 -> Stride 1)
        out = F.interpolate(out, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
        
        return out
