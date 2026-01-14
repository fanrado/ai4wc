import torch
import os
import torch.nn as nn

class nuViT(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=1000, 
                 dim=768, depth=12, heads=12, mlp_dim=3072, dropout=0.1):
        super(nuViT, self).__init__()
        
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # Transformer encoder
        ### Implements the original architecture described in the Attention is All you need paper.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, 
            nhead=heads, 
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Classification head
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, dim)
        
        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer(x)
        
        # Classification
        x = self.norm(x[:, 0])
        x = self.head(x)
        
        return x