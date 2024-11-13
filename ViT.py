import torch
import torch.nn as nn
import torch.nn.functional as F
import math 

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        assert H == self.image_size and W == self.image_size, f"Input image size ({H}*{W}) doesn't match model image size ({self.image_size}*{self.image_size})"

        x = self.projection(x).flatten(2).transpose(1, 2)   # [B, embed_dim, num_patches]
        
        return x

class PositionEmbedding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super(PositionEmbedding, self).__init__()
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
    
    def forward(self, x):
        return x + self.position_embeddings

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # 多头自注意力
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # 前馈网络
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, num_heads=8, 
                 mlp_dim=2048, num_layers=12, num_classes=1000, dropout=0.1):
        super(VisionTransformer, self).__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.position_embedding = PositionEmbedding(self.patch_embedding.num_patches, embed_dim)
        
        self.transformer_encoders = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads, mlp_dim, dropout) for _ in range(num_layers)
        ])
        
        # 分类标识符（cls token），通常用于图像分类任务
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
    
    def forward(self, x):
        # Patch Embedding
        x = self.patch_embedding(x)  # [B, num_patches, embed_dim]
        
        # 添加分类标识符
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, num_patches + 1, embed_dim]
        
        # 添加位置嵌入
        x = self.position_embedding(x)
        x = self.dropout(x)
        
        # 通过多层Transformer Encoder
        for encoder in self.transformer_encoders:
            x = encoder(x)
        
        # 输出分类标识符的表示
        x = x[:, 0]  # [B, embed_dim]
        print(x.shape)
        # x = self.mlp_head(x)  # [B, num_classes]
        return x
