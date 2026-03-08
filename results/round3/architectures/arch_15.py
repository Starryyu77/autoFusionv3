import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalFusion(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Feature dimensions
        self.vision_dim = 1024
        self.audio_dim = 512
        self.text_dim = 768
        
        # Project all modalities to common dimension
        self.fusion_dim = 512
        
        self.vision_proj = nn.Linear(self.vision_dim, self.fusion_dim)
        self.audio_proj = nn.Linear(self.audio_dim, self.fusion_dim)
        self.text_proj = nn.Linear(self.text_dim, self.fusion_dim)
        
        # Layer norms for stability
        self.vision_ln = nn.LayerNorm(self.fusion_dim)
        self.audio_ln = nn.LayerNorm(self.fusion_dim)
        self.text_ln = nn.LayerNorm(self.fusion_dim)
        
        # Cross-modal attention for fusion
        self.num_heads = 8
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.fusion_dim,
            num_heads=self.num_heads,
            batch_first=True
        )
        
        # Gating mechanism for adaptive fusion
        self.gate_proj = nn.Sequential(
            nn.Linear(self.fusion_dim * 3, self.fusion_dim),
            nn.ReLU(),
            nn.Linear(self.fusion_dim, 3),
            nn.Softmax(dim=-1)
        )
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 10)
        )
        
    def forward(self, vision, audio, text):
        batch_size = vision.shape[0]
        
        # Mean pooling for variable-length sequences
        vision_pooled = vision.mean(dim=1)  # [B, 1024]
        audio_pooled = audio.mean(dim=1)    # [B, 512]
        text_pooled = text.mean(dim=1)      # [B, 768]
        
        # Project to common dimension
        vision_feat = self.vision_ln(self.vision_proj(vision_pooled))  # [B, 512]
        audio_feat = self.audio_ln(self.audio_proj(audio_pooled))      # [B, 512]
        text_feat = self.text_ln(self.text_proj(text_pooled))          # [B, 512]
        
        # Stack for cross-modal attention: [B, 3, 512]
        stacked = torch.stack([vision_feat, audio_feat, text_feat], dim=1)
        
        # Self-attention across modalities
        attn_out, _ = self.cross_attn(stacked, stacked, stacked)  # [B, 3, 512]
        
        # Global average pooling across modalities
        attn_pooled = attn_out.mean(dim=1)  # [B, 512]
        
        # Compute adaptive gates based on original features
        concat_feats = torch.cat([vision_feat, audio_feat, text_feat], dim=-1)  # [B, 1536]
        gates = self.gate_proj(concat_feats)  # [B, 3]
        
        # Reshape gates for broadcasting: [B, 3, 1]
        gates = gates.reshape(batch_size, 3, 1)
        
        # Weighted combination using attention output
        weighted = (attn_out * gates).sum(dim=1)  # [B, 512]
        
        # Residual connection with pooled attention
        fused = weighted + attn_pooled  # [B, 512]
        
        # Final classification
        output = self.classifier(fused)  # [B, 10]
        
        return output