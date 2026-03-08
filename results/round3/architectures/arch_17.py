import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalFusion(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Fixed dimensions
        self.vision_dim = 1024
        self.audio_dim = 512
        self.text_dim = 768
        self.fused_dim = 512
        
        # Projection layers to common dimension
        self.vision_proj = nn.Linear(self.vision_dim, self.fused_dim)
        self.audio_proj = nn.Linear(self.audio_dim, self.fused_dim)
        self.text_proj = nn.Linear(self.text_dim, self.fused_dim)
        
        # Layer norms for stability
        self.vision_norm = nn.LayerNorm(self.fused_dim)
        self.audio_norm = nn.LayerNorm(self.fused_dim)
        self.text_norm = nn.LayerNorm(self.fused_dim)
        
        # Gating mechanism for modality control
        self.gate_proj = nn.Sequential(
            nn.Linear(self.fused_dim * 3, 3),
            nn.Softmax(dim=-1)
        )
        
        # Cross-modal attention
        self.cross_attn = nn.MultiheadAttention(self.fused_dim, num_heads=8, batch_first=True)
        
        # Self-attention for fused representation
        self.self_attn = nn.MultiheadAttention(self.fused_dim, num_heads=8, batch_first=True)
        
        # Final fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 10)
        )
        
    def forward(self, vision, audio, text):
        batch_size = vision.shape[0]
        
        # Mean pooling to handle variable-length sequences
        vision_pooled = vision.mean(dim=1)  # [B, 1024]
        audio_pooled = audio.mean(dim=1)    # [B, 512]
        text_pooled = text.mean(dim=1)      # [B, 768]
        
        # Project to common dimension
        vision_feat = self.vision_norm(self.vision_proj(vision_pooled))  # [B, 512]
        audio_feat = self.audio_norm(self.audio_proj(audio_pooled))      # [B, 512]
        text_feat = self.text_norm(self.text_proj(text_pooled))          # [B, 512]
        
        # Stack features for cross-modal attention: [B, 3, 512]
        stacked = torch.stack([vision_feat, audio_feat, text_feat], dim=1)
        
        # Cross-modal attention
        attn_out, _ = self.cross_attn(stacked, stacked, stacked)  # [B, 3, 512]
        
        # Compute gating weights based on pooled features
        concat_feats = torch.cat([vision_feat, audio_feat, text_feat], dim=-1)  # [B, 1536]
        gate_weights = self.gate_proj(concat_feats)  # [B, 3]
        
        # Apply gating: weighted combination of attended features
        # Reshape gate_weights for broadcasting: [B, 3, 1]
        gate_weights = gate_weights.reshape(batch_size, 3, 1)
        gated = attn_out * gate_weights  # [B, 3, 512]
        
        # Sum pooled representation
        fused = gated.sum(dim=1)  # [B, 512]
        
        # Self-attention refinement (treat as sequence of 1)
        fused_seq = fused.reshape(batch_size, 1, self.fused_dim)  # [B, 1, 512]
        refined, _ = self.self_attn(fused_seq, fused_seq, fused_seq)  # [B, 1, 512]
        refined = refined.reshape(batch_size, self.fused_dim)  # [B, 512]
        
        # Residual connection
        refined = refined + fused
        
        # Final output
        output = self.fusion_mlp(refined)  # [B, 10]
        
        return output