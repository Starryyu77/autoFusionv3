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
        self.fused_dim = 512
        
        # Projection layers to common dimension
        self.vision_proj = nn.Linear(self.vision_dim, self.fused_dim)
        self.audio_proj = nn.Linear(self.audio_dim, self.fused_dim)
        self.text_proj = nn.Linear(self.text_dim, self.fused_dim)
        
        # Layer norms for stability
        self.vision_ln = nn.LayerNorm(self.fused_dim)
        self.audio_ln = nn.LayerNorm(self.fused_dim)
        self.text_ln = nn.LayerNorm(self.fused_dim)
        
        # Cross-modal attention for fusion
        self.num_heads = 8
        self.cross_attn = nn.MultiheadAttention(self.fused_dim, self.num_heads, batch_first=True)
        
        # Self-attention for refined fusion
        self.self_attn = nn.MultiheadAttention(self.fused_dim, self.num_heads, batch_first=True)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(self.fused_dim, self.fused_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.fused_dim * 2, self.fused_dim),
            nn.Dropout(0.1)
        )
        
        # Gating mechanism for adaptive fusion
        self.gate = nn.Sequential(
            nn.Linear(self.fused_dim * 3, self.fused_dim),
            nn.Sigmoid()
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.fused_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 10)
        )
        
        # Residual projection for dimension matching
        self.residual_proj = nn.Linear(self.fused_dim * 3, self.fused_dim)
        
    def forward(self, vision, audio, text):
        batch_size = vision.shape[0]
        
        # Handle variable-length sequences with mean pooling
        # Vision: [B, 576, 1024] -> [B, 1024]
        vision_pooled = vision.mean(dim=1)  # [B, 1024]
        # Audio: [B, 400, 512] -> [B, 512]
        audio_pooled = audio.mean(dim=1)    # [B, 512]
        # Text: [B, 77, 768] -> [B, 768]
        text_pooled = text.mean(dim=1)      # [B, 768]
        
        # Project to common dimension
        v_feat = self.vision_ln(self.vision_proj(vision_pooled))  # [B, 512]
        a_feat = self.audio_ln(self.audio_proj(audio_pooled))     # [B, 512]
        t_feat = self.text_ln(self.text_proj(text_pooled))        # [B, 512]
        
        # Stack for attention: [B, 3, 512]
        stacked = torch.stack([v_feat, a_feat, t_feat], dim=1)    # [B, 3, 512]
        
        # Cross-modal attention
        attn_out, _ = self.cross_attn(stacked, stacked, stacked)  # [B, 3, 512]
        
        # Self-attention for refinement
        self_attn_out, _ = self.self_attn(attn_out, attn_out, attn_out)  # [B, 3, 512]
        
        # Residual connection around attention
        attn_residual = attn_out + self_attn_out  # [B, 3, 512]
        
        # Global pooling across modalities
        fused = attn_residual.mean(dim=1)  # [B, 512]
        
        # Feed-forward with residual
        ffn_out = self.ffn(fused)          # [B, 512]
        ffn_residual = fused + ffn_out     # [B, 512]
        
        # Gating mechanism
        concat_feats = torch.cat([v_feat, a_feat, t_feat], dim=-1)  # [B, 1536]
        gate_weights = self.gate(concat_feats)                      # [B, 512]
        
        # Apply gating
        gated = ffn_residual * gate_weights  # [B, 512]
        
        # Residual from raw features
        raw_residual = self.residual_proj(concat_feats)  # [B, 512]
        final_feat = gated + raw_residual                # [B, 512]
        
        # Final classification
        output = self.classifier(final_feat)  # [B, 10]
        
        return output