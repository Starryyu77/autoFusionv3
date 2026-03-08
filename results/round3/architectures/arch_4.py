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
            nn.Linear(self.fused_dim * 3, self.fused_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.fused_dim * 2, self.fused_dim),
            nn.Dropout(0.1)
        )
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(self.fused_dim * 3, 3),
            nn.Softmax(dim=-1)
        )
        
        # Output head
        self.output_proj = nn.Sequential(
            nn.LayerNorm(self.fused_dim),
            nn.Linear(self.fused_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 10)
        )
        
        # Residual projections
        self.residual_proj = nn.Linear(self.fused_dim * 3, self.fused_dim)
        
    def forward(self, vision, audio, text):
        batch_size = vision.shape[0]
        
        # Mean pooling for variable-length sequences
        # Vision: [B, 576, 1024] -> [B, 1024]
        v_pooled = vision.mean(dim=1)
        # Audio: [B, 400, 512] -> [B, 512]
        a_pooled = audio.mean(dim=1)
        # Text: [B, 77, 768] -> [B, 768]
        t_pooled = text.mean(dim=1)
        
        # Project to common dimension
        v_feat = self.vision_ln(self.vision_proj(v_pooled))  # [B, 512]
        a_feat = self.audio_ln(self.audio_proj(a_pooled))    # [B, 512]
        t_feat = self.text_ln(self.text_proj(t_pooled))      # [B, 512]
        
        # Stack for attention: [B, 3, 512]
        stacked = torch.stack([v_feat, a_feat, t_feat], dim=1)
        
        # Cross-modal attention
        attn_out, _ = self.cross_attn(stacked, stacked, stacked)  # [B, 3, 512]
        
        # Self-attention refinement
        refined, _ = self.self_attn(attn_out, attn_out, attn_out)  # [B, 3, 512]
        
        # Add residual
        refined = refined + attn_out
        
        # Flatten: [B, 3, 512] -> [B, 1536] using reshape
        flat = refined.reshape(batch_size, -1)  # [B, 1536]
        
        # FFN
        ffn_out = self.ffn(flat)  # [B, 512]
        
        # Compute gates
        gates = self.gate(flat)  # [B, 3]
        
        # Weighted combination using einsum
        weighted = torch.einsum('bn,bnd->bd', gates, refined)  # [B, 512]
        
        # Residual connection
        residual = self.residual_proj(flat)  # [B, 512]
        fused = weighted + residual + ffn_out  # [B, 512]
        
        # Final normalization and output
        output = self.output_proj(fused)  # [B, 10]
        
        return output