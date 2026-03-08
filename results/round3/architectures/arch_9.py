import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalFusion(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Project all modalities to common dimension
        self.vision_proj = nn.Linear(1024, 512)
        self.audio_proj = nn.Linear(512, 512)
        self.text_proj = nn.Linear(768, 512)
        
        # Layer norms for stability
        self.vision_ln = nn.LayerNorm(512)
        self.audio_ln = nn.LayerNorm(512)
        self.text_ln = nn.LayerNorm(512)
        
        # Cross-modal attention for fusion
        self.fusion_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        
        # Gating mechanism for adaptive fusion
        self.gate = nn.Sequential(
            nn.Linear(512 * 3, 512),
            nn.Sigmoid()
        )
        
        # Fusion projection
        self.fusion_proj = nn.Sequential(
            nn.Linear(512 * 3, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Output head with residual
        self.output_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 10)
        )
        
        # Residual projection
        self.residual_proj = nn.Linear(512, 10)
        
    def forward(self, vision, audio, text):
        batch_size = vision.shape[0]
        
        # Mean pooling to handle variable-length sequences
        vision_pooled = vision.mean(dim=1)  # [B, 1024]
        audio_pooled = audio.mean(dim=1)    # [B, 512]
        text_pooled = text.mean(dim=1)      # [B, 768]
        
        # Project to common dimension
        v = self.vision_ln(self.vision_proj(vision_pooled))  # [B, 512]
        a = self.audio_ln(self.audio_proj(audio_pooled))     # [B, 512]
        t = self.text_ln(self.text_proj(text_pooled))        # [B, 512]
        
        # Stack for cross-modal attention
        # Reshape for attention: [B, 3, 512]
        stacked = torch.stack([v, a, t], dim=1)  # [B, 3, 512]
        
        # Self-attention across modalities
        attn_out, _ = self.fusion_attn(stacked, stacked, stacked)  # [B, 3, 512]
        
        # Reshape attention output: flatten modalities
        attn_flat = attn_out.reshape(batch_size, -1)  # [B, 1536]
        
        # Compute adaptive gate
        concat_features = torch.cat([v, a, t], dim=-1)  # [B, 1536]
        gate_weights = self.gate(concat_features)  # [B, 512]
        
        # Fusion with gating
        fused = self.fusion_proj(attn_flat)  # [B, 512]
        fused = fused * gate_weights  # Apply gating
        
        # Residual connection from pooled features
        residual_input = (v + a + t) / 3.0  # Simple average residual
        
        # Final output with residual
        output = self.output_head(fused)  # [B, 10]
        residual = self.residual_proj(residual_input)  # [B, 10]
        
        return output + residual