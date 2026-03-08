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
        
        # Multi-head self-attention for intra-modal processing
        self.vision_attn = nn.MultiheadAttention(self.fused_dim, num_heads=8, batch_first=True)
        self.audio_attn = nn.MultiheadAttention(self.fused_dim, num_heads=8, batch_first=True)
        self.text_attn = nn.MultiheadAttention(self.fused_dim, num_heads=8, batch_first=True)
        
        # Cross-modal attention for fusion
        self.cross_attn_vision = nn.MultiheadAttention(self.fused_dim, num_heads=8, batch_first=True)
        self.cross_attn_audio = nn.MultiheadAttention(self.fused_dim, num_heads=8, batch_first=True)
        self.cross_attn_text = nn.MultiheadAttention(self.fused_dim, num_heads=8, batch_first=True)
        
        # Gating mechanism for adaptive fusion
        self.gate_vision = nn.Sequential(
            nn.Linear(self.fused_dim * 3, self.fused_dim),
            nn.Sigmoid()
        )
        self.gate_audio = nn.Sequential(
            nn.Linear(self.fused_dim * 3, self.fused_dim),
            nn.Sigmoid()
        )
        self.gate_text = nn.Sequential(
            nn.Linear(self.fused_dim * 3, self.fused_dim),
            nn.Sigmoid()
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.fused_dim * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 10)
        )
        
    def forward(self, vision, audio, text):
        batch_size = vision.shape[0]
        
        # Project to common dimension
        vision_proj = self.vision_proj(vision)  # [B, 576, 512]
        audio_proj = self.audio_proj(audio)     # [B, 400, 512]
        text_proj = self.text_proj(text)        # [B, 77, 512]
        
        # Apply layer norm
        vision_proj = self.vision_norm(vision_proj)
        audio_proj = self.audio_norm(audio_proj)
        text_proj = self.text_norm(text_proj)
        
        # Intra-modal self-attention
        vision_feat, _ = self.vision_attn(vision_proj, vision_proj, vision_proj)
        audio_feat, _ = self.audio_attn(audio_proj, audio_proj, audio_proj)
        text_feat, _ = self.text_attn(text_proj, text_proj, text_proj)
        
        # Mean pooling for variable-length sequences
        vision_pooled = vision_feat.mean(dim=1)  # [B, 512]
        audio_pooled = audio_feat.mean(dim=1)    # [B, 512]
        text_pooled = text_feat.mean(dim=1)      # [B, 512]
        
        # Expand for cross-attention
        vision_exp = vision_pooled.unsqueeze(1)  # [B, 1, 512]
        audio_exp = audio_pooled.unsqueeze(1)    # [B, 1, 512]
        text_exp = text_pooled.unsqueeze(1)      # [B, 1, 512]
        
        # Cross-modal attention
        vision_cross, _ = self.cross_attn_vision(
            vision_exp, 
            torch.cat([audio_exp, text_exp], dim=1),
            torch.cat([audio_exp, text_exp], dim=1)
        )
        audio_cross, _ = self.cross_attn_audio(
            audio_exp,
            torch.cat([vision_exp, text_exp], dim=1),
            torch.cat([vision_exp, text_exp], dim=1)
        )
        text_cross, _ = self.cross_attn_text(
            text_exp,
            torch.cat([vision_exp, audio_exp], dim=1),
            torch.cat([vision_exp, audio_exp], dim=1)
        )
        
        # Squeeze back
        vision_cross = vision_cross.squeeze(1)  # [B, 512]
        audio_cross = audio_cross.squeeze(1)    # [B, 512]
        text_cross = text_cross.squeeze(1)      # [B, 512]
        
        # Concatenate for gating
        concat_all = torch.cat([vision_cross, audio_cross, text_cross], dim=-1)  # [B, 1536]
        
        # Adaptive gating
        gate_v = self.gate_vision(concat_all)
        gate_a = self.gate_audio(concat_all)
        gate_t = self.gate_text(concat_all)
        
        # Apply gates
        vision_gated = gate_v * vision_cross
        audio_gated = gate_a * audio_cross
        text_gated = gate_t * text_cross
        
        # Final fusion
        fused = torch.cat([vision_gated, audio_gated, text_gated], dim=-1)  # [B, 1536]
        
        # Reshape for safety (though not strictly needed here)
        fused = fused.reshape(batch_size, -1)
        
        # Classify
        output = self.classifier(fused)  # [B, 10]
        
        # Final reshape to ensure correct output shape
        output = output.reshape(batch_size, 10)
        
        return output