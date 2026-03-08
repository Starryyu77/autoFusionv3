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
        
        # Self-attention for each modality
        self.vision_attn = nn.MultiheadAttention(self.fusion_dim, num_heads=8, batch_first=True)
        self.audio_attn = nn.MultiheadAttention(self.fusion_dim, num_heads=8, batch_first=True)
        self.text_attn = nn.MultiheadAttention(self.fusion_dim, num_heads=8, batch_first=True)
        
        # Cross-modal attention
        self.cross_attn = nn.MultiheadAttention(self.fusion_dim, num_heads=8, batch_first=True)
        
        # Gating mechanism
        self.gate_vision = nn.Linear(self.fusion_dim, 1)
        self.gate_audio = nn.Linear(self.fusion_dim, 1)
        self.gate_text = nn.Linear(self.fusion_dim, 1)
        
        # Final fusion layers
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.fusion_dim * 3, self.fusion_dim),
            nn.LayerNorm(self.fusion_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.fusion_dim, self.fusion_dim // 2),
            nn.LayerNorm(self.fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
        self.output_head = nn.Linear(self.fusion_dim // 2, 10)
        
        # Layer norms
        self.norm_vision = nn.LayerNorm(self.fusion_dim)
        self.norm_audio = nn.LayerNorm(self.fusion_dim)
        self.norm_text = nn.LayerNorm(self.fusion_dim)
        
    def forward(self, vision, audio, text):
        batch_size = vision.shape[0]
        
        # Project to common dimension
        vision_feat = self.vision_proj(vision)  # [B, 576, 512]
        audio_feat = self.audio_proj(audio)     # [B, 400, 512]
        text_feat = self.text_proj(text)        # [B, 77, 512]
        
        # Self-attention with mean pooling
        vision_attn, _ = self.vision_attn(vision_feat, vision_feat, vision_feat)
        vision_feat = self.norm_vision(vision_feat + vision_attn)
        vision_pooled = vision_feat.mean(dim=1)  # [B, 512]
        
        audio_attn, _ = self.audio_attn(audio_feat, audio_feat, audio_feat)
        audio_feat = self.norm_audio(audio_feat + audio_attn)
        audio_pooled = audio_feat.mean(dim=1)    # [B, 512]
        
        text_attn, _ = self.text_attn(text_feat, text_feat, text_feat)
        text_feat = self.norm_text(text_feat + text_attn)
        text_pooled = text_feat.mean(dim=1)      # [B, 512]
        
        # Cross-modal attention: each modality attends to others
        # Concatenate features for cross-attention
        combined = torch.cat([vision_feat, audio_feat, text_feat], dim=1)  # [B, 1053, 512]
        
        # Each pooled feature attends to combined features
        vision_cross, _ = self.cross_attn(
            vision_pooled.unsqueeze(1), combined, combined
        )
        audio_cross, _ = self.cross_attn(
            audio_pooled.unsqueeze(1), combined, combined
        )
        text_cross, _ = self.cross_attn(
            text_pooled.unsqueeze(1), combined, combined
        )
        
        vision_cross = vision_cross.squeeze(1)  # [B, 512]
        audio_cross = audio_cross.squeeze(1)    # [B, 512]
        text_cross = text_cross.squeeze(1)      # [B, 512]
        
        # Compute gating weights
        gate_v = torch.sigmoid(self.gate_vision(vision_cross))
        gate_a = torch.sigmoid(self.gate_audio(audio_cross))
        gate_t = torch.sigmoid(self.gate_text(text_cross))
        
        gate_sum = gate_v + gate_a + gate_t + 1e-8
        gate_v = gate_v / gate_sum
        gate_a = gate_a / gate_sum
        gate_t = gate_t / gate_sum
        
        # Apply gating
        vision_gated = vision_cross * gate_v
        audio_gated = audio_cross * gate_a
        text_gated = text_cross * gate_t
        
        # Concatenate gated features
        fused = torch.cat([vision_gated, audio_gated, text_gated], dim=-1)  # [B, 1536]
        
        # Final processing
        fused = self.fusion_mlp(fused)  # [B, 256]
        output = self.output_head(fused)  # [B, 10]
        
        return output