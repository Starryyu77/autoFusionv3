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
        
        # Cross-modal fusion attention
        self.fusion_attn = nn.MultiheadAttention(self.fused_dim, num_heads=8, batch_first=True)
        
        # Gating mechanism for adaptive fusion
        self.gate_vision = nn.Linear(self.fused_dim, self.fused_dim)
        self.gate_audio = nn.Linear(self.fused_dim, self.fused_dim)
        self.gate_text = nn.Linear(self.fused_dim, self.fused_dim)
        
        # Output layers
        self.fusion_fc = nn.Sequential(
            nn.Linear(self.fused_dim * 3, self.fused_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.fused_dim, self.fused_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.fused_dim // 2, 10)
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
        vision_attn, _ = self.vision_attn(vision_proj, vision_proj, vision_proj)
        audio_attn, _ = self.audio_attn(audio_proj, audio_proj, audio_proj)
        text_attn, _ = self.text_attn(text_proj, text_proj, text_proj)
        
        # Residual connections
        vision_feat = vision_proj + vision_attn
        audio_feat = audio_proj + audio_attn
        text_feat = text_proj + text_attn
        
        # Mean pooling for variable-length sequences
        vision_pooled = vision_feat.mean(dim=1)  # [B, 512]
        audio_pooled = audio_feat.mean(dim=1)    # [B, 512]
        text_pooled = text_feat.mean(dim=1)      # [B, 512]
        
        # Gating mechanism
        gate_v = torch.sigmoid(self.gate_vision(vision_pooled))
        gate_a = torch.sigmoid(self.gate_audio(audio_pooled))
        gate_t = torch.sigmoid(self.gate_text(text_pooled))
        
        # Apply gates
        vision_gated = vision_pooled * gate_v
        audio_gated = audio_pooled * gate_a
        text_gated = text_pooled * gate_t
        
        # Concatenate all modalities
        # Use reshape instead of view as required
        multimodal = torch.cat([vision_gated, audio_gated, text_gated], dim=-1)
        multimodal = multimodal.reshape(batch_size, -1)  # [B, 1536]
        
        # Final prediction
        output = self.fusion_fc(multimodal)  # [B, 10]
        
        return output