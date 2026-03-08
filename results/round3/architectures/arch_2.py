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
        
        # Self-attention for intra-modal processing
        self.vision_attn = nn.MultiheadAttention(512, num_heads=8, batch_first=True)
        self.audio_attn = nn.MultiheadAttention(512, num_heads=8, batch_first=True)
        self.text_attn = nn.MultiheadAttention(512, num_heads=8, batch_first=True)
        
        # Cross-modal attention for fusion
        self.cross_attn_vision = nn.MultiheadAttention(512, num_heads=8, batch_first=True)
        self.cross_attn_audio = nn.MultiheadAttention(512, num_heads=8, batch_first=True)
        self.cross_attn_text = nn.MultiheadAttention(512, num_heads=8, batch_first=True)
        
        # Gating mechanism for modality control
        self.gate_vision = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.gate_audio = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.gate_text = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Layer norms
        self.norm_vision = nn.LayerNorm(512)
        self.norm_audio = nn.LayerNorm(512)
        self.norm_text = nn.LayerNorm(512)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(512 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Output classifier
        self.classifier = nn.Linear(256, 10)
        
    def forward(self, vision, audio, text):
        batch_size = vision.shape[0]
        
        # Project to common dimension
        vision_feat = self.vision_proj(vision)  # [B, 576, 512]
        audio_feat = self.audio_proj(audio)     # [B, 400, 512]
        text_feat = self.text_proj(text)        # [B, 77, 512]
        
        # Intra-modal self-attention
        vision_attended, _ = self.vision_attn(vision_feat, vision_feat, vision_feat)
        audio_attended, _ = self.audio_attn(audio_feat, audio_feat, audio_feat)
        text_attended, _ = self.text_attn(text_feat, text_feat, text_feat)
        
        vision_feat = self.norm_vision(vision_feat + vision_attended)
        audio_feat = self.norm_audio(audio_feat + audio_attended)
        text_feat = self.norm_text(text_feat + text_attended)
        
        # Mean pooling for variable length sequences
        vision_pooled = vision_feat.mean(dim=1)  # [B, 512]
        audio_pooled = audio_feat.mean(dim=1)    # [B, 512]
        text_pooled = text_feat.mean(dim=1)      # [B, 512]
        
        # Compute gates
        gate_v = self.gate_vision(vision_pooled)  # [B, 1]
        gate_a = self.gate_audio(audio_pooled)    # [B, 1]
        gate_t = self.gate_text(text_pooled)      # [B, 1]
        
        # Apply gating
        vision_gated = vision_pooled * gate_v
        audio_gated = audio_pooled * gate_a
        text_gated = text_pooled * gate_t
        
        # Concatenate gated features
        fused = torch.cat([vision_gated, audio_gated, text_gated], dim=-1)  # [B, 1536]
        
        # Fusion processing
        fused = self.fusion(fused)  # [B, 256]
        
        # Output
        output = self.classifier(fused)  # [B, 10]
        
        return output