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
        
        # Self-attention for each modality (handle variable length)
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
            nn.Linear(256, 512),
            nn.Sigmoid()
        )
        self.gate_audio = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.Sigmoid()
        )
        self.gate_text = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.Sigmoid()
        )
        
        # Fusion layers
        self.fusion_norm = nn.LayerNorm(512 * 3)
        self.fusion_fc = nn.Sequential(
            nn.Linear(512 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Output head
        self.output_fc = nn.Linear(256, 10)
        
        # Layer norms
        self.norm_vision = nn.LayerNorm(512)
        self.norm_audio = nn.LayerNorm(512)
        self.norm_text = nn.LayerNorm(512)
        
    def forward(self, vision, audio, text):
        batch_size = vision.shape[0]
        
        # Project to common dimension
        vision_feat = self.vision_proj(vision)  # [2, 576, 512]
        audio_feat = self.audio_proj(audio)     # [2, 400, 512]
        text_feat = self.text_proj(text)        # [2, 77, 512]
        
        # Self-attention for each modality (handle variable length)
        vision_feat, _ = self.vision_attn(vision_feat, vision_feat, vision_feat)
        vision_feat = self.norm_vision(vision_feat)
        
        audio_feat, _ = self.audio_attn(audio_feat, audio_feat, audio_feat)
        audio_feat = self.norm_audio(audio_feat)
        
        text_feat, _ = self.text_attn(text_feat, text_feat, text_feat)
        text_feat = self.norm_text(text_feat)
        
        # Mean pooling to get fixed-size representations
        vision_pooled = vision_feat.mean(dim=1)  # [2, 512]
        audio_pooled = audio_feat.mean(dim=1)    # [2, 512]
        text_pooled = text_feat.mean(dim=1)      # [2, 512]
        
        # Reshape for cross-modal attention
        vision_seq = vision_pooled.unsqueeze(1)  # [2, 1, 512]
        audio_seq = audio_pooled.unsqueeze(1)    # [2, 1, 512]
        text_seq = text_pooled.unsqueeze(1)      # [2, 1, 512]
        
        # Cross-modal fusion
        vision_cross, _ = self.cross_attn_vision(vision_seq, audio_seq, text_seq)
        audio_cross, _ = self.cross_attn_audio(audio_seq, text_seq, vision_seq)
        text_cross, _ = self.cross_attn_text(text_seq, vision_seq, audio_seq)
        
        # Reshape back
        vision_cross = vision_cross.reshape(batch_size, -1)  # [2, 512]
        audio_cross = audio_cross.reshape(batch_size, -1)    # [2, 512]
        text_cross = text_cross.reshape(batch_size, -1)      # [2, 512]
        
        # Gating mechanism
        vision_gate = self.gate_vision(vision_cross)
        audio_gate = self.gate_audio(audio_cross)
        text_gate = self.gate_text(text_cross)
        
        # Apply gates
        vision_gated = vision_cross * vision_gate
        audio_gated = audio_cross * audio_gate
        text_gated = text_cross * text_gate
        
        # Concatenate all modalities
        fused = torch.cat([vision_gated, audio_gated, text_gated], dim=-1)  # [2, 1536]
        
        # Fusion processing
        fused = self.fusion_norm(fused)
        fused = self.fusion_fc(fused)  # [2, 256]
        
        # Output
        output = self.output_fc(fused)  # [2, 10]
        
        return output