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
        
        # Self-attention for each modality
        self.vision_attn = nn.MultiheadAttention(512, num_heads=8, batch_first=True)
        self.audio_attn = nn.MultiheadAttention(512, num_heads=8, batch_first=True)
        self.text_attn = nn.MultiheadAttention(512, num_heads=8, batch_first=True)
        
        # Cross-modal attention for fusion
        self.cross_attn = nn.MultiheadAttention(512, num_heads=8, batch_first=True)
        
        # Gating mechanism for adaptive fusion
        self.gate_vision = nn.Linear(512, 512)
        self.gate_audio = nn.Linear(512, 512)
        self.gate_text = nn.Linear(512, 512)
        self.gate_fusion = nn.Linear(512 * 3, 3)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 10)
        )
        
        self.layer_norm = nn.LayerNorm(512)
        
    def forward(self, vision, audio, text):
        batch_size = vision.shape[0]
        
        # Project to common dimension
        vision_feat = self.vision_proj(vision)  # [B, 576, 512]
        audio_feat = self.audio_proj(audio)     # [B, 400, 512]
        text_feat = self.text_proj(text)        # [B, 77, 512]
        
        # Self-attention with mean pooling for variable-length sequences
        vision_attn_out, _ = self.vision_attn(vision_feat, vision_feat, vision_feat)
        vision_pooled = vision_attn_out.mean(dim=1)  # [B, 512]
        
        audio_attn_out, _ = self.audio_attn(audio_feat, audio_feat, audio_feat)
        audio_pooled = audio_attn_out.mean(dim=1)    # [B, 512]
        
        text_attn_out, _ = self.text_attn(text_feat, text_feat, text_feat)
        text_pooled = text_attn_out.mean(dim=1)      # [B, 512]
        
        # Apply layer normalization
        vision_pooled = self.layer_norm(vision_pooled)
        audio_pooled = self.layer_norm(audio_pooled)
        text_pooled = self.layer_norm(text_pooled)
        
        # Gating mechanism
        gate_input = torch.cat([vision_pooled, audio_pooled, text_pooled], dim=-1)
        gate_weights = F.softmax(self.gate_fusion(gate_input), dim=-1)  # [B, 3]
        
        # Reshape gate weights for broadcasting using reshape (not view)
        gw_vision = gate_weights[:, 0:1].reshape(batch_size, 1).expand(batch_size, 512)
        gw_audio = gate_weights[:, 1:2].reshape(batch_size, 1).expand(batch_size, 512)
        gw_text = gate_weights[:, 2:3].reshape(batch_size, 1).expand(batch_size, 512)
        
        # Apply gates
        vision_gated = vision_pooled * torch.sigmoid(self.gate_vision(vision_pooled)) * gw_vision
        audio_gated = audio_pooled * torch.sigmoid(self.gate_audio(audio_pooled)) * gw_audio
        text_gated = text_pooled * torch.sigmoid(self.gate_text(text_pooled)) * gw_text
        
        # Fusion: concatenated features passed through cross-attention
        stacked = torch.stack([vision_gated, audio_gated, text_gated], dim=1)  # [B, 3, 512]
        
        # Cross-modal attention
        fused, _ = self.cross_attn(stacked, stacked, stacked)
        fused = fused.mean(dim=1)  # [B, 512]
        
        # Final classification
        output = self.classifier(fused)  # [B, 10]
        
        return output