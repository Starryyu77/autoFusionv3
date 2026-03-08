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
        self.common_dim = 512
        
        self.vision_proj = nn.Linear(self.vision_dim, self.common_dim)
        self.audio_proj = nn.Linear(self.audio_dim, self.common_dim)
        self.text_proj = nn.Linear(self.text_dim, self.common_dim)
        
        # Cross-attention: vision queries attend to text keys/values
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.common_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Self-attention for audio to capture temporal patterns
        self.audio_self_attn = nn.MultiheadAttention(
            embed_dim=self.common_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Gated fusion weights
        self.gate_vision = nn.Linear(self.common_dim * 2, self.common_dim)
        self.gate_audio = nn.Linear(self.common_dim * 2, self.common_dim)
        self.gate_text = nn.Linear(self.common_dim * 2, self.common_dim)
        
        # Final fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.common_dim * 3, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 10)
        )
        
        self.layer_norm_vision = nn.LayerNorm(self.common_dim)
        self.layer_norm_audio = nn.LayerNorm(self.common_dim)
        self.layer_norm_text = nn.LayerNorm(self.common_dim)
        
    def forward(self, vision, audio, text):
        batch_size = vision.shape[0]
        
        # Project modalities to common dimension
        vision_proj = self.vision_proj(vision)  # [B, 576, 512]
        audio_proj = self.audio_proj(audio)     # [B, 400, 512]
        text_proj = self.text_proj(text)        # [B, 77, 512]
        
        # Cross-attention: vision attends to text
        vision_attended, _ = self.cross_attn(
            query=vision_proj,
            key=text_proj,
            value=text_proj
        )
        vision_attended = self.layer_norm_vision(vision_proj + vision_attended)
        
        # Self-attention for audio
        audio_attended, _ = self.audio_self_attn(
            query=audio_proj,
            key=audio_proj,
            value=audio_proj
        )
        audio_attended = self.layer_norm_audio(audio_proj + audio_attended)
        
        # Mean pooling for variable-length sequences
        vision_pooled = vision_attended.mean(dim=1)  # [B, 512]
        audio_pooled = audio_attended.mean(dim=1)    # [B, 512]
        text_pooled = text_proj.mean(dim=1)          # [B, 512]
        
        # Gated fusion
        concat_features = torch.cat([vision_pooled, audio_pooled, text_pooled], dim=-1)
        
        # Compute gates
        gate_v = torch.sigmoid(self.gate_vision(torch.cat([vision_pooled, text_pooled], dim=-1)))
        gate_a = torch.sigmoid(self.gate_audio(torch.cat([audio_pooled, text_pooled], dim=-1)))
        gate_t = torch.sigmoid(self.gate_text(torch.cat([text_pooled, vision_pooled], dim=-1)))
        
        # Apply gates
        vision_gated = gate_v * vision_pooled
        audio_gated = gate_a * audio_pooled
        text_gated = gate_t * text_pooled
        
        # Concatenate gated features
        fused = torch.cat([vision_gated, audio_gated, text_gated], dim=-1)
        
        # Final prediction
        output = self.fusion_mlp(fused)
        
        # Ensure output shape using reshape
        output = output.reshape(batch_size, 10)
        
        return output