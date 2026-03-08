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
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
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
        )  # [B, 576, 512]
        
        # Self-attention for audio
        audio_attended, _ = self.audio_self_attn(
            query=audio_proj,
            key=audio_proj,
            value=audio_proj
        )  # [B, 400, 512]
        
        # Mean pooling for sequence reduction (handles variable lengths)
        vision_pooled = vision_attended.mean(dim=1)  # [B, 512]
        audio_pooled = audio_attended.mean(dim=1)    # [B, 512]
        text_pooled = text_proj.mean(dim=1)          # [B, 512]
        
        # Concatenate for gating context
        concat_all = torch.cat([vision_pooled, audio_pooled, text_pooled], dim=-1)
        concat_all = concat_all.reshape(batch_size, -1)  # Ensure shape
        
        # Compute gates
        vision_gate = torch.sigmoid(self.gate_vision(torch.cat([vision_pooled, text_pooled], dim=-1)))
        audio_gate = torch.sigmoid(self.gate_audio(torch.cat([audio_pooled, text_pooled], dim=-1)))
        text_gate = torch.sigmoid(self.gate_text(torch.cat([text_pooled, vision_pooled], dim=-1)))
        
        # Apply gates
        vision_gated = vision_pooled * vision_gate
        audio_gated = audio_pooled * audio_gate
        text_gated = text_pooled * text_gate
        
        # Concatenate all gated features
        fused = torch.cat([vision_gated, audio_gated, text_gated], dim=-1)
        fused = fused.reshape(batch_size, -1)  # [B, 1536]
        
        # Final prediction
        output = self.fusion_mlp(fused)  # [B, 10]
        output = output.reshape(batch_size, 10)  # Ensure correct output shape
        
        return output