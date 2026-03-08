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
        
        # Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.common_dim * 3, self.common_dim),
            nn.LayerNorm(self.common_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.common_dim, self.common_dim // 2),
            nn.LayerNorm(self.common_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Output classifier
        self.classifier = nn.Linear(self.common_dim // 2, 10)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, vision, audio, text):
        batch_size = vision.shape[0]
        
        # Project to common dimension
        vision_proj = self.vision_proj(vision)  # [B, 576, 512]
        audio_proj = self.audio_proj(audio)      # [B, 400, 512]
        text_proj = self.text_proj(text)         # [B, 77, 512]
        
        # Cross-attention: vision attends to text
        vision_cross, _ = self.cross_attn(
            query=vision_proj,
            key=text_proj,
            value=text_proj
        )  # [B, 576, 512]
        
        # Self-attention for audio
        audio_self, _ = self.audio_self_attn(
            query=audio_proj,
            key=audio_proj,
            value=audio_proj
        )  # [B, 400, 512]
        
        # Mean pooling for variable-length sequences
        vision_pooled = vision_cross.mean(dim=1)  # [B, 512]
        audio_pooled = audio_self.mean(dim=1)     # [B, 512]
        text_pooled = text_proj.mean(dim=1)       # [B, 512]
        
        # Concatenate for gating
        concat_features = torch.cat([vision_pooled, audio_pooled, text_pooled], dim=-1)
        concat_features = concat_features.reshape(batch_size, -1)  # Ensure shape
        
        # Gated fusion
        gate_input_v = torch.cat([vision_pooled, text_pooled], dim=-1)
        gate_input_a = torch.cat([audio_pooled, text_pooled], dim=-1)
        gate_input_t = torch.cat([text_pooled, vision_pooled], dim=-1)
        
        gate_v = torch.sigmoid(self.gate_vision(gate_input_v))
        gate_a = torch.sigmoid(self.gate_audio(gate_input_a))
        gate_t = torch.sigmoid(self.gate_text(gate_input_t))
        
        # Apply gates
        vision_gated = gate_v * vision_pooled
        audio_gated = gate_a * audio_pooled
        text_gated = gate_t * text_pooled
        
        # Concatenate all gated features
        fused = torch.cat([vision_gated, audio_gated, text_gated], dim=-1)
        fused = fused.reshape(batch_size, -1)  # [B, 1536]
        
        # Fusion MLP
        fused = self.fusion_mlp(fused)  # [B, 256]
        
        # Final classification
        output = self.classifier(fused)  # [B, 10]
        
        return output