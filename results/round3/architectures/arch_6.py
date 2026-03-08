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
        
        # Vision projection
        self.vision_proj = nn.Linear(self.vision_dim, self.fused_dim)
        
        # Audio projection
        self.audio_proj = nn.Linear(self.audio_dim, self.fused_dim)
        
        # Text projection
        self.text_proj = nn.Linear(self.text_dim, self.fused_dim)
        
        # Cross-attention: vision attends to text
        self.cross_attn_vision_text = nn.MultiheadAttention(
            embed_dim=self.fused_dim, num_heads=8, batch_first=True
        )
        
        # Self-attention for audio
        self.audio_self_attn = nn.MultiheadAttention(
            embed_dim=self.fused_dim, num_heads=8, batch_first=True
        )
        
        # Gating mechanism for fusion
        self.gate_vision = nn.Linear(self.fused_dim, self.fused_dim)
        self.gate_audio = nn.Linear(self.fused_dim, self.fused_dim)
        self.gate_text = nn.Linear(self.fused_dim, self.fused_dim)
        
        # Fusion layers
        self.fusion_layer1 = nn.Linear(self.fused_dim * 3, self.fused_dim * 2)
        self.fusion_layer2 = nn.Linear(self.fused_dim * 2, self.fused_dim)
        
        # Output classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 10)
        )
        
        # Layer norms
        self.ln_vision = nn.LayerNorm(self.fused_dim)
        self.ln_audio = nn.LayerNorm(self.fused_dim)
        self.ln_text = nn.LayerNorm(self.fused_dim)
        
    def forward(self, vision, audio, text):
        batch_size = vision.shape[0]
        
        # Project to common dimension
        vision_proj = self.vision_proj(vision)  # [B, 576, 512]
        audio_proj = self.audio_proj(audio)      # [B, 400, 512]
        text_proj = self.text_proj(text)         # [B, 77, 512]
        
        # Mean pooling for initial representations (handle variable lengths)
        vision_pooled = vision_proj.mean(dim=1)  # [B, 512]
        audio_pooled = audio_proj.mean(dim=1)    # [B, 512]
        text_pooled = text_proj.mean(dim=1)      # [B, 512]
        
        # Cross-attention: vision attends to text
        vision_attended, _ = self.cross_attn_vision_text(
            vision_proj, text_proj, text_proj
        )
        vision_attended = self.ln_vision(vision_attended.mean(dim=1))  # [B, 512]
        
        # Self-attention for audio
        audio_attended, _ = self.audio_self_attn(
            audio_proj, audio_proj, audio_proj
        )
        audio_attended = self.ln_audio(audio_attended.mean(dim=1))  # [B, 512]
        
        # Text representation with residual
        text_repr = self.ln_text(text_pooled)  # [B, 512]
        
        # Gating mechanism
        gate_v = torch.sigmoid(self.gate_vision(vision_attended))
        gate_a = torch.sigmoid(self.gate_audio(audio_attended))
        gate_t = torch.sigmoid(self.gate_text(text_repr))
        
        # Apply gates
        vision_gated = vision_attended * gate_v
        audio_gated = audio_attended * gate_a
        text_gated = text_repr * gate_t
        
        # Concatenate all modalities
        fused = torch.cat([vision_gated, audio_gated, text_gated], dim=-1)  # [B, 1536]
        
        # Fusion MLP
        fused = F.relu(self.fusion_layer1(fused))
        fused = F.relu(self.fusion_layer2(fused))
        
        # Final classification
        output = self.classifier(fused)  # [B, 10]
        
        # Ensure correct output shape using reshape
        output = output.reshape(batch_size, 10)
        
        return output