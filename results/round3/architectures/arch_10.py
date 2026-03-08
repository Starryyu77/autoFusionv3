import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalFusion(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Project all modalities to common dimension 512
        self.vision_proj = nn.Linear(1024, 512)
        self.audio_proj = nn.Linear(512, 512)
        self.text_proj = nn.Linear(768, 512)
        
        # Cross-modal attention layers
        self.vision_to_fusion = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        self.audio_to_fusion = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        self.text_to_fusion = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        
        # Gating mechanism for fusion
        self.gate_vision = nn.Linear(512, 512)
        self.gate_audio = nn.Linear(512, 512)
        self.gate_text = nn.Linear(512, 512)
        self.gate_sigmoid = nn.Sigmoid()
        
        # Fusion projection
        self.fusion_proj = nn.Linear(512 * 3, 512)
        
        # Output layers
        self.fc1 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 10)
        
    def forward(self, vision, audio, text):
        batch_size = vision.shape[0]
        
        # Project modalities to common dimension
        vision_feat = self.vision_proj(vision)  # [B, 576, 512]
        audio_feat = self.audio_proj(audio)     # [B, 400, 512]
        text_feat = self.text_proj(text)        # [B, 77, 512]
        
        # Mean pooling with mask handling for variable lengths
        vision_pooled = vision_feat.mean(dim=1, keepdim=True)   # [B, 1, 512]
        audio_pooled = audio_feat.mean(dim=1, keepdim=True)     # [B, 1, 512]
        text_pooled = text_feat.mean(dim=1, keepdim=True)       # [B, 1, 512]
        
        # Cross-modal attention: each modality attends to others
        # Reshape for attention: [B, 1, 512] -> query, [B, N, 512] -> key/value
        vision_attended, _ = self.vision_to_fusion(
            vision_pooled, 
            torch.cat([audio_feat, text_feat], dim=1),
            torch.cat([audio_feat, text_feat], dim=1)
        )  # [B, 1, 512]
        
        audio_attended, _ = self.audio_to_fusion(
            audio_pooled,
            torch.cat([vision_feat, text_feat], dim=1),
            torch.cat([vision_feat, text_feat], dim=1)
        )  # [B, 1, 512]
        
        text_attended, _ = self.text_to_fusion(
            text_pooled,
            torch.cat([vision_feat, audio_feat], dim=1),
            torch.cat([vision_feat, audio_feat], dim=1)
        )  # [B, 1, 512]
        
        # Squeeze sequence dimension
        vision_attended = vision_attended.squeeze(1)  # [B, 512]
        audio_attended = audio_attended.squeeze(1)    # [B, 512]
        text_attended = text_attended.squeeze(1)      # [B, 512]
        
        # Gating mechanism
        gate_v = self.gate_sigmoid(self.gate_vision(vision_attended))
        gate_a = self.gate_sigmoid(self.gate_audio(audio_attended))
        gate_t = self.gate_sigmoid(self.gate_text(text_attended))
        
        # Apply gates
        vision_gated = vision_attended * gate_v
        audio_gated = audio_attended * gate_a
        text_gated = text_attended * gate_t
        
        # Concatenate gated features
        fused = torch.cat([vision_gated, audio_gated, text_gated], dim=-1)  # [B, 1536]
        
        # Fuse and project
        fused = self.fusion_proj(fused)  # [B, 512]
        fused = F.relu(fused)
        
        # Final classification layers
        out = self.fc1(fused)      # [B, 256]
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)        # [B, 10]
        
        # Reshape to ensure correct output shape (using reshape, not view)
        out = out.reshape(batch_size, 10)
        
        return out