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
        
        # Layer norms for stability
        self.vision_ln = nn.LayerNorm(self.common_dim)
        self.audio_ln = nn.LayerNorm(self.common_dim)
        self.text_ln = nn.LayerNorm(self.common_dim)
        
        # Multi-head self-attention for intra-modal processing
        self.vision_attn = nn.MultiheadAttention(self.common_dim, num_heads=8, batch_first=True)
        self.audio_attn = nn.MultiheadAttention(self.common_dim, num_heads=8, batch_first=True)
        self.text_attn = nn.MultiheadAttention(self.common_dim, num_heads=8, batch_first=True)
        
        # Cross-modal attention for fusion
        # Vision attends to audio+text, Audio attends to vision+text, Text attends to vision+audio
        self.cross_attn_vision = nn.MultiheadAttention(self.common_dim, num_heads=8, batch_first=True)
        self.cross_attn_audio = nn.MultiheadAttention(self.common_dim, num_heads=8, batch_first=True)
        self.cross_attn_text = nn.MultiheadAttention(self.common_dim, num_heads=8, batch_first=True)
        
        # Gating mechanism for adaptive fusion
        self.gate_fc = nn.Sequential(
            nn.Linear(self.common_dim * 3, self.common_dim),
            nn.ReLU(),
            nn.Linear(self.common_dim, 3),
            nn.Softmax(dim=-1)
        )
        
        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.common_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 10)
        )
        
    def forward(self, vision, audio, text):
        batch_size = vision.shape[0]
        
        # Project to common dimension
        vision_proj = self.vision_ln(self.vision_proj(vision))  # [B, 576, 512]
        audio_proj = self.audio_ln(self.audio_proj(audio))      # [B, 400, 512]
        text_proj = self.text_ln(self.text_proj(text))          # [B, 77, 512]
        
        # Intra-modal self-attention
        vision_intra, _ = self.vision_attn(vision_proj, vision_proj, vision_proj)
        audio_intra, _ = self.audio_attn(audio_proj, audio_proj, audio_proj)
        text_intra, _ = self.text_attn(text_proj, text_proj, text_proj)
        
        # Add residual connections
        vision_feat = vision_proj + vision_intra
        audio_feat = audio_proj + audio_intra
        text_feat = text_proj + text_intra
        
        # Mean pooling to get global features for cross-attention keys/values
        vision_pooled = vision_feat.mean(dim=1, keepdim=True)  # [B, 1, 512]
        audio_pooled = audio_feat.mean(dim=1, keepdim=True)    # [B, 1, 512]
        text_pooled = text_feat.mean(dim=1, keepdim=True)      # [B, 1, 512]
        
        # Concatenate pooled features for cross-attention
        audio_text_kv = torch.cat([audio_pooled, text_pooled], dim=1)  # [B, 2, 512]
        vision_text_kv = torch.cat([vision_pooled, text_pooled], dim=1)  # [B, 2, 512]
        vision_audio_kv = torch.cat([vision_pooled, audio_pooled], dim=1)  # [B, 2, 512]
        
        # Cross-modal attention: each modality queries the others
        vision_cross, _ = self.cross_attn_vision(vision_feat, audio_text_kv, audio_text_kv)
        audio_cross, _ = self.cross_attn_audio(audio_feat, vision_text_kv, vision_text_kv)
        text_cross, _ = self.cross_attn_text(text_feat, vision_audio_kv, vision_audio_kv)
        
        # Residual connections for cross-attention
        vision_fused = vision_feat + vision_cross
        audio_fused = audio_feat + audio_cross
        text_fused = text_feat + text_cross
        
        # Global average pooling for each modality
        vision_global = vision_fused.mean(dim=1)  # [B, 512]
        audio_global = audio_fused.mean(dim=1)    # [B, 512]
        text_global = text_fused.mean(dim=1)      # [B, 512]
        
        # Compute adaptive fusion gates
        concat_features = torch.cat([vision_global, audio_global, text_global], dim=-1)  # [B, 1536]
        gates = self.gate_fc(concat_features)  # [B, 3]
        
        # Split gates
        g_v = gates[:, 0:1]  # [B, 1]
        g_a = gates[:, 1:2]  # [B, 1]
        g_t = gates[:, 2:3]  # [B, 1]
        
        # Apply gating and fuse
        # Use reshape instead of view as required
        g_v = g_v.reshape(batch_size, 1).expand(batch_size, self.common_dim)
        g_a = g_a.reshape(batch_size, 1).expand(batch_size, self.common_dim)
        g_t = g_t.reshape(batch_size, 1).expand(batch_size, self.common_dim)
        
        fused = g_v * vision_global + g_a * audio_global + g_t * text_global  # [B, 512]
        
        # Final classification
        output = self.classifier(fused)  # [B, 10]
        
        return output