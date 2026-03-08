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
        
        # Cross-modal attention for fusion
        self.cross_attn_vision = nn.MultiheadAttention(self.fused_dim, num_heads=8, batch_first=True)
        self.cross_attn_audio = nn.MultiheadAttention(self.fused_dim, num_heads=8, batch_first=True)
        self.cross_attn_text = nn.MultiheadAttention(self.fused_dim, num_heads=8, batch_first=True)
        
        # Gating mechanism for adaptive fusion
        self.gate_vision = nn.Sequential(
            nn.Linear(self.fused_dim * 3, self.fused_dim),
            nn.Sigmoid()
        )
        self.gate_audio = nn.Sequential(
            nn.Linear(self.fused_dim * 3, self.fused_dim),
            nn.Sigmoid()
        )
        self.gate_text = nn.Sequential(
            nn.Linear(self.fused_dim * 3, self.fused_dim),
            nn.Sigmoid()
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.fused_dim * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 10)
        )
        
    def forward(self, vision, audio, text):
        batch_size = vision.size(0)
        
        # Project to common dimension
        v = self.vision_proj(vision)  # [B, 576, 512]
        a = self.audio_proj(audio)    # [B, 400, 512]
        t = self.text_proj(text)      # [B, 77, 512]
        
        # Apply layer norm
        v = self.vision_norm(v)
        a = self.audio_norm(a)
        t = self.text_norm(t)
        
        # Intra-modal self-attention
        v_attn, _ = self.vision_attn(v, v, v)  # [B, 576, 512]
        a_attn, _ = self.audio_attn(a, a, a)   # [B, 400, 512]
        t_attn, _ = self.text_attn(t, t, t)    # [B, 77, 512]
        
        # Mean pooling for variable-length sequences
        v_pooled = v_attn.mean(dim=1)  # [B, 512]
        a_pooled = a_attn.mean(dim=1)  # [B, 512]
        t_pooled = t_attn.mean(dim=1)  # [B, 512]
        
        # Concatenate for cross-modal attention queries
        v_query = v_pooled.unsqueeze(1)  # [B, 1, 512]
        a_query = a_pooled.unsqueeze(1)  # [B, 1, 512]
        t_query = t_pooled.unsqueeze(1)  # [B, 1, 512]
        
        # Cross-modal attention: each modality attends to others
        # Vision attends to audio and text
        va_cross, _ = self.cross_attn_vision(v_query, a, a)  # [B, 1, 512]
        vt_cross, _ = self.cross_attn_vision(v_query, t, t)  # [B, 1, 512]
        
        # Audio attends to vision and text
        av_cross, _ = self.cross_attn_audio(a_query, v, v)   # [B, 1, 512]
        at_cross, _ = self.cross_attn_audio(a_query, t, t)   # [B, 1, 512]
        
        # Text attends to vision and audio
        tv_cross, _ = self.cross_attn_text(t_query, v, v)    # [B, 1, 512]
        ta_cross, _ = self.cross_attn_text(t_query, a, a)    # [B, 1, 512]
        
        # Aggregate cross-modal features
        v_cross = (va_cross + vt_cross) / 2  # [B, 1, 512]
        a_cross = (av_cross + at_cross) / 2  # [B, 1, 512]
        t_cross = (tv_cross + ta_cross) / 2  # [B, 1, 512]
        
        # Squeeze
        v_cross = v_cross.squeeze(1)  # [B, 512]
        a_cross = a_cross.squeeze(1)  # [B, 512]
        t_cross = t_cross.squeeze(1)  # [B, 512]
        
        # Combine pooled and cross-modal features
        v_feat = v_pooled + v_cross  # [B, 512]
        a_feat = a_pooled + a_cross  # [B, 512]
        t_feat = t_pooled + t_cross  # [B, 512]
        
        # Gating mechanism
        concat_feat = torch.cat([v_feat, a_feat, t_feat], dim=-1)  # [B, 1536]
        
        g_v = self.gate_vision(concat_feat)  # [B, 512]
        g_a = self.gate_audio(concat_feat)   # [B, 512]
        g_t = self.gate_text(concat_feat)    # [B, 512]
        
        # Apply gates
        v_gated = v_feat * g_v  # [B, 512]
        a_gated = a_feat * g_a  # [B, 512]
        t_gated = t_feat * g_t  # [B, 512]
        
        # Final fusion
        fused = torch.cat([v_gated, a_gated, t_gated], dim=-1)  # [B, 1536]
        
        # Use reshape instead of view
        fused = fused.reshape(batch_size, -1)  # [B, 1536]
        
        # Classify
        output = self.classifier(fused)  # [B, 10]
        
        return output