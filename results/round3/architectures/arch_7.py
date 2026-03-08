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
        
        # Layer norms for stability
        self.vision_ln = nn.LayerNorm(512)
        self.audio_ln = nn.LayerNorm(512)
        self.text_ln = nn.LayerNorm(512)
        
        # Cross-modal attention for fusion
        self.fusion_dim = 512
        self.num_heads = 8
        
        # Self-attention for each modality
        self.vision_attn = nn.MultiheadAttention(512, self.num_heads, batch_first=True)
        self.audio_attn = nn.MultiheadAttention(512, self.num_heads, batch_first=True)
        self.text_attn = nn.MultiheadAttention(512, self.num_heads, batch_first=True)
        
        # Cross-modal attention: vision attends to audio+text, etc.
        self.cross_vision = nn.MultiheadAttention(512, self.num_heads, batch_first=True)
        self.cross_audio = nn.MultiheadAttention(512, self.num_heads, batch_first=True)
        self.cross_text = nn.MultiheadAttention(512, self.num_heads, batch_first=True)
        
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
        
        # Global context for gating
        self.global_vision = nn.Linear(512, 512)
        self.global_audio = nn.Linear(512, 512)
        self.global_text = nn.Linear(512, 512)
        
        # Fusion layers
        self.fusion_mlp = nn.Sequential(
            nn.Linear(512 * 3, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
        
    def forward(self, vision, audio, text):
        batch_size = vision.shape[0]
        
        # Project to common dimension
        v = self.vision_proj(vision)  # [B, 576, 512]
        a = self.audio_proj(audio)    # [B, 400, 512]
        t = self.text_proj(text)      # [B, 77, 512]
        
        # Layer normalization
        v = self.vision_ln(v)
        a = self.audio_ln(a)
        t = self.text_ln(t)
        
        # Self-attention for each modality
        v_attn, _ = self.vision_attn(v, v, v)  # [B, 576, 512]
        a_attn, _ = self.audio_attn(a, a, a)   # [B, 400, 512]
        t_attn, _ = self.text_attn(t, t, t)    # [B, 77, 512]
        
        # Mean pooling for global representation
        v_pooled = v_attn.mean(dim=1)  # [B, 512]
        a_pooled = a_attn.mean(dim=1)  # [B, 512]
        t_pooled = t_attn.mean(dim=1)  # [B, 512]
        
        # Cross-modal attention using pooled as query
        v_pooled_expanded = v_pooled.unsqueeze(1)  # [B, 1, 512]
        a_pooled_expanded = a_pooled.unsqueeze(1)  # [B, 1, 512]
        t_pooled_expanded = t_pooled.unsqueeze(1)  # [B, 1, 512]
        
        # Vision attends to audio and text concatenated
        at_concat = torch.cat([a_attn, t_attn], dim=1)  # [B, 477, 512]
        v_cross, _ = self.cross_vision(v_pooled_expanded, at_concat, at_concat)
        v_cross = v_cross.squeeze(1)  # [B, 512]
        
        # Audio attends to vision and text concatenated
        vt_concat = torch.cat([v_attn, t_attn], dim=1)  # [B, 653, 512]
        a_cross, _ = self.cross_audio(a_pooled_expanded, vt_concat, vt_concat)
        a_cross = a_cross.squeeze(1)  # [B, 512]
        
        # Text attends to vision and audio concatenated
        va_concat = torch.cat([v_attn, a_attn], dim=1)  # [B, 976, 512]
        t_cross, _ = self.cross_text(t_pooled_expanded, va_concat, va_concat)
        t_cross = t_cross.squeeze(1)  # [B, 512]
        
        # Compute global context for gating
        global_v = self.global_vision(v_pooled)
        global_a = self.global_audio(a_pooled)
        global_t = self.global_text(t_pooled)
        
        # Gating mechanism
        gate_v = self.gate_vision(global_v + global_a + global_t)
        gate_a = self.gate_audio(global_v + global_a + global_t)
        gate_t = self.gate_text(global_v + global_a + global_t)
        
        # Apply gates
        v_gated = v_cross * gate_v  # [B, 512]
        a_gated = a_cross * gate_a  # [B, 512]
        t_gated = t_cross * gate_t  # [B, 512]
        
        # Concatenate gated representations
        fused = torch.cat([v_gated, a_gated, t_gated], dim=-1)  # [B, 1536]
        
        # Fusion MLP
        fused = self.fusion_mlp(fused)  # [B, 512]
        
        # Residual connection with pooled average
        residual = (v_pooled + a_pooled + t_pooled) / 3.0
        fused = fused + residual
        
        # Output head
        output = self.output_head(fused)  # [B, 10]
        
        # Ensure correct shape using reshape
        output = output.reshape(batch_size, 10)
        
        return output