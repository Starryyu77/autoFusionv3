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
        
        # Cross-modal attention layers
        self.vision_to_fusion = nn.MultiheadAttention(512, 8, batch_first=True)
        self.audio_to_fusion = nn.MultiheadAttention(512, 8, batch_first=True)
        self.text_to_fusion = nn.MultiheadAttention(512, 8, batch_first=True)
        
        # Self-attention for fused representation
        self.fusion_attn = nn.MultiheadAttention(512, 8, batch_first=True)
        
        # Gating mechanism
        self.gate_vision = nn.Sequential(nn.Linear(512, 512), nn.Sigmoid())
        self.gate_audio = nn.Sequential(nn.Linear(512, 512), nn.Sigmoid())
        self.gate_text = nn.Sequential(nn.Linear(512, 512), nn.Sigmoid())
        
        # Fusion MLP with residual
        self.fusion_mlp = nn.Sequential(
            nn.Linear(512 * 3, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512)
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 10)
        )
        
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
        
        # Project and normalize
        v = self.vision_ln(self.vision_proj(vision))  # [B, 576, 512]
        a = self.audio_ln(self.audio_proj(audio))     # [B, 400, 512]
        t = self.text_ln(self.text_proj(text))        # [B, 77, 512]
        
        # Mean pooling for global features
        v_pooled = v.mean(dim=1, keepdim=True)   # [B, 1, 512]
        a_pooled = a.mean(dim=1, keepdim=True)   # [B, 1, 512]
        t_pooled = t.mean(dim=1, keepdim=True)   # [B, 1, 512]
        
        # Concatenate for cross-modal attention queries
        pooled_cat = torch.cat([v_pooled, a_pooled, t_pooled], dim=1)  # [B, 3, 512]
        
        # Cross-modal attention: each modality attends to others
        v_fused, _ = self.vision_to_fusion(pooled_cat, v, v)   # [B, 3, 512]
        a_fused, _ = self.audio_to_fusion(pooled_cat, a, a)    # [B, 3, 512]
        t_fused, _ = self.text_to_fusion(pooled_cat, t, t)     # [B, 3, 512]
        
        # Mean pool the fused representations
        v_fused = v_fused.mean(dim=1)  # [B, 512]
        a_fused = a_fused.mean(dim=1)  # [B, 512]
        t_fused = t_fused.mean(dim=1)  # [B, 512]
        
        # Gating for adaptive fusion
        g_v = self.gate_vision(v_fused)
        g_a = self.gate_audio(a_fused)
        g_t = self.gate_text(t_fused)
        
        v_gated = v_fused * g_v
        a_gated = a_fused * g_a
        t_gated = t_fused * g_t
        
        # Concatenate gated features
        fused = torch.cat([v_gated, a_gated, t_gated], dim=-1)  # [B, 1536]
        
        # MLP fusion with residual connection to mean of inputs
        residual = (v_fused + a_fused + t_fused) / 3.0  # [B, 512]
        
        # Reshape for MLP: [B, 1536] -> process -> [B, 512]
        fused_mlp = self.fusion_mlp(fused)  # [B, 512]
        
        # Add residual (project residual to match if needed, but dims match here)
        fused_final = fused_mlp + residual  # [B, 512]
        
        # Final classification
        output = self.classifier(fused_final)  # [B, 10]
        
        # Ensure correct shape using reshape
        output = output.reshape(batch_size, 10)
        
        return output