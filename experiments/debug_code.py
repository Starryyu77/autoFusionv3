#!/usr/bin/env python3
"""Debug script to test generated code"""

import sys
import traceback
sys.path.insert(0, 'src')

import torch
import torch.nn as nn

# Generated code from LLM
code = '''import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalFusion(nn.Module):
    def __init__(self):
        super().__init__()

        # Vision projection: 1024 -> 512
        self.vision_proj = nn.Sequential(
            nn.Linear(1024, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Linear(768, 512),
            nn.LayerNorm(512)
        )

        # Audio projection: 512 -> 512 (same dim)
        self.audio_proj = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512)
        )

        # Text projection: 768 -> 512
        self.text_proj = nn.Sequential(
            nn.Linear(768, 640),
            nn.LayerNorm(640),
            nn.GELU(),
            nn.Linear(640, 512),
            nn.LayerNorm(512)
        )

        # Cross-modal attention for fusion
        self.cross_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)

        # Self-attention for refined fusion
        self.self_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)

        # Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(512 * 3, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Output head
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 10)
        )

    def forward(self, vision, audio, text):
        B = vision.size(0)

        # Project to common dimension
        v_feat = self.vision_proj(vision)  # [B, 576, 512]
        a_feat = self.audio_proj(audio)    # [B, 400, 512]
        t_feat = self.text_proj(text)      # [B, 77, 512]

        # Mean pooling over sequence dimension
        v_pooled = v_feat.mean(dim=1)      # [B, 512]
        a_pooled = a_feat.mean(dim=1)      # [B, 512]
        t_pooled = t_feat.mean(dim=1)      # [B, 512]

        # Stack for cross-attention
        concat = torch.stack([v_pooled, a_pooled, t_pooled], dim=1)  # [B, 3, 512]

        # Self-attention
        attn_out, _ = self.self_attn(concat, concat, concat)  # [B, 3, 512]

        # Flatten and fusion
        fused = attn_out.view(B, -1)  # [B, 1536]
        fused = self.fusion_mlp(fused)  # [B, 512]

        # Classification
        output = self.classifier(fused)  # [B, 10]
        return output
'''

try:
    namespace = {}
    exec(code, namespace)
    ModelClass = namespace['MultimodalFusion']
    model = ModelClass()

    # Test forward pass
    vision = torch.randn(2, 576, 1024)
    audio = torch.randn(2, 400, 512)
    text = torch.randn(2, 77, 768)

    output = model(vision, audio, text)
    print(f'Output shape: {output.shape}')
    print('SUCCESS!')
except Exception as e:
    print(f'Error: {type(e).__name__}: {e}')
    traceback.print_exc()
