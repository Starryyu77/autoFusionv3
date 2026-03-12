import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalFusion(nn.Module):
    """
    多模态融合架构：基于Cross-Modal Attention与Gated Fusion的高效融合
    
    创新点：
    1. 跨模态注意力机制：vision作为query，audio和text作为key/value进行交叉注意力
    2. 门控融合：学习自适应的模态重要性权重
    3. 多尺度特征：局部时序特征 + 全局池化特征
    4. 残差连接与层归一化稳定训练
    """

    def __init__(self, dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        
        # 投影层：将不同输入维度统一映射到dim
        self.vision_proj = nn.Linear(35, dim)
        self.audio_proj = nn.Linear(74, dim)
        self.text_proj = nn.Linear(300, dim)
        
        # 时序位置编码
        self.temporal_pos_enc = nn.Parameter(torch.randn(1, 50, dim) * 0.02)
        
        # 跨模态注意力：vision作为query，融合audio+text作为key/value
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        
        # 自注意力：增强单模态表示
        self.self_attn_v = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.self_attn_a = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.self_attn_t = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        
        # 门控融合参数
        self.gate_v = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.gate_a = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.gate_t = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        
        # 融合前的变换
        self.fusion_transform = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # 深层融合MLP
        self.deep_fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )
        
        # 残差缩放参数
        self.residual_scale = nn.Parameter(torch.ones(1))
        
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
        B = vision.size(0)
        
        # Step 1: 投影到统一维度并添加位置编码
        # vision: [B, 50, 35] -> [B, 50, dim]
        # audio: [B, 50, 74] -> [B, 50, dim]  
        # text: [B, 50, 300] -> [B, 50, dim]
        v_proj = self.vision_proj(vision) + self.temporal_pos_enc
        a_proj = self.audio_proj(audio) + self.temporal_pos_enc
        t_proj = self.text_proj(text) + self.temporal_pos_enc
        
        # Step 2: 自注意力增强单模态表示
        v_self, _ = self.self_attn_v(v_proj, v_proj, v_proj)
        a_self, _ = self.self_attn_a(a_proj, a_proj, a_proj)
        t_self, _ = self.self_attn_t(t_proj, t_proj, t_proj)
        
        # 残差连接
        v_enhanced = v_proj + v_self
        a_enhanced = a_proj + a_self
        t_enhanced = t_proj + t_self
        
        # Step 3: 时序池化获取全局特征 [B, dim]
        v_global = v_enhanced.mean(dim=1)
        a_global = a_enhanced.mean(dim=1)
        t_global = t_enhanced.mean(dim=1)
        
        # Step 4: 跨模态注意力（vision作为query，audio+text作为key/value）
        # 拼接audio和text作为跨模态上下文
        at_context = torch.cat([a_enhanced, t_enhanced], dim=1)  # [B, 100, dim]
        
        # 使用vision的时序特征作为query进行交叉注意力
        v_cross, attn_weights = self.cross_attn(
            v_enhanced, at_context, at_context
        )  # [B, 50, dim]
        
        # 池化跨模态特征
        v_cross_global = v_cross.mean(dim=1)  # [B, dim]
        
        # Step 5: 门控融合 - 学习自适应模态权重
        gate_v = self.gate_v(v_global)
        gate_a = self.gate_a(a_global)
        gate_t = self.gate_t(t_global)
        
        # 应用门控
        v_gated = v_global * gate_v + v_cross_global * (1 - gate_v)
        a_gated = a_global * gate_a
        t_gated = t_global * gate_t
        
        # Step 6: 多尺度特征拼接
        # 全局特征 + 跨模态交互特征
        multimodal_feat = torch.cat([v_gated, a_gated, t_gated], dim=-1)  # [B, dim*3]
        
        # Step 7: 深层融合
        transformed = self.fusion_transform(multimodal_feat)  # [B, dim*2]
        
        # 添加残差：从transformed中提取与原始全局特征相关的部分
        residual_input = torch.cat([
            v_global, 
            (a_global + t_global) / 2
        ], dim=-1)  # [B, dim*2]
        
        # 融合并添加缩放残差
        fused = self.deep_fusion(transformed + self.residual_scale * residual_input)
        
        # Step 8: 最终输出投影
        output = self.output_proj(fused)
        
        return output


# 测试代码
if __name__ == "__main__":
    # 创建测试输入
    B = 2
    vision = torch.randn(B, 50, 35)
    audio = torch.randn(B, 50, 74)
    text = torch.randn(B, 50, 300)
    
    # 初始化模型
    model = MultimodalFusion(dim=256, num_heads=8, dropout=0.1)
    
    # 前向传播
    output = model(vision, audio, text)
    
    print(f"Vision input shape:  {vision.shape}")
    print(f"Audio input shape:   {audio.shape}")
    print(f"Text input shape:    {text.shape}")
    print(f"Output shape:        {output.shape}")
    print(f"Expected shape:      torch.Size([{B}, 256])")
    print(f"\nOutput stats: mean={output.mean().item():.4f}, std={output.std().item():.4f}")
    
    # 验证输出维度
    assert output.shape == torch.Size([B, 256]), f"Expected shape {[B, 256]}, got {output.shape}"
    print("\n✓ 维度验证通过！")