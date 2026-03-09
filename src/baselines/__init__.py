"""
基线方法模块 v2

完整端到端基线模型:

简单基线:
- MeanFusionModel: 简单平均融合
- ConcatFusionModel: 拼接+线性融合
- AttentionFusionModel: 自注意力融合
- MaxFusionModel: 最大值池化融合

固定架构基线:
- DynMMCompleteModel: 动态多模态融合
- TFNCompleteModel: 张量融合网络
- ADMNCompleteModel: 自适应动态多模态网络
- CentaurCompleteModel: 鲁棒多模态融合
- FDSNetCompleteModel: 特征分歧选择网络

NAS基线（含真实搜索）:
- DARTSCompleteModel: 可微分架构搜索
- LLMaticCompleteModel: LLM+质量多样性搜索
- EvoPromptingCompleteModel: 进化提示工程
"""

# 简单基线
from .simple_baselines_complete import (
    MeanFusionModel,
    ConcatFusionModel,
    AttentionFusionModel,
    MaxFusionModel
)

# 固定架构基线
from .dynmm_complete import DynMMCompleteModel
from .tfn_complete import TFNCompleteModel
from .admn_complete import ADMNCompleteModel
from .centaur_complete import CentaurCompleteModel
from .fdsnet_complete import FDSNetCompleteModel

# NAS基线
from .darts_complete import DARTSCompleteModel
from .llmatic_complete import LLMaticCompleteModel
from .evoprompting_complete import EvoPromptingCompleteModel

# 基类
from .base_complete_model import CompleteBaselineModel

__all__ = [
    # 简单基线
    'MeanFusionModel',
    'ConcatFusionModel',
    'AttentionFusionModel',
    'MaxFusionModel',
    # 固定架构基线
    'DynMMCompleteModel',
    'TFNCompleteModel',
    'ADMNCompleteModel',
    'CentaurCompleteModel',
    'FDSNetCompleteModel',
    # NAS基线
    'DARTSCompleteModel',
    'LLMaticCompleteModel',
    'EvoPromptingCompleteModel',
    # 基类
    'CompleteBaselineModel',
]
