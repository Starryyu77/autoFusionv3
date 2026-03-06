"""
检查点管理模块

用于保存和恢复实验状态，支持中断后恢复
"""

import os
import json
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime


class CheckpointManager:
    """
    检查点管理器

    支持:
    - 定期保存实验状态
    - 中断后恢复
    - 自动清理旧检查点
    """

    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 5,
        save_interval: int = 10
    ):
        """
        初始化检查点管理器

        Args:
            checkpoint_dir: 检查点保存目录
            max_checkpoints: 最大保留检查点数
            save_interval: 保存间隔(轮数)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.save_interval = save_interval

    def save(
        self,
        generation: int,
        state: Dict[str, Any],
        is_best: bool = False
    ):
        """
        保存检查点

        Args:
            generation: 当前轮数
            state: 状态字典(包含模型、优化器、随机状态等)
            is_best: 是否为最佳模型
        """
        checkpoint = {
            'generation': generation,
            'timestamp': datetime.now().isoformat(),
            'state': state
        }

        # 保存常规检查点
        if generation % self.save_interval == 0:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_gen_{generation}.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"✅ Checkpoint saved: {checkpoint_path}")

        # 保存最佳模型
        if is_best:
            best_path = self.checkpoint_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)
            print(f"✅ Best checkpoint saved: {best_path}")

        # 保存最新模型
        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)

        # 清理旧检查点
        self._cleanup_old_checkpoints()

    def load(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """
        加载检查点

        Args:
            checkpoint_path: 检查点路径，默认加载latest

        Returns:
            状态字典
        """
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / "checkpoint_latest.pt"

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path)
        print(f"✅ Checkpoint loaded: {checkpoint_path}")
        print(f"   Generation: {checkpoint['generation']}")
        print(f"   Timestamp: {checkpoint['timestamp']}")

        return checkpoint

    def load_best(self) -> Dict[str, Any]:
        """加载最佳检查点"""
        best_path = self.checkpoint_dir / "checkpoint_best.pt"
        return self.load(best_path)

    def _cleanup_old_checkpoints(self):
        """清理旧检查点，只保留最新的max_checkpoints个"""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_gen_*.pt"),
            key=lambda x: int(x.stem.split('_')[-1])
        )

        if len(checkpoints) > self.max_checkpoints:
            for ckpt in checkpoints[:-self.max_checkpoints]:
                ckpt.unlink()
                print(f"🗑️  Removed old checkpoint: {ckpt}")

    def list_checkpoints(self):
        """列出所有检查点"""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
        print(f"\nCheckpoints in {self.checkpoint_dir}:")
        for ckpt in sorted(checkpoints):
            size_mb = ckpt.stat().st_size / (1024 * 1024)
            print(f"  {ckpt.name} ({size_mb:.1f} MB)")
        return checkpoints
