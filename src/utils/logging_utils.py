"""
日志工具模块

提供实验日志记录功能
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


class ExperimentLogger:
    """
    实验日志记录器

    统一记录实验过程中的所有信息，包括:
    - 配置信息
    - 训练过程
    - 评估结果
    - API调用统计
    """

    def __init__(self, experiment_name: str, output_dir: str, log_level: int = logging.INFO):
        """
        初始化实验日志记录器

        Args:
            experiment_name: 实验名称
            output_dir: 输出目录
            log_level: 日志级别
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 创建logger
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(log_level)
        self.logger.handlers = []  # 清除已有handler

        # 文件handler
        log_file = self.output_dir / f"{experiment_name}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # 控制台handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # 实验元数据
        self.metadata = {
            'experiment_name': experiment_name,
            'start_time': datetime.now().isoformat(),
            'output_dir': str(output_dir),
            'logs': []
        }

        self.logger.info(f"Experiment started: {experiment_name}")

    def log_config(self, config: Dict[str, Any]):
        """记录实验配置"""
        self.metadata['config'] = config
        self.logger.info(f"Config: {json.dumps(config, indent=2)}")

    def log_metric(self, step: int, metrics: Dict[str, float], prefix: str = ""):
        """
        记录指标

        Args:
            step: 步数/轮数
            metrics: 指标字典
            prefix: 前缀(如train/val)
        """
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"[{prefix}] Step {step}: {metric_str}")

        # 保存到metadata
        self.metadata['logs'].append({
            'step': step,
            'prefix': prefix,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })

    def log_api_call(self, call_info: Dict[str, Any]):
        """记录API调用"""
        self.metadata.setdefault('api_calls', []).append(call_info)

    def save_metadata(self):
        """保存元数据到JSON文件"""
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        self.logger.info(f"Metadata saved to {metadata_file}")

    def finish(self, final_metrics: Optional[Dict[str, Any]] = None):
        """结束实验"""
        if final_metrics:
            self.metadata['final_metrics'] = final_metrics

        self.metadata['end_time'] = datetime.now().isoformat()
        self.save_metadata()
        self.logger.info(f"Experiment finished: {self.experiment_name}")


def setup_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO):
    """
    设置基础logger

    Args:
        name: logger名称
        log_file: 日志文件路径(可选)
        level: 日志级别

    Returns:
        logging.Logger: 配置好的logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []

    # 控制台handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)

    # 文件handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
