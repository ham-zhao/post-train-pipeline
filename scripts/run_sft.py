#!/usr/bin/env python3
"""
SFT 训练脚本
用法: python scripts/run_sft.py
耗时 >30 分钟时建议: caffeinate -i python scripts/run_sft.py

产出文件：
  results/checkpoints/sft/model.safetensors    - SFT 模型权重
  results/checkpoints/sft/tokenizer.json       - Tokenizer
  results/checkpoints/sft/training_log.json    - 训练日志（loss/eval_loss per step）
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.sft_trainer import run_sft_training
from src.utils.config_loader import get_run_config


def main():
    config = get_run_config()
    print(f"运行模式: {config['run_mode']}")

    model, tokenizer, logger = run_sft_training()

    print("\n" + "=" * 60)
    print("SFT 训练完成！")
    print("=" * 60)
    print(f"训练步数: {len(logger.train_losses)}")
    if logger.train_losses:
        print(f"最终 loss: {logger.train_losses[-1]:.4f}")
    print(f"\n下一步: python scripts/run_dpo.py")


if __name__ == "__main__":
    main()
