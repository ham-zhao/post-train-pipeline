#!/usr/bin/env python3
"""
DPO 训练脚本
用法: python scripts/run_dpo.py
耗时 >30 分钟时建议: caffeinate -i python scripts/run_dpo.py

产出文件：
  results/checkpoints/dpo/model.safetensors    - DPO 模型权重
  results/checkpoints/dpo/tokenizer.json       - Tokenizer
  results/checkpoints/dpo/training_log.json    - 训练日志（loss/chosen_rewards/rejected_rewards per step）
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.dpo_trainer import run_dpo_training
from src.utils.config_loader import get_run_config


def main():
    config = get_run_config()
    print(f"运行模式: {config['run_mode']}")

    model, tokenizer, logger = run_dpo_training()

    print("\n" + "=" * 60)
    print("DPO 训练完成！")
    print("=" * 60)
    print(f"训练步数: {len(logger.train_losses)}")
    if logger.train_losses:
        print(f"最终 loss: {logger.train_losses[-1]:.4f}")
    if logger.chosen_rewards:
        margin = logger.chosen_rewards[-1] - logger.rejected_rewards[-1]
        print(f"最终 reward margin: {margin:.4f}")
    print(f"\n下一步: python scripts/run_eval.py")


if __name__ == "__main__":
    main()
