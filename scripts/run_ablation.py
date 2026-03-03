#!/usr/bin/env python3
"""
消融实验脚本
5 组消融，验证各数据组件的贡献

对应 Tülu 3 论文:
- Section 3.1: 数据配比消融
- 验证安全数据正交性、数学数据贡献等

用法: python scripts/run_ablation.py
"""

import sys
import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import get_run_config
from src.data_preparation.subsample import save_samples
from src.training.sft_trainer import run_sft_training
from src.evaluation.benchmark_runner import run_eval_pipeline
from src.evaluation.safety_evaluator import evaluate_safety
from src.training.training_utils import load_model_and_tokenizer

# 消融实验配置
ABLATION_CONFIGS = {
    "full": {
        "description": "完整数据（对照基准）",
        "exclude_sources": [],
    },
    "no_safety": {
        "description": "去掉安全数据（验证安全正交性）",
        "exclude_sources": ["wildguardmix", "wildjailbreak"],
    },
    "no_math": {
        "description": "去掉数学数据（验证数学贡献）",
        "exclude_sources": ["numinamath", "persona_math"],
    },
    "no_coconot": {
        "description": "去掉 CoCoNot（验证防过度拒绝）",
        "exclude_sources": ["coconot"],
    },
    "no_norobots": {
        "description": "去掉 No Robots（验证人工数据价值）",
        "exclude_sources": ["no_robots"],
    },
    "no_flan": {
        "description": "去掉 FLAN v2（验证通用指令数据）",
        "exclude_sources": ["flan_v2"],
    },
}


def prepare_ablation_data(ablation_name: str, config: dict) -> str:
    """为消融实验准备数据（排除指定子集）"""
    exclude = ABLATION_CONFIGS[ablation_name]["exclude_sources"]

    # 读取完整数据
    full_data_path = PROJECT_ROOT / "data/sft_mix/train.jsonl"
    samples = []
    with open(full_data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                source = sample.get("source", "")
                if not any(ex in source.lower() for ex in exclude):
                    samples.append(sample)

    # 保存消融数据
    output_path = PROJECT_ROOT / f"data/sft_mix/ablation_{ablation_name}.jsonl"
    save_samples(samples, str(output_path))

    return str(output_path)


def run_single_ablation(ablation_name: str, config: dict) -> dict:
    """运行单个消融实验"""
    print(f"\n{'='*60}")
    print(f"消融实验: {ablation_name}")
    print(f"说明: {ABLATION_CONFIGS[ablation_name]['description']}")
    print(f"{'='*60}")

    # 准备数据
    data_path = prepare_ablation_data(ablation_name, config)

    # 训练
    output_dir = str(PROJECT_ROOT / f"results/checkpoints/ablation_{ablation_name}")
    model, tokenizer, logger = run_sft_training(
        data_path=data_path,
        output_dir=output_dir,
        max_steps=config.get("max_steps", 500),
    )

    # 评估 benchmark
    benchmark_results = run_eval_pipeline(output_dir, f"ablation_{ablation_name}")

    # 安全评估
    device = config["model"]["device"]
    safety_results = evaluate_safety(model, tokenizer, device=device)

    result = {
        "name": ablation_name,
        "description": ABLATION_CONFIGS[ablation_name]["description"],
        "benchmarks": benchmark_results,
        "safety": {
            "ASR": safety_results["ASR"],
            "Over-refusal": safety_results["Over-refusal"],
        },
        "train_loss_final": logger.train_losses[-1] if logger.train_losses else None,
    }

    # 保存单个结果
    result_path = PROJECT_ROOT / f"results/eval_results/ablation_{ablation_name}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    # 清理（保留模型太占空间）
    import torch
    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return result


def main():
    config = get_run_config()
    print(f"运行模式: {config['run_mode']}")
    print(f"消融实验: {len(ABLATION_CONFIGS)} 组")

    all_results = {}

    for ablation_name in ABLATION_CONFIGS:
        try:
            result = run_single_ablation(ablation_name, config)
            all_results[ablation_name] = result
        except Exception as e:
            print(f"\n消融 {ablation_name} 失败: {e}")
            continue

    # 汇总对比
    print("\n" + "=" * 60)
    print("消融实验汇总")
    print("=" * 60)

    header = f"{'实验':<15} {'HellaSwag':<12} {'ASR↓':<10} {'Over-ref↓':<12} {'说明'}"
    print(header)
    print("-" * 80)

    for name, result in all_results.items():
        hellaswag = result.get("benchmarks", {}).get("hellaswag", "—")
        asr = result.get("safety", {}).get("ASR", "—")
        over_ref = result.get("safety", {}).get("Over-refusal", "—")
        desc = ABLATION_CONFIGS[name]["description"]

        hs_str = f"{hellaswag:.4f}" if isinstance(hellaswag, float) else str(hellaswag)
        asr_str = f"{asr:.1f}%" if isinstance(asr, (int, float)) else str(asr)
        or_str = f"{over_ref:.1f}%" if isinstance(over_ref, (int, float)) else str(over_ref)

        print(f"{name:<15} {hs_str:<12} {asr_str:<10} {or_str:<12} {desc}")

    # 保存汇总
    summary_path = PROJECT_ROOT / "results/eval_results/ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n结果已保存到: {summary_path}")


if __name__ == "__main__":
    main()
