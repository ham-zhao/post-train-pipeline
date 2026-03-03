#!/usr/bin/env python3
"""
评估脚本
运行 Base / SFT / DPO 三阶段的 benchmark 和安全评估

用法: python scripts/run_eval.py
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import get_run_config, get_eval_config
from src.evaluation.benchmark_runner import run_eval_pipeline, compare_stages
from src.evaluation.safety_evaluator import evaluate_safety
from src.evaluation.generation_quality import compare_generations, format_comparison_table, save_comparisons
from src.training.training_utils import load_model_and_tokenizer


def main():
    config = get_run_config()
    eval_config = get_eval_config()

    print(f"运行模式: {config['run_mode']}")
    print(f"评估任务: {config['eval_tasks']}")

    model_name = config["model"]["name"]
    sft_path = str(PROJECT_ROOT / "results/checkpoints/sft")
    dpo_path = str(PROJECT_ROOT / "results/checkpoints/dpo")

    # ============================================================
    # Step 1: Benchmark 评估
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 1: Benchmark 评估")
    print("=" * 60)

    results = {}

    # Base Model
    print("\n--- Base Model ---")
    results["base"] = run_eval_pipeline(model_name, "base")

    # SFT Model
    if Path(sft_path).exists():
        print("\n--- SFT Model ---")
        results["sft"] = run_eval_pipeline(sft_path, "sft")
    else:
        print(f"\nSFT 模型不存在: {sft_path}，跳过")

    # DPO Model
    if Path(dpo_path).exists():
        print("\n--- DPO Model ---")
        results["dpo"] = run_eval_pipeline(dpo_path, "dpo")
    else:
        print(f"\nDPO 模型不存在: {dpo_path}，跳过")

    # 对比表
    print("\n" + compare_stages(results))

    # ============================================================
    # Step 2: 安全评估
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 2: 安全评估")
    print("=" * 60)

    device = config["model"]["device"]
    safety_results = {}

    for stage, path in [("base", model_name), ("sft", sft_path), ("dpo", dpo_path)]:
        if stage != "base" and not Path(path).exists():
            continue

        print(f"\n--- {stage.upper()} 安全评估 ---")
        model, tokenizer = load_model_and_tokenizer(
            path, dtype=config["model"]["dtype"], device=device
        )
        safety = evaluate_safety(model, tokenizer, device=device)
        safety_results[stage] = safety

        # 保存
        output_dir = PROJECT_ROOT / f"results/eval_results/{stage}"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "safety.json", "w", encoding="utf-8") as f:
            json.dump(safety, f, ensure_ascii=False, indent=2)

        # 释放内存
        del model
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # ============================================================
    # Step 3: 生成质量对比
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 3: 生成质量对比")
    print("=" * 60)

    models = {}
    for stage, path in [("base", model_name), ("sft", sft_path), ("dpo", dpo_path)]:
        if stage != "base" and not Path(path).exists():
            continue
        model, tokenizer = load_model_and_tokenizer(
            path, dtype=config["model"]["dtype"], device=device
        )
        models[stage] = (model, tokenizer)

    if models:
        comparisons = compare_generations(models, device=device)
        print("\n" + format_comparison_table(comparisons))
        save_comparisons(
            comparisons,
            str(PROJECT_ROOT / "results/eval_results/generation_comparison.json")
        )

    # ============================================================
    # 总结
    # ============================================================
    print("\n" + "=" * 60)
    print("评估完成！")
    print("=" * 60)

    # 保存汇总结果
    summary = {
        "benchmarks": results,
        "safety": {k: {"ASR": v["ASR"], "Over-refusal": v["Over-refusal"]}
                   for k, v in safety_results.items()},
    }
    with open(PROJECT_ROOT / "results/eval_results/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"结果已保存到: results/eval_results/")


if __name__ == "__main__":
    main()
