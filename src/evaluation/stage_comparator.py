"""
阶段对比器
汇总 Base / SFT / DPO 三阶段的评估结果，生成对比表和 Dashboard

对应 Tülu 3 论文:
- Section 5: 多维度评估对比
"""

import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config_loader import PROJECT_ROOT
from src.utils.visualization import plot_dashboard, plot_radar_chart, plot_safety_comparison


def load_stage_results(stage: str) -> dict:
    """加载某阶段的评估结果"""
    result_dir = PROJECT_ROOT / f"results/eval_results/{stage}"

    results = {}

    # 加载 benchmark 分数
    scores_file = result_dir / "scores.json"
    if scores_file.exists():
        with open(scores_file) as f:
            results["benchmarks"] = json.load(f)

    # 加载安全评估
    safety_file = result_dir / "safety.json"
    if safety_file.exists():
        with open(safety_file) as f:
            safety = json.load(f)
            results["safety"] = {
                "ASR": safety.get("ASR", 0),
                "Over-refusal": safety.get("Over-refusal", 0),
            }

    # 加载训练日志
    for checkpoint_dir in ["results/checkpoints/sft", "results/checkpoints/dpo"]:
        log_file = PROJECT_ROOT / checkpoint_dir / "training_log.json"
        if log_file.exists() and stage in checkpoint_dir:
            with open(log_file) as f:
                results["training_log"] = json.load(f)

    return results


def generate_comparison_report() -> dict:
    """
    生成完整三方对比报告

    Returns:
        包含所有对比数据的字典
    """
    stages = ["base", "sft", "dpo"]
    all_results = {}

    for stage in stages:
        all_results[stage] = load_stage_results(stage)

    # 生成对比表
    report = {
        "stages": stages,
        "results": all_results,
        "comparison_table": build_comparison_table(all_results),
    }

    return report


def build_comparison_table(results: dict) -> str:
    """构建对比表文本"""
    stages = ["base", "sft", "dpo"]

    # 收集所有指标
    all_metrics = set()
    for stage_results in results.values():
        if "benchmarks" in stage_results:
            all_metrics.update(stage_results["benchmarks"].keys())

    # 构建表格
    header = f"{'指标':<20}" + "".join(f"{s.upper():<15}" for s in stages) + f"{'LIFT':<15}"
    lines = ["=" * 75, header, "-" * 75]

    # Benchmark 指标
    for metric in sorted(all_metrics):
        row = f"{metric:<20}"
        values = []
        for stage in stages:
            val = results.get(stage, {}).get("benchmarks", {}).get(metric)
            if val is not None:
                row += f"{val:<15.4f}"
                values.append(val)
            else:
                row += f"{'—':<15}"
                values.append(None)

        # LIFT
        if values[0] is not None and values[-1] is not None:
            lift = values[-1] - values[0]
            row += f"{lift:+.4f}"
        else:
            row += "—"
        lines.append(row)

    # 安全指标
    lines.append("-" * 75)
    for metric in ["ASR", "Over-refusal"]:
        row = f"{metric + ' ↓':<20}"
        values = []
        for stage in stages:
            val = results.get(stage, {}).get("safety", {}).get(metric)
            if val is not None:
                row += f"{val:<15.1f}%"
                values.append(val)
            else:
                row += f"{'—':<15}"
                values.append(None)

        if values[0] is not None and values[-1] is not None:
            lift = values[-1] - values[0]
            row += f"{lift:+.1f}%"
        else:
            row += "—"
        lines.append(row)

    lines.append("=" * 75)
    return "\n".join(lines)


def generate_dashboard(results: dict, save_path: str = None):
    """生成四象限 Dashboard"""
    # 准备雷达图数据
    categories = []
    scores = {"Base": [], "SFT": [], "DPO": []}

    for metric in sorted(results.get("base", {}).get("benchmarks", {}).keys()):
        categories.append(metric)
        for stage, label in [("base", "Base"), ("sft", "SFT"), ("dpo", "DPO")]:
            val = results.get(stage, {}).get("benchmarks", {}).get(metric, 0)
            scores[label].append(val * 100)  # 转为百分比

    radar_data = {"categories": categories, "scores": scores}

    # 准备安全数据
    safety_data = {}
    for stage, label in [("base", "Base"), ("sft", "SFT"), ("dpo", "DPO")]:
        safety_data[label] = results.get(stage, {}).get("safety", {"ASR": 0, "Over-refusal": 0})

    # 准备训练 loss 数据
    sft_log = results.get("sft", {}).get("training_log", {})
    dpo_log = results.get("dpo", {}).get("training_log", {})

    sft_losses = sft_log.get("train_losses", [0])
    dpo_chosen = dpo_log.get("chosen_rewards", [0])
    dpo_rejected = dpo_log.get("rejected_rewards", [0])
    dpo_margins = [c - r for c, r in zip(dpo_chosen, dpo_rejected)] if dpo_chosen and dpo_rejected else [0]

    save_path = save_path or str(PROJECT_ROOT / "results/figures/dashboard.png")

    fig = plot_dashboard(
        train_losses_sft=sft_losses,
        train_losses_dpo=[],
        dpo_margins=dpo_margins,
        radar_data=radar_data,
        safety_data=safety_data,
        save_path=save_path,
    )

    print(f"Dashboard 已保存到: {save_path}")
    return fig
