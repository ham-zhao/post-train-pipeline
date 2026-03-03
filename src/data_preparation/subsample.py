"""
数据子采样器
根据 run_config.yaml 中的配置对数据集进行子采样

对应 Tülu 3 论文:
- Section 3.1: 数据配比策略
- Tülu 3 的"技能隔离"方法：先单独调配每个技能的数据量
"""

import json
import random
from pathlib import Path
from collections import Counter
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config_loader import get_run_config, PROJECT_ROOT


def subsample_sft(dataset, config: dict) -> list:
    """
    对 SFT 数据集按配置进行子采样
    smoke_test: 取 2000 条（按比例从各子集采样）
    full_run: 取 57K 条（按配置的子集大小采样）
    """
    run_mode = config["run_mode"]
    total_target = config["sft_sample_size"]
    sft_subsets = config["sft_subsets"]
    seed = config["seed"]

    random.seed(seed)

    print(f"\nSFT 子采样 (模式: {run_mode}, 目标: {total_target:,} 条)")
    print("-" * 50)

    # 按来源分组
    if hasattr(dataset, 'column_names') and "dataset" in dataset.column_names:
        # HuggingFace Dataset 对象
        groups = {}
        for i in range(len(dataset)):
            source = dataset[i].get("dataset", "unknown")
            if source not in groups:
                groups[source] = []
            groups[source].append(dataset[i])
    elif hasattr(dataset, 'column_names') and "source" in dataset.column_names:
        groups = {}
        for i in range(len(dataset)):
            source = dataset[i].get("source", "unknown")
            if source not in groups:
                groups[source] = []
            groups[source].append(dataset[i])
    else:
        # 如果没有来源标记，直接随机采样
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        selected = indices[:total_target]
        samples = [dataset[i] for i in selected]
        print(f"  无来源标记，随机采样 {len(samples):,} 条")
        return samples

    # 按子集配比采样
    if run_mode == "smoke_test":
        # smoke_test: 按各子集目标量等比缩放到总量 2000
        total_subsample = sum(v["subsample"] for v in sft_subsets.values())
        scale = total_target / total_subsample
    else:
        scale = 1.0

    all_samples = []
    for subset_name, subset_info in sft_subsets.items():
        target = int(subset_info["subsample"] * scale)

        # 在分组数据中查找匹配的来源
        matched_key = None
        for key in groups:
            if subset_name.lower() in key.lower() or key.lower() in subset_name.lower():
                matched_key = key
                break

        if matched_key and len(groups[matched_key]) > 0:
            available = groups[matched_key]
            actual = min(target, len(available))
            sampled = random.sample(available, actual)
            # 标记来源
            for s in sampled:
                s["source"] = subset_name
            all_samples.extend(sampled)
            print(f"  {subset_name}: {actual:,} / {len(available):,} (目标 {target:,})")
        else:
            print(f"  {subset_name}: 未找到匹配数据 (目标 {target:,})")

    # 如果采样不足，从剩余数据补充
    if len(all_samples) < total_target:
        remaining = []
        used_indices = set()
        for group_samples in groups.values():
            for s in group_samples:
                if id(s) not in used_indices:
                    remaining.append(s)
        shortfall = total_target - len(all_samples)
        if remaining:
            extra = random.sample(remaining, min(shortfall, len(remaining)))
            all_samples.extend(extra)
            print(f"  补充采样: {len(extra):,} 条")

    random.shuffle(all_samples)
    print(f"\n总计: {len(all_samples):,} 条")
    return all_samples


def subsample_dpo(dataset, config: dict) -> list:
    """
    对 DPO 数据集进行子采样
    smoke_test: 取 1000 对
    full_run: 取 30K 对
    """
    target = config["dpo_sample_size"]
    seed = config["seed"]

    random.seed(seed)
    print(f"\nDPO 子采样 (目标: {target:,} 对)")
    print("-" * 50)

    total = len(dataset)
    actual = min(target, total)

    indices = list(range(total))
    random.shuffle(indices)
    selected = indices[:actual]

    samples = [dataset[i] for i in sorted(selected)]
    print(f"  采样: {actual:,} / {total:,}")

    return samples


def save_samples(samples: list, output_path: str, format_type: str = "jsonl"):
    """将采样数据保存为 JSONL 文件"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            # 确保可序列化
            clean_sample = {}
            for k, v in sample.items():
                try:
                    json.dumps(v)
                    clean_sample[k] = v
                except (TypeError, ValueError):
                    clean_sample[k] = str(v)
            f.write(json.dumps(clean_sample, ensure_ascii=False) + "\n")

    print(f"已保存 {len(samples):,} 条到 {output_path}")
