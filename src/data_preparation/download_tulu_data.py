"""
Tülu 3 数据集下载器
从 HuggingFace 下载 SFT 和 DPO 数据集，打印统计信息

对应 Tülu 3 论文:
- SFT 数据: Section 3, Table 2
- DPO 数据: Section 4.1
"""

import json
import os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config_loader import get_run_config, PROJECT_ROOT


def download_sft_mixture(config: dict):
    """
    下载 Tülu 3 SFT 混合数据集
    allenai/tulu-3-sft-mixture 是官方预混合版本
    """
    print("=" * 60)
    print("下载 SFT 混合数据集: allenai/tulu-3-sft-mixture")
    print("=" * 60)

    try:
        # 尝试直接加载官方混合数据集
        dataset = load_dataset("allenai/tulu-3-sft-mixture", split="train")
        print(f"\n总样本数: {len(dataset):,}")
        print(f"字段: {dataset.column_names}")

        # 打印各子集统计
        if "dataset" in dataset.column_names:
            from collections import Counter
            subset_counts = Counter(dataset["dataset"])
            print("\n各子集规模:")
            for name, count in sorted(subset_counts.items(), key=lambda x: -x[1]):
                print(f"  {name}: {count:,}")

        return dataset

    except Exception as e:
        print(f"\n直接加载混合数据集失败: {e}")
        print("将逐个下载子数据集...")
        return download_sft_subsets(config)


def download_sft_subsets(config: dict):
    """逐个下载 SFT 子数据集（备选方案）"""
    all_samples = []
    sft_subsets = config["sft_subsets"]

    for subset_name, subset_info in tqdm(sft_subsets.items(), desc="下载 SFT 子集"):
        hf_id = subset_info["hf_id"]
        print(f"\n--- 下载 {subset_name}: {hf_id} ---")
        try:
            ds = load_dataset(hf_id, split="train")
            print(f"  规模: {len(ds):,}")
            print(f"  字段: {ds.column_names}")

            # 将每条样本标记来源
            for item in ds:
                item["source"] = subset_name
                all_samples.append(item)

        except Exception as e:
            print(f"  下载失败: {e}")
            continue

    print(f"\n总计下载: {len(all_samples):,} 条")
    return all_samples


def download_dpo_mixture(config: dict):
    """
    下载 Tülu 3 DPO 偏好数据集
    allenai/llama-3.1-tulu-3-8b-preference-mixture
    """
    print("\n" + "=" * 60)
    print("下载 DPO 偏好数据集")
    print("=" * 60)

    dpo_info = config["dpo_dataset"]
    hf_id = dpo_info["hf_id"]

    try:
        dataset = load_dataset(hf_id, split="train")
        print(f"\n总样本数: {len(dataset):,}")
        print(f"字段: {dataset.column_names}")

        # 打印前 3 条样本的字段值（截断显示）
        print("\n前 3 条样本预览:")
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            print(f"\n--- 样本 {i+1} ---")
            for key, value in sample.items():
                val_str = str(value)[:200]
                print(f"  {key}: {val_str}...")

        return dataset

    except Exception as e:
        print(f"下载失败: {e}")
        raise


def print_dataset_summary(sft_data, dpo_data):
    """打印数据集总结报告"""
    print("\n" + "=" * 60)
    print("数据集下载总结")
    print("=" * 60)

    sft_size = len(sft_data) if hasattr(sft_data, '__len__') else "未知"
    dpo_size = len(dpo_data) if hasattr(dpo_data, '__len__') else "未知"

    print(f"SFT 数据: {sft_size} 条")
    print(f"DPO 数据: {dpo_size} 对")
    print(f"\n数据将保存到:")
    print(f"  SFT: data/sft_mix/")
    print(f"  DPO: data/dpo_mix/")


if __name__ == "__main__":
    config = get_run_config()
    print(f"运行模式: {config['run_mode']}")
    sft_data = download_sft_mixture(config)
    dpo_data = download_dpo_mixture(config)
    print_dataset_summary(sft_data, dpo_data)
