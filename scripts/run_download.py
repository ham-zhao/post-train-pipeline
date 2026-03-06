#!/usr/bin/env python3
"""
数据下载与准备脚本（高效版）
使用 streaming 模式，只下载需要的数据量，避免一次性加载全量数据

用法: python scripts/run_download.py

产出文件：
  data/sft_mix/train.jsonl          - SFT 训练数据（统一格式）
  data/dpo_mix/train.jsonl          - DPO 偏好数据（prompt/chosen/rejected）
  results/reports/data_stats.json   - 数据统计报告（各 source 样本数）
  results/reports/decontamination_report.json - 去污染报告

SFT 数据格式 (allenai/tulu-3-sft-mixture):
  {"id": "...", "messages": [{"role": "user", "content": "..."}, ...], "source": "ai2-adapt-dev/..."}

DPO 数据格式 (allenai/llama-3.1-tulu-3-8b-preference-mixture):
  {"chosen": [messages], "rejected": [messages], "source": "..."}
"""

import sys
import json
import random
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tqdm import tqdm


# =============================================================================
# 配置
# =============================================================================

def load_config():
    """加载运行配置"""
    import yaml
    config_path = PROJECT_ROOT / "configs" / "run_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    run_mode = config["run_mode"]
    mode_config = config[run_mode]
    return {
        "run_mode": run_mode,
        "seed": config["seed"],
        **mode_config,
    }


# SFT source 名称映射：我们的简称 → 数据集中实际的 source 字段值
# 通过扫描 allenai/tulu-3-sft-mixture 全量 939K 条获得的实际 source 名称
SOURCE_MAPPING = {
    "flan_v2": "ai2-adapt-dev/flan_v2_converted",
    "openassistant": "ai2-adapt-dev/oasst1_converted",
    "no_robots": "ai2-adapt-dev/no_robots_converted",
    "numinamath": "ai2-adapt-dev/numinamath_tir_math_decontaminated",
    "wildguardmix": "ai2-adapt-dev/tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k",
    "wildjailbreak": "ai2-adapt-dev/tulu_v3.9_wildjailbreak_decontaminated_50k",
    "persona_if": "ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980",
    "persona_math": "allenai/tulu-3-sft-personas-math-grade",
    "persona_code": "ai2-adapt-dev/personahub_code_v2_34999",
    "coconot": "ai2-adapt-dev/coconot_converted",
    # 额外子集（原始 spec 未列但数据集中存在）
    "wildchat": "ai2-adapt-dev/tulu_v3.9_wildchat_100k",
    "aya": "ai2-adapt-dev/tulu_v3.9_aya_100k",
    "evol_code": "ai2-adapt-dev/evol_codealpaca_heval_decontaminated",
    "gsm8k": "ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k",
    "personahub_math_v5": "ai2-adapt-dev/personahub_math_v5_regen_149960",
    "personahub_math_algebra": "ai2-adapt-dev/tulu_v3.9_personahub_math_interm_algebra_20k",
    "sciriff": "ai2-adapt-dev/tulu_v3.9_sciriff_10k",
    "table_gpt": "ai2-adapt-dev/tulu_v3.9_table_gpt_5k",
    "hard_coded": "ai2-adapt-dev/tulu_hard_coded_repeated_10",
}

# 反向映射
REVERSE_SOURCE_MAPPING = {v: k for k, v in SOURCE_MAPPING.items()}

# 各子集的 full_run 目标量（对应 Tülu 3 论文 Table 2 的配比）
SUBSET_TARGETS = {
    "flan_v2": 10000,
    "numinamath": 5000,
    "wildguardmix": 10000,
    "wildjailbreak": 10000,
    "persona_if": 5000,
    "persona_math": 3000,
    "persona_code": 2000,
    "coconot": 5000,
    "no_robots": 9500,
    "openassistant": 7132,
}


# =============================================================================
# Step 1: 下载 SFT 数据（streaming 模式）
# =============================================================================

def download_sft_data(config: dict) -> list:
    """
    使用 streaming 下载 SFT 数据，按 source 分桶采样
    """
    from datasets import load_dataset

    target_total = config["sft_sample_size"]
    seed = config["seed"]
    run_mode = config["run_mode"]
    random.seed(seed)

    print(f"\n{'='*60}")
    print(f"下载 SFT 数据 (目标: {target_total:,} 条)")
    print(f"{'='*60}")

    # 计算每个子集的目标数量
    if run_mode == "smoke_test":
        full_total = sum(SUBSET_TARGETS.values())
        scale = target_total / full_total
        targets = {k: max(10, int(v * scale)) for k, v in SUBSET_TARGETS.items()}
    else:
        targets = SUBSET_TARGETS.copy()

    print("\n各子集采样目标:")
    for name, target in targets.items():
        print(f"  {name}: {target}")

    # streaming 下载，按 source 分桶
    print("\n开始 streaming 下载...")
    ds = load_dataset("allenai/tulu-3-sft-mixture", split="train", streaming=True)

    buckets = defaultdict(list)  # source_key -> [samples]
    source_counts = Counter()
    all_filled = False
    processed = 0

    for item in tqdm(ds, desc="Streaming SFT"):
        processed += 1
        raw_source = item.get("source", "unknown")

        # 映射到我们的键名
        source_key = REVERSE_SOURCE_MAPPING.get(raw_source)
        if source_key is None:
            # 尝试模糊匹配
            for key, hf_id in SOURCE_MAPPING.items():
                if hf_id.lower() in raw_source.lower() or raw_source.lower() in hf_id.lower():
                    source_key = key
                    break
        if source_key is None:
            source_key = raw_source  # 保留原始 source

        source_counts[source_key] += 1

        # 如果该子集还需要更多样本
        target = targets.get(source_key, 0)
        if target > 0 and len(buckets[source_key]) < target:
            # SFT 数据已经是 messages 格式，直接使用
            sample = {
                "messages": item["messages"],
                "source": source_key,
            }
            buckets[source_key].append(sample)

        # 检查是否所有桶都满了
        all_filled = all(
            len(buckets.get(k, [])) >= t
            for k, t in targets.items()
        )
        if all_filled:
            break

        # 安全上限：处理了足够多的数据
        if processed > 950000:
            print(f"\n达到处理上限 ({processed:,} 条)，停止")
            break

    # 汇总
    print(f"\n处理了 {processed:,} 条原始数据")
    print(f"\n数据源分布 (已采样):")
    all_samples = []
    for source_key, target in targets.items():
        actual = len(buckets.get(source_key, []))
        all_samples.extend(buckets.get(source_key, []))
        status = "OK" if actual >= target else f"不足 (差 {target - actual})"
        print(f"  {source_key}: {actual:,} / {target:,}  {status}")

    # 对于未映射到目标的 source，也收集一些
    for key, samples in buckets.items():
        if key not in targets:
            print(f"  {key}: {len(samples):,} (额外)")

    random.shuffle(all_samples)
    print(f"\nSFT 总计: {len(all_samples):,} 条")
    return all_samples


# =============================================================================
# Step 2: 下载 DPO 数据（streaming 模式）
# =============================================================================

def download_dpo_data(config: dict) -> list:
    """
    使用 streaming 下载 DPO 偏好数据
    """
    from datasets import load_dataset

    target = config["dpo_sample_size"]
    seed = config["seed"]
    random.seed(seed)

    print(f"\n{'='*60}")
    print(f"下载 DPO 数据 (目标: {target:,} 对)")
    print(f"{'='*60}")

    # streaming 模式下载
    print("开始 streaming 下载...")
    ds = load_dataset(
        "allenai/llama-3.1-tulu-3-8b-preference-mixture",
        split="train",
        streaming=True,
    )

    samples = []
    failed = 0

    for item in tqdm(ds, desc="Streaming DPO", total=target):
        # 解析偏好对
        dpo_sample = convert_dpo_item(item)
        if dpo_sample:
            samples.append(dpo_sample)
        else:
            failed += 1

        if len(samples) >= target:
            break

    print(f"\nDPO 采样完成: {len(samples):,} 对 (失败: {failed})")

    # 打印前 2 条样本预览
    for i in range(min(2, len(samples))):
        s = samples[i]
        print(f"\n--- DPO 样本 {i+1} ---")
        print(f"  prompt: {s['prompt'][:100]}...")
        print(f"  chosen: {s['chosen'][:100]}...")
        print(f"  rejected: {s['rejected'][:100]}...")

    return samples


def convert_dpo_item(item: dict) -> dict:
    """
    将 Tülu 3 DPO 数据项转换为 {prompt, chosen, rejected} 格式

    Tülu 3 偏好数据格式可能是:
    - chosen/rejected 为 messages 列表: [{"role": "user", ...}, {"role": "assistant", ...}]
    - 或直接的 prompt/chosen/rejected 字符串
    """
    try:
        chosen_data = item.get("chosen", [])
        rejected_data = item.get("rejected", [])

        # 如果 chosen/rejected 是 messages 列表
        if isinstance(chosen_data, list) and len(chosen_data) > 0:
            if isinstance(chosen_data[0], dict):
                # 提取 user 消息作为 prompt
                prompt_parts = []
                chosen_response = ""
                for msg in chosen_data:
                    if msg.get("role") in ("user", "human"):
                        prompt_parts.append(msg.get("content", ""))
                    elif msg.get("role") in ("assistant", "gpt"):
                        chosen_response = msg.get("content", "")

                rejected_response = ""
                for msg in rejected_data:
                    if msg.get("role") in ("assistant", "gpt"):
                        rejected_response = msg.get("content", "")

                prompt = "\n".join(prompt_parts)
                if prompt and chosen_response and rejected_response:
                    return {
                        "prompt": prompt,
                        "chosen": chosen_response,
                        "rejected": rejected_response,
                        "source": item.get("source", "tulu-3-preference"),
                    }

        # 如果已经是字符串格式
        elif isinstance(chosen_data, str):
            prompt = item.get("prompt", item.get("question", ""))
            if prompt and chosen_data and rejected_data:
                return {
                    "prompt": str(prompt),
                    "chosen": str(chosen_data),
                    "rejected": str(rejected_data),
                    "source": item.get("source", "tulu-3-preference"),
                }

        return None
    except Exception:
        return None


# =============================================================================
# Step 3: 去污染
# =============================================================================

def run_decontamination(sft_data: list, config: dict) -> tuple:
    """
    对 SFT 数据做去污染检查
    使用 13-gram 匹配（与 Tülu 3 一致）
    """
    from datasets import load_dataset

    print(f"\n{'='*60}")
    print("去污染检查")
    print(f"{'='*60}")

    # 加载评估集
    benchmarks = {}

    print("加载评估基准集...")
    eval_datasets = [
        ("hellaswag", "Rowan/hellaswag", "validation", "ctx"),
        ("arc_easy", "allenai/ai2_arc", "ARC-Easy", "question"),
    ]

    for bench_name, hf_id, split_or_config, text_field in eval_datasets:
        try:
            if bench_name == "arc_easy":
                ds = load_dataset(hf_id, split_or_config, split="test", streaming=True)
            else:
                ds = load_dataset(hf_id, split=split_or_config, streaming=True)

            texts = []
            for i, item in enumerate(ds):
                texts.append(item[text_field])
                if i >= 499:
                    break
            benchmarks[bench_name] = texts
            print(f"  {bench_name}: {len(texts)} 条")
        except Exception as e:
            print(f"  {bench_name} 加载失败: {e}")

    if not benchmarks:
        print("无法加载评估集，跳过去污染")
        return sft_data, {"skipped": True}

    # N-gram 匹配
    N = 13
    THRESHOLD = 0.7
    contaminated_indices = set()
    report = {"total": len(sft_data), "benchmarks": {}}

    for bench_name, bench_texts in benchmarks.items():
        print(f"\n检测 vs {bench_name}...")

        # 构建 benchmark N-gram 集合
        bench_ngrams_all = set()
        for text in bench_texts:
            words = text.lower().split()
            for i in range(len(words) - N + 1):
                bench_ngrams_all.add(tuple(words[i:i+N]))

        matches = 0
        for idx, sample in enumerate(tqdm(sft_data, desc=f"  vs {bench_name}")):
            # 提取训练样本文本
            text = " ".join(msg.get("content", "") for msg in sample.get("messages", []))
            words = text.lower().split()

            # 快速检查：是否有任何 N-gram 与 benchmark 重叠
            has_overlap = False
            for i in range(len(words) - N + 1):
                if tuple(words[i:i+N]) in bench_ngrams_all:
                    has_overlap = True
                    break

            if has_overlap:
                contaminated_indices.add(idx)
                matches += 1

        rate = matches / len(sft_data) * 100 if sft_data else 0
        report["benchmarks"][bench_name] = {
            "matches": matches,
            "rate": f"{rate:.2f}%",
        }
        print(f"  匹配: {matches} ({rate:.2f}%)")

    # 去除被污染样本
    clean_data = [s for i, s in enumerate(sft_data) if i not in contaminated_indices]
    removed = len(sft_data) - len(clean_data)
    report["removed"] = removed
    report["clean_total"] = len(clean_data)

    print(f"\n去污染结果: 移除 {removed} 条，保留 {len(clean_data)} 条")
    print(f"总污染率: {removed / len(sft_data) * 100:.2f}%")

    return clean_data, report


# =============================================================================
# Step 4: 保存
# =============================================================================

def save_jsonl(data: list, output_path: str):
    """保存为 JSONL"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"已保存 {len(data):,} 条到 {path}")


# =============================================================================
# 主函数
# =============================================================================

def main():
    config = load_config()
    print(f"运行模式: {config['run_mode']}")
    print(f"SFT 目标: {config['sft_sample_size']:,} 条")
    print(f"DPO 目标: {config['dpo_sample_size']:,} 对")

    # Step 1: 下载 SFT 数据
    sft_data = download_sft_data(config)

    # Step 2: 下载 DPO 数据
    dpo_data = download_dpo_data(config)

    # Step 3: 去污染
    sft_clean, decontam_report = run_decontamination(sft_data, config)

    # Step 4: 保存
    print(f"\n{'='*60}")
    print("保存数据")
    print(f"{'='*60}")

    sft_output = str(PROJECT_ROOT / "data/sft_mix/train.jsonl")
    dpo_output = str(PROJECT_ROOT / "data/dpo_mix/train.jsonl")

    save_jsonl(sft_clean, sft_output)
    save_jsonl(dpo_data, dpo_output)

    # 保存去污染报告
    report_path = PROJECT_ROOT / "results/reports/decontamination_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(decontam_report, f, indent=2, ensure_ascii=False)

    # 保存数据统计
    sft_source_counts = Counter(s.get("source", "unknown") for s in sft_clean)
    stats = {
        "run_mode": config["run_mode"],
        "sft_total": len(sft_clean),
        "dpo_total": len(dpo_data),
        "sft_by_source": dict(sft_source_counts.most_common()),
        "decontamination": decontam_report,
    }
    stats_path = PROJECT_ROOT / "results/reports/data_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # 总结
    print(f"\n{'='*60}")
    print("数据准备完成！")
    print(f"{'='*60}")
    print(f"SFT 数据: {sft_output} ({len(sft_clean):,} 条)")
    print(f"DPO 数据: {dpo_output} ({len(dpo_data):,} 对)")
    print(f"统计报告: {stats_path}")
    print(f"\n各 source 统计:")
    for src, cnt in sft_source_counts.most_common():
        print(f"  {src}: {cnt:,}")
    print(f"\n下一步: python scripts/run_sft.py")


if __name__ == "__main__":
    main()
