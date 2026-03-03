"""
数据去污染检查器
检测训练数据与评估集的重叠，防止数据泄漏导致虚高分数

对应 Tülu 3 论文:
- Section 3.3: Data Decontamination
- 三种检测方法: 全字符串匹配、N-gram 重叠、embedding 相似度
- 发现部分公开数据集有 >5% 泄漏
"""

import json
import hashlib
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config_loader import PROJECT_ROOT


def extract_ngrams(text: str, n: int = 13) -> set:
    """提取文本的 N-gram 集合（默认 13-gram，与 Tülu 3 一致）"""
    text = text.lower().strip()
    words = text.split()
    if len(words) < n:
        return {tuple(words)}
    return {tuple(words[i:i+n]) for i in range(len(words) - n + 1)}


def exact_match(text1: str, text2: str) -> bool:
    """全字符串精确匹配（归一化后比较）"""
    def normalize(t):
        return " ".join(t.lower().split())
    return normalize(text1) == normalize(text2)


def ngram_overlap(text1: str, text2: str, n: int = 13, threshold: float = 0.7) -> float:
    """
    计算两段文本的 N-gram 重叠率
    threshold: 重叠率超过此值视为泄漏
    """
    ngrams1 = extract_ngrams(text1, n)
    ngrams2 = extract_ngrams(text2, n)

    if not ngrams1 or not ngrams2:
        return 0.0

    intersection = ngrams1 & ngrams2
    # Jaccard 相似度
    union = ngrams1 | ngrams2
    if not union:
        return 0.0
    return len(intersection) / len(union)


def load_eval_benchmarks() -> dict:
    """
    加载评估 benchmark 数据（用于去污染检测）
    从 lm-evaluation-harness 的数据集中提取
    """
    from datasets import load_dataset

    benchmarks = {}

    # HellaSwag
    try:
        ds = load_dataset("Rowan/hellaswag", split="validation")
        texts = [item["ctx"] for item in ds]
        benchmarks["hellaswag"] = texts[:500]  # 取前 500 条加速
        print(f"  HellaSwag: {len(benchmarks['hellaswag'])} 条")
    except Exception as e:
        print(f"  HellaSwag 加载失败: {e}")

    # ARC-Easy
    try:
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
        texts = [item["question"] for item in ds]
        benchmarks["arc_easy"] = texts[:500]
        print(f"  ARC-Easy: {len(benchmarks['arc_easy'])} 条")
    except Exception as e:
        print(f"  ARC-Easy 加载失败: {e}")

    # MMLU（取部分 subject）
    try:
        ds = load_dataset("cais/mmlu", "all", split="test")
        texts = [item["question"] for item in ds]
        benchmarks["mmlu"] = texts[:500]
        print(f"  MMLU: {len(benchmarks['mmlu'])} 条")
    except Exception as e:
        print(f"  MMLU 加载失败: {e}")

    return benchmarks


def check_contamination(train_data: list, benchmarks: dict,
                        n: int = 13, threshold: float = 0.7) -> dict:
    """
    检查训练数据与评估集的重叠

    Args:
        train_data: 训练数据列表（每条需包含文本内容）
        benchmarks: {"benchmark_name": [text1, text2, ...]}
        n: N-gram 大小
        threshold: 重叠阈值

    Returns:
        去污染报告
    """
    report = {
        "total_train_samples": len(train_data),
        "benchmarks": {},
        "contaminated_indices": set(),
    }

    # 提取训练数据文本
    train_texts = []
    for sample in train_data:
        if "messages" in sample:
            # SFT 格式
            text = " ".join(msg.get("content", "") for msg in sample["messages"])
        elif "prompt" in sample:
            # DPO 格式
            text = sample["prompt"]
        else:
            text = str(sample)
        train_texts.append(text)

    # 对每个 benchmark 做检测
    for bench_name, bench_texts in benchmarks.items():
        print(f"\n检测 {bench_name}...")

        # 先用 hash 做精确匹配（快速）
        bench_hashes = set()
        for t in bench_texts:
            h = hashlib.md5(" ".join(t.lower().split()).encode()).hexdigest()
            bench_hashes.add(h)

        exact_matches = 0
        ngram_matches = 0
        contaminated = []

        # 构建 benchmark 的 N-gram 索引（加速查询）
        bench_ngram_sets = [extract_ngrams(t, n) for t in bench_texts]
        # 合并所有 benchmark ngrams
        all_bench_ngrams = set()
        for ns in bench_ngram_sets:
            all_bench_ngrams.update(ns)

        for idx, train_text in enumerate(tqdm(train_texts, desc=f"  vs {bench_name}")):
            # 精确匹配
            train_hash = hashlib.md5(" ".join(train_text.lower().split()).encode()).hexdigest()
            if train_hash in bench_hashes:
                exact_matches += 1
                contaminated.append(idx)
                report["contaminated_indices"].add(idx)
                continue

            # N-gram 匹配（只检查有交集的）
            train_ngrams = extract_ngrams(train_text, n)
            if train_ngrams & all_bench_ngrams:
                # 有共同 ngram，做详细检查
                for bench_text in bench_texts:
                    overlap = ngram_overlap(train_text, bench_text, n, threshold)
                    if overlap >= threshold:
                        ngram_matches += 1
                        contaminated.append(idx)
                        report["contaminated_indices"].add(idx)
                        break

        total_contaminated = exact_matches + ngram_matches
        contamination_rate = total_contaminated / len(train_texts) * 100 if train_texts else 0

        report["benchmarks"][bench_name] = {
            "exact_matches": exact_matches,
            "ngram_matches": ngram_matches,
            "total_contaminated": total_contaminated,
            "contamination_rate": f"{contamination_rate:.2f}%",
            "contaminated_indices": contaminated,
        }

        print(f"  精确匹配: {exact_matches}")
        print(f"  N-gram 匹配: {ngram_matches}")
        print(f"  污染率: {contamination_rate:.2f}%")

    return report


def decontaminate(train_data: list, contaminated_indices: set) -> list:
    """去除被污染的训练样本"""
    clean_data = [
        sample for idx, sample in enumerate(train_data)
        if idx not in contaminated_indices
    ]
    removed = len(train_data) - len(clean_data)
    print(f"\n去污染: 移除 {removed} 条 / 保留 {len(clean_data)} 条")
    return clean_data


def print_contamination_report(report: dict):
    """打印格式化的去污染报告"""
    print("\n" + "=" * 60)
    print("去污染报告")
    print("=" * 60)
    print(f"训练数据总量: {report['total_train_samples']:,}")
    print(f"被污染样本数: {len(report['contaminated_indices'])}")
    print(f"总污染率: {len(report['contaminated_indices']) / report['total_train_samples'] * 100:.2f}%")

    print("\n各评估集污染详情:")
    print(f"{'评估集':<15} {'精确匹配':<10} {'N-gram':<10} {'总计':<10} {'污染率':<10}")
    print("-" * 55)
    for bench_name, info in report["benchmarks"].items():
        print(f"{bench_name:<15} {info['exact_matches']:<10} "
              f"{info['ngram_matches']:<10} {info['total_contaminated']:<10} "
              f"{info['contamination_rate']:<10}")

    # 对标 Tülu 3 论文发现
    print("\n📌 Tülu 3 论文发现: 部分公开数据集有 >5% 泄漏")
    print("   本项目使用 13-gram 匹配，threshold=0.7")
