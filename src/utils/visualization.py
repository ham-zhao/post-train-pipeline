"""
可视化工具
提供训练曲线、数据分布、评估对比等图表生成函数
"""

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
from pathlib import Path

# 支持中文显示
matplotlib.rcParams['font.family'] = ['Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

# 统一配色方案
COLORS = {
    "base": "#95a5a6",      # 灰色 - Base Model
    "sft": "#3498db",       # 蓝色 - SFT
    "dpo": "#e74c3c",       # 红色 - DPO
    "safety": "#e67e22",    # 橙色 - 安全相关
    "ability": "#2ecc71",   # 绿色 - 能力相关
}

STAGE_COLORS = [COLORS["base"], COLORS["sft"], COLORS["dpo"]]


def plot_training_loss(losses: list, steps: list = None, title: str = "Training Loss",
                       save_path: str = None):
    """绘制训练 loss 曲线"""
    fig, ax = plt.subplots(figsize=(10, 5))
    if steps is None:
        steps = list(range(len(losses)))
    ax.plot(steps, losses, color=COLORS["sft"], linewidth=1.5)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_dpo_margins(chosen_rewards: list, rejected_rewards: list,
                     steps: list = None, save_path: str = None):
    """绘制 DPO reward margin 曲线（chosen - rejected 应逐步增大）"""
    fig, ax = plt.subplots(figsize=(10, 5))
    if steps is None:
        steps = list(range(len(chosen_rewards)))
    margins = [c - r for c, r in zip(chosen_rewards, rejected_rewards)]
    ax.plot(steps, chosen_rewards, color=COLORS["ability"], label="Chosen reward", linewidth=1.5)
    ax.plot(steps, rejected_rewards, color=COLORS["safety"], label="Rejected reward", linewidth=1.5)
    ax.fill_between(steps, rejected_rewards, chosen_rewards, alpha=0.15, color=COLORS["dpo"])
    ax.set_xlabel("Steps")
    ax.set_ylabel("Reward")
    ax.set_title("DPO Reward Margin (Chosen - Rejected)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_data_composition(subset_names: list, subset_sizes: list,
                          categories: list = None, save_path: str = None):
    """绘制数据集构成堆叠柱状图"""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette("husl", len(subset_names))
    bars = ax.barh(subset_names, subset_sizes, color=colors)

    # 在柱状图上标注数量
    for bar, size in zip(bars, subset_sizes):
        ax.text(bar.get_width() + 100, bar.get_y() + bar.get_height() / 2,
                f'{size:,}', va='center', fontsize=9)

    ax.set_xlabel("样本数量")
    ax.set_title("SFT 数据集构成")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_length_distribution(lengths_dict: dict, title: str = "长度分布",
                             save_path: str = None):
    """绘制多组数据的长度分布对比"""
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, lengths in lengths_dict.items():
        ax.hist(lengths, bins=50, alpha=0.5, label=label, density=True)
    ax.set_xlabel("Token 长度")
    ax.set_ylabel("密度")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_radar_chart(categories: list, scores_dict: dict, save_path: str = None):
    """
    绘制能力雷达图（Base vs SFT vs DPO）
    categories: 维度名称列表
    scores_dict: {"Base": [s1,s2,...], "SFT": [...], "DPO": [...]}
    """
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, (label, scores) in enumerate(scores_dict.items()):
        values = scores + scores[:1]  # 闭合
        color = STAGE_COLORS[i] if i < len(STAGE_COLORS) else f'C{i}'
        ax.plot(angles, values, 'o-', linewidth=2, label=label, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    ax.set_title("能力雷达图: Base vs SFT vs DPO", size=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_safety_comparison(metrics: dict, save_path: str = None):
    """
    绘制安全指标柱状图
    metrics: {"Base": {"ASR": x, "Over-refusal": y}, "SFT": ..., "DPO": ...}
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    stages = list(metrics.keys())

    # ASR（越低越好）
    asr_values = [metrics[s].get("ASR", 0) for s in stages]
    axes[0].bar(stages, asr_values, color=STAGE_COLORS[:len(stages)])
    axes[0].set_title("Attack Success Rate ↓ (越低越好)")
    axes[0].set_ylabel("ASR (%)")
    for i, v in enumerate(asr_values):
        axes[0].text(i, v + 0.5, f'{v:.1f}%', ha='center')

    # Over-refusal（越低越好）
    or_values = [metrics[s].get("Over-refusal", 0) for s in stages]
    axes[1].bar(stages, or_values, color=STAGE_COLORS[:len(stages)])
    axes[1].set_title("Over-refusal Rate ↓ (越低越好)")
    axes[1].set_ylabel("Over-refusal (%)")
    for i, v in enumerate(or_values):
        axes[1].text(i, v + 0.5, f'{v:.1f}%', ha='center')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_dashboard(train_losses_sft, train_losses_dpo, dpo_margins,
                   radar_data, safety_data, save_path=None):
    """
    生成四象限 Dashboard（可导出 PNG）
    左上：能力雷达图 | 右上：安全指标柱状图
    左下：SFT loss   | 右下：DPO margin
    """
    fig = plt.figure(figsize=(16, 12))

    # 左上：雷达图
    ax1 = fig.add_subplot(221, polar=True)
    categories = list(radar_data["categories"])
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    for i, (label, scores) in enumerate(radar_data["scores"].items()):
        values = scores + scores[:1]
        color = STAGE_COLORS[i] if i < len(STAGE_COLORS) else f'C{i}'
        ax1.plot(angles, values, 'o-', linewidth=2, label=label, color=color)
        ax1.fill(angles, values, alpha=0.1, color=color)
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(categories, size=9)
    ax1.set_title("能力雷达图", pad=15)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)

    # 右上：安全柱状图
    ax2 = fig.add_subplot(222)
    stages = list(safety_data.keys())
    x = np.arange(len(stages))
    width = 0.35
    asr = [safety_data[s].get("ASR", 0) for s in stages]
    ore = [safety_data[s].get("Over-refusal", 0) for s in stages]
    ax2.bar(x - width / 2, asr, width, label='ASR ↓', color='#e74c3c', alpha=0.8)
    ax2.bar(x + width / 2, ore, width, label='Over-refusal ↓', color='#f39c12', alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(stages)
    ax2.set_title("安全指标对比")
    ax2.set_ylabel("%")
    ax2.legend()

    # 左下：SFT loss
    ax3 = fig.add_subplot(223)
    ax3.plot(train_losses_sft, color=COLORS["sft"], linewidth=1.5)
    ax3.set_title("SFT Training Loss")
    ax3.set_xlabel("Steps")
    ax3.set_ylabel("Loss")
    ax3.grid(True, alpha=0.3)

    # 右下：DPO margin
    ax4 = fig.add_subplot(224)
    ax4.plot(dpo_margins, color=COLORS["dpo"], linewidth=1.5)
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax4.set_title("DPO Reward Margin (Chosen - Rejected)")
    ax4.set_xlabel("Steps")
    ax4.set_ylabel("Margin")
    ax4.grid(True, alpha=0.3)

    plt.suptitle("Post-Training Pipeline Dashboard", fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
