#!/usr/bin/env python3
"""
生成三方对比报告和 Dashboard
用法: python scripts/generate_comparison.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.stage_comparator import (
    generate_comparison_report, generate_dashboard
)


def main():
    print("生成三方对比报告...")
    report = generate_comparison_report()

    # 打印对比表
    print("\n" + report["comparison_table"])

    # 生成 Dashboard
    print("\n生成 Dashboard...")
    try:
        generate_dashboard(
            report["results"],
            save_path=str(PROJECT_ROOT / "results/figures/dashboard.png"),
        )
    except Exception as e:
        print(f"Dashboard 生成出错: {e}")

    print("\n报告生成完成！")
    print(f"  对比表: results/eval_results/")
    print(f"  Dashboard: results/figures/dashboard.png")


if __name__ == "__main__":
    main()
