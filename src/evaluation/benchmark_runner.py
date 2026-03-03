"""
Benchmark 评估运行器
使用 lm-evaluation-harness 框架评估模型

对应 Tülu 3 论文:
- Section 5: Evaluation
- 0-shot 评估，多个 benchmark
"""

import json
import subprocess
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config_loader import get_run_config, get_eval_config, PROJECT_ROOT


def run_lm_eval(model_path: str, tasks: str, output_dir: str,
                batch_size: int = 8, limit: int = None,
                device: str = "mps", num_fewshot: int = 0) -> dict:
    """
    运行 lm-evaluation-harness 评估

    Args:
        model_path: 模型路径（本地或 HuggingFace ID）
        tasks: 逗号分隔的评估任务
        output_dir: 结果输出目录
        batch_size: 评估 batch size
        limit: 每个任务的样本上限
        device: 设备
        num_fewshot: few-shot 数量（Tülu 3 用 0-shot）

    Returns:
        评估结果字典
    """
    print(f"\n运行 lm-eval:")
    print(f"  模型: {model_path}")
    print(f"  任务: {tasks}")
    print(f"  设备: {device}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 构建命令
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},trust_remote_code=True",
        "--tasks", tasks,
        "--batch_size", str(batch_size),
        "--num_fewshot", str(num_fewshot),
        "--output_path", output_dir,
        "--device", device,
    ]

    if limit and limit > 0:
        cmd.extend(["--limit", str(limit)])

    print(f"  命令: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 小时超时
        )

        if result.returncode != 0:
            print(f"  评估出错: {result.stderr[:500]}")
            # 尝试从输出中提取结果
            return parse_lm_eval_output(result.stdout)

        return parse_lm_eval_output(result.stdout)

    except subprocess.TimeoutExpired:
        print("  评估超时（>1小时）")
        return {}
    except Exception as e:
        print(f"  评估异常: {e}")
        return {}


def parse_lm_eval_output(output: str) -> dict:
    """解析 lm-eval 的输出，提取各任务分数"""
    results = {}

    for line in output.split("\n"):
        line = line.strip()
        # lm-eval 输出格式: |task|...|metric|...|value|...|
        if "|" in line and "acc" in line.lower():
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) >= 4:
                task = parts[0]
                try:
                    # 找到数值列
                    for p in parts[1:]:
                        try:
                            value = float(p)
                            if 0 <= value <= 1:
                                results[task] = value
                                break
                        except ValueError:
                            continue
                except (ValueError, IndexError):
                    pass

    return results


def parse_lm_eval_json(output_dir: str) -> dict:
    """从 lm-eval 保存的 JSON 结果文件中提取分数（更可靠）"""
    import glob

    results = {}
    pattern = f"{output_dir}/**/results_*.json"
    json_files = glob.glob(pattern, recursive=True)

    for json_file in json_files:
        try:
            with open(json_file) as f:
                data = json.load(f)
            for task, metrics in data.get("results", {}).items():
                # 优先取 acc_norm，其次 acc，再 mc2
                if f"acc_norm,none" in metrics:
                    results[task] = metrics[f"acc_norm,none"]
                elif f"acc,none" in metrics:
                    results[task] = metrics[f"acc,none"]
                elif f"mc2,none" in metrics:
                    results[task] = metrics[f"mc2,none"]
        except Exception:
            continue

    return results


def run_eval_pipeline(model_path: str, stage_name: str = "model") -> dict:
    """
    运行完整评估流水线

    Args:
        model_path: 模型路径
        stage_name: 阶段名称（base/sft/dpo），用于结果文件命名

    Returns:
        评估结果
    """
    run_config = get_run_config()
    eval_config = get_eval_config()

    tasks = run_config.get("eval_tasks", "hellaswag")
    limit = run_config.get("eval_limit", -1)
    batch_size = eval_config["evaluation"]["batch_size"]
    device = eval_config["evaluation"]["device"]

    output_dir = str(PROJECT_ROOT / f"results/eval_results/{stage_name}")

    results = run_lm_eval(
        model_path=model_path,
        tasks=tasks,
        output_dir=output_dir,
        batch_size=batch_size,
        limit=limit if limit > 0 else None,
        device=device,
    )

    # 如果 stdout 解析结果不可靠，从 lm-eval JSON 文件提取
    json_results = parse_lm_eval_json(output_dir)
    if json_results:
        results = json_results

    # 保存结果
    result_file = Path(output_dir) / "scores.json"
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{stage_name} 评估结果:")
    for task, score in results.items():
        print(f"  {task}: {score:.4f}")

    return results


def compare_stages(results: dict) -> str:
    """
    生成三方对比表（Base vs SFT vs DPO）

    Args:
        results: {"base": {...}, "sft": {...}, "dpo": {...}}

    Returns:
        格式化的对比表字符串
    """
    stages = ["base", "sft", "dpo"]
    all_tasks = set()
    for stage_results in results.values():
        all_tasks.update(stage_results.keys())

    # 表头
    header = f"{'指标':<20}" + "".join(f"{s:<15}" for s in stages) + f"{'LIFT (DPO-Base)':<18}"
    separator = "-" * len(header)

    lines = [separator, header, separator]

    for task in sorted(all_tasks):
        row = f"{task:<20}"
        values = []
        for stage in stages:
            val = results.get(stage, {}).get(task, None)
            if val is not None:
                row += f"{val:<15.4f}"
                values.append(val)
            else:
                row += f"{'N/A':<15}"
                values.append(None)

        # 计算 LIFT
        if values[0] is not None and values[-1] is not None:
            lift = values[-1] - values[0]
            row += f"{lift:+.4f}"
        else:
            row += "N/A"

        lines.append(row)

    lines.append(separator)
    return "\n".join(lines)
