"""
配置加载器
从 configs/ 目录加载 YAML 配置文件，根据 run_mode 自动切换参数
"""

import os
import yaml
from pathlib import Path


# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_yaml(config_name: str) -> dict:
    """加载指定的 YAML 配置文件"""
    config_path = PROJECT_ROOT / "configs" / config_name
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_run_config() -> dict:
    """
    加载运行配置，根据 run_mode 合并对应的参数
    返回包含当前模式参数的扁平字典
    """
    config = load_yaml("run_config.yaml")
    run_mode = config["run_mode"]
    mode_config = config[run_mode]

    return {
        "run_mode": run_mode,
        "model": config["model"],
        "seed": config["seed"],
        "sft_subsets": config["sft_subsets"],
        "dpo_dataset": config["dpo_dataset"],
        **mode_config,
    }


def get_sft_config() -> dict:
    """加载 SFT 训练配置"""
    return load_yaml("sft_config.yaml")


def get_dpo_config() -> dict:
    """加载 DPO 训练配置"""
    return load_yaml("dpo_config.yaml")


def get_eval_config() -> dict:
    """加载评估配置"""
    return load_yaml("eval_config.yaml")


def get_device():
    """
    获取可用设备，优先 MPS，回退 CPU
    对应项目约束：统一 MPS backend，不支持时自动回退
    """
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    print("警告: MPS 不可用，回退到 CPU")
    return "cpu"


def get_hf_endpoint() -> str:
    """获取 HuggingFace endpoint，支持代理访问"""
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co")
