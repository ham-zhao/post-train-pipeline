"""
训练工具函数
模型加载、设备管理、训练回调等
"""

import os
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_name: str, dtype: str = "bfloat16",
                              device: str = "mps"):
    """
    加载模型和 tokenizer
    Qwen2.5-1.5B bf16 ≈ 3GB，128GB 内存完全无压力
    """
    print(f"加载模型: {model_name}")
    print(f"  dtype: {dtype}")
    print(f"  device: {device}")

    # dtype 映射
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # 确保有 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch_dtype,
        trust_remote_code=True,
    )

    # 移动到设备
    if device == "mps" and torch.backends.mps.is_available():
        model = model.to("mps")
    elif device == "cuda" and torch.cuda.is_available():
        model = model.to("cuda")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {param_count / 1e9:.2f}B")
    print(f"  显存占用: ~{param_count * 2 / 1e9:.1f}GB (bf16)")

    return model, tokenizer


def format_chat_prompt(messages: list, tokenizer) -> str:
    """
    使用 tokenizer 的 chat template 格式化对话
    Qwen2.5 有内置 chat template
    """
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        # 回退：手动拼接
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"<|system|>\n{content}")
            elif role == "user":
                parts.append(f"<|user|>\n{content}")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}")
        return "\n".join(parts)


def get_training_device():
    """获取训练设备，优先 MPS"""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def estimate_training_time(num_samples: int, batch_size: int, num_epochs: int,
                           steps_per_second: float = 1.5) -> str:
    """估算训练时间"""
    total_steps = (num_samples // batch_size) * num_epochs
    total_seconds = total_steps / steps_per_second
    hours = total_seconds / 3600
    if hours < 1:
        return f"约 {total_seconds / 60:.0f} 分钟"
    return f"约 {hours:.1f} 小时"


class TrainingLogger:
    """训练过程记录器，记录 loss 和指标用于后续可视化"""

    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.train_steps = []
        self.eval_steps = []
        self.chosen_rewards = []
        self.rejected_rewards = []
        self.reward_steps = []

    def log_train_loss(self, step: int, loss: float):
        self.train_steps.append(step)
        self.train_losses.append(loss)

    def log_eval_loss(self, step: int, loss: float):
        self.eval_steps.append(step)
        self.eval_losses.append(loss)

    def log_dpo_rewards(self, step: int, chosen_reward: float, rejected_reward: float):
        self.reward_steps.append(step)
        self.chosen_rewards.append(chosen_reward)
        self.rejected_rewards.append(rejected_reward)

    def get_summary(self) -> dict:
        return {
            "train_losses": self.train_losses,
            "train_steps": self.train_steps,
            "eval_losses": self.eval_losses,
            "eval_steps": self.eval_steps,
            "chosen_rewards": self.chosen_rewards,
            "rejected_rewards": self.rejected_rewards,
            "reward_steps": self.reward_steps,
        }
