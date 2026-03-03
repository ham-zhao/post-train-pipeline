"""
DPO 训练器
使用 TRL DPOTrainer 进行直接偏好优化

对应 Tülu 3 论文:
- Section 4: Preference Tuning with DPO
- β=5, lr=5e-7, epochs=3
- DPO 合并了 RM 训练 + Policy 优化为一步
"""

import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import TrainingArguments, TrainerCallback
from trl import DPOTrainer, DPOConfig

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config_loader import get_run_config, get_dpo_config, PROJECT_ROOT
from src.training.training_utils import load_model_and_tokenizer, TrainingLogger


class DPOLoggerCallback(TrainerCallback):
    """回调：记录 DPO 训练指标"""

    def __init__(self, logger: TrainingLogger):
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            if "loss" in logs:
                self.logger.log_train_loss(state.global_step, logs["loss"])
            if "eval_loss" in logs:
                self.logger.log_eval_loss(state.global_step, logs["eval_loss"])
            # DPO 特有指标
            chosen_reward = logs.get("rewards/chosen", logs.get("rewards_chosen"))
            rejected_reward = logs.get("rewards/rejected", logs.get("rewards_rejected"))
            if chosen_reward is not None and rejected_reward is not None:
                self.logger.log_dpo_rewards(
                    state.global_step,
                    float(chosen_reward),
                    float(rejected_reward),
                )


def prepare_dpo_dataset(data_path: str, tokenizer, eval_split: float = 0.05):
    """
    准备 DPO 训练数据集
    格式: {"prompt": ..., "chosen": ..., "rejected": ...}
    """
    print(f"加载 DPO 数据: {data_path}")

    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                # DPOTrainer 需要 prompt/chosen/rejected 字段
                if all(k in sample for k in ["prompt", "chosen", "rejected"]):
                    samples.append({
                        "prompt": str(sample["prompt"]),
                        "chosen": str(sample["chosen"]),
                        "rejected": str(sample["rejected"]),
                    })

    print(f"  有效样本数: {len(samples):,}")

    dataset = Dataset.from_list(samples)

    # 分割
    split = dataset.train_test_split(test_size=eval_split, seed=42)
    print(f"  训练集: {len(split['train']):,}")
    print(f"  验证集: {len(split['test']):,}")

    return split["train"], split["test"]


def run_dpo_training(
    sft_model_path: str = None,
    data_path: str = None,
    output_dir: str = None,
    max_steps: int = -1,
):
    """
    执行 DPO 训练

    Args:
        sft_model_path: SFT 模型路径（作为初始化和参考模型）
        data_path: DPO 训练数据路径
        output_dir: 输出目录
        max_steps: 最大训练步数

    Returns:
        (model, tokenizer, training_logger)
    """
    # 加载配置
    run_config = get_run_config()
    dpo_config = get_dpo_config()

    sft_model_path = sft_model_path or str(PROJECT_ROOT / dpo_config["ref_model"]["path"])
    data_path = data_path or str(PROJECT_ROOT / dpo_config["data"]["train_file"])
    output_dir = output_dir or str(PROJECT_ROOT / dpo_config["output"]["dir"])
    log_dir = str(PROJECT_ROOT / dpo_config["output"]["logging_dir"])

    if max_steps == -1:
        max_steps = run_config.get("max_steps", -1)

    # 加载 SFT 模型（作为训练起点）
    print("加载 SFT 模型作为 DPO 起点...")
    model, tokenizer = load_model_and_tokenizer(
        sft_model_path,
        dtype=run_config["model"]["dtype"],
        device="cpu",  # DPOTrainer 自己管理设备
    )

    # 加载参考模型（SFT 的冻结副本）
    print("加载参考模型（SFT 冻结副本）...")
    ref_model, _ = load_model_and_tokenizer(
        sft_model_path,
        dtype=run_config["model"]["dtype"],
        device="cpu",
    )

    # 准备数据
    train_dataset, eval_dataset = prepare_dpo_dataset(
        data_path, tokenizer,
        eval_split=dpo_config["data"]["eval_split"],
    )

    # DPO 训练参数
    training = dpo_config["training"]
    logging_config = dpo_config["logging"]
    dpo_params = dpo_config["dpo"]

    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=training["num_train_epochs"],
        per_device_train_batch_size=training["per_device_train_batch_size"],
        per_device_eval_batch_size=training["per_device_train_batch_size"],
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        learning_rate=training["learning_rate"],
        warmup_ratio=training["warmup_ratio"],
        weight_decay=training["weight_decay"],
        lr_scheduler_type=training["lr_scheduler_type"],
        logging_steps=logging_config["logging_steps"],
        eval_steps=logging_config["eval_steps"],
        save_steps=logging_config.get("save_steps", logging_config["eval_steps"]),
        eval_strategy=logging_config["eval_strategy"],
        save_strategy=logging_config["save_strategy"],
        load_best_model_at_end=logging_config["load_best_model_at_end"],
        metric_for_best_model=logging_config["metric_for_best_model"],
        logging_dir=log_dir,
        max_steps=max_steps if max_steps > 0 else -1,
        gradient_checkpointing=training.get("gradient_checkpointing", False),
        optim=dpo_config["optimizer"]["name"],
        report_to="none",
        save_total_limit=2,
        seed=run_config["seed"],
        # DPO 特有参数
        beta=dpo_params["beta"],                    # β=5（Tülu 3）
        loss_type=[dpo_params["loss_type"]],         # TRL 0.29 要求 list[str]
        label_smoothing=dpo_params["label_smoothing"],
        max_length=training["max_length"],
        # MPS 兼容
        dataloader_pin_memory=False,
    )

    # 训练日志
    training_logger = TrainingLogger()

    # 创建 DPOTrainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[DPOLoggerCallback(training_logger)],
    )

    # 开始训练
    print("\n" + "=" * 60)
    print("开始 DPO 训练")
    print(f"  SFT 模型: {sft_model_path}")
    print(f"  数据: {len(train_dataset):,} 对")
    print(f"  β (KL 惩罚): {dpo_params['beta']}")
    print(f"  Epochs: {training['num_train_epochs']}")
    print(f"  有效 batch size: {training['per_device_train_batch_size'] * training['gradient_accumulation_steps']}")
    print(f"  学习率: {training['learning_rate']}")
    print(f"  Max steps: {max_steps}")
    print("=" * 60)

    train_result = trainer.train()

    # 保存模型
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n模型已保存到: {output_dir}")

    # 保存训练日志
    log_path = Path(output_dir) / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(training_logger.get_summary(), f)
    print(f"训练日志已保存到: {log_path}")

    return model, tokenizer, training_logger


if __name__ == "__main__":
    run_dpo_training()
