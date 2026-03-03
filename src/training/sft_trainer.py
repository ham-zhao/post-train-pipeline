"""
SFT 训练器
使用 HuggingFace Transformers Trainer 进行监督微调

对应 Tülu 3 论文:
- Section 3: Supervised Finetuning
- lr=5e-6, epochs=2, linear scheduler
"""

import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config_loader import get_run_config, get_sft_config, PROJECT_ROOT
from src.training.training_utils import load_model_and_tokenizer, TrainingLogger


class LossLoggerCallback(TrainerCallback):
    """回调：记录每步训练 loss"""

    def __init__(self, logger: TrainingLogger):
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.logger.log_train_loss(state.global_step, logs["loss"])
        if logs and "eval_loss" in logs:
            self.logger.log_eval_loss(state.global_step, logs["eval_loss"])


def prepare_sft_dataset(data_path: str, tokenizer, max_seq_length: int = 2048,
                         eval_split: float = 0.05):
    """
    准备 SFT 训练数据集
    将 messages 格式转换为 tokenized 格式
    """
    print(f"加载 SFT 数据: {data_path}")

    # 读取 JSONL
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    print(f"  总样本数: {len(samples):,}")

    # tokenize
    def tokenize_fn(example):
        messages = example["messages"]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            # 回退拼接
            parts = []
            for msg in messages:
                parts.append(f"{msg['role']}: {msg['content']}")
            text = "\n".join(parts)

        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded

    dataset = Dataset.from_list(samples)
    tokenized = dataset.map(
        tokenize_fn,
        remove_columns=dataset.column_names,
        desc="Tokenizing SFT data",
        num_proc=1,
    )

    # 分割训练/验证集
    split = tokenized.train_test_split(test_size=eval_split, seed=42)
    print(f"  训练集: {len(split['train']):,}")
    print(f"  验证集: {len(split['test']):,}")

    return split["train"], split["test"]


def run_sft_training(
    model_name: str = None,
    data_path: str = None,
    output_dir: str = None,
    max_steps: int = -1,
):
    """
    执行 SFT 训练

    Args:
        model_name: 模型名称（默认从配置读取）
        data_path: 训练数据路径（默认从配置读取）
        output_dir: 输出目录（默认从配置读取）
        max_steps: 最大训练步数（-1 表示跑完所有数据）

    Returns:
        (model, tokenizer, training_logger)
    """
    # 加载配置
    run_config = get_run_config()
    sft_config = get_sft_config()

    model_name = model_name or run_config["model"]["name"]
    data_path = data_path or str(PROJECT_ROOT / sft_config["data"]["train_file"])
    output_dir = output_dir or str(PROJECT_ROOT / sft_config["output"]["dir"])
    log_dir = str(PROJECT_ROOT / sft_config["output"]["logging_dir"])

    if max_steps == -1:
        max_steps = run_config.get("max_steps", -1)

    # 加载模型
    model, tokenizer = load_model_and_tokenizer(
        model_name,
        dtype=run_config["model"]["dtype"],
        device=run_config["model"]["device"],
    )

    # 准备数据
    train_dataset, eval_dataset = prepare_sft_dataset(
        data_path, tokenizer,
        max_seq_length=sft_config["training"]["max_seq_length"],
        eval_split=sft_config["data"]["eval_split"],
    )

    # 训练参数
    training = sft_config["training"]
    logging_config = sft_config["logging"]

    # MPS 不支持 bf16 training，使用 fp32
    # 注意：模型加载用 bf16 节省内存，但训练计算用 fp32 确保兼容
    training_args = TrainingArguments(
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
        optim=sft_config["optimizer"]["name"],
        report_to="none",  # 不上报到 wandb 等
        save_total_limit=2,
        seed=run_config["seed"],
        # MPS 兼容设置
        dataloader_pin_memory=False,  # MPS 不支持 pin_memory
    )

    # 训练日志记录器
    training_logger = TrainingLogger()

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        max_length=training["max_seq_length"],
    )

    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[LossLoggerCallback(training_logger)],
    )

    # 开始训练
    print("\n" + "=" * 60)
    print("开始 SFT 训练")
    print(f"  模型: {model_name}")
    print(f"  数据: {len(train_dataset):,} 条")
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
    import json
    log_path = Path(output_dir) / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(training_logger.get_summary(), f)
    print(f"训练日志已保存到: {log_path}")

    return model, tokenizer, training_logger


if __name__ == "__main__":
    run_sft_training()
