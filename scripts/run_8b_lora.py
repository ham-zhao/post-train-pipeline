#!/usr/bin/env python3
"""
可选：Llama-3.1-8B QLoRA 训练脚本
用法: python scripts/run_8b_lora.py

产出文件：
  results/checkpoints/lora_8b/adapter_model.safetensors  - LoRA adapter 权重
  results/checkpoints/lora_8b/adapter_config.json        - LoRA 配置
  results/checkpoints/lora_8b/training_log.json          - 训练日志

8B 模型在 128GB M4 Max 上无法全量微调，需要使用 QLoRA（4-bit 量化 + LoRA）。
内存估算：
  - 4-bit 量化模型: ~4.5GB
  - LoRA adapter: ~0.1GB
  - 优化器状态: ~0.3GB
  - 激活值: ~2-4GB
  - 总计: ~8-12GB（128GB 内存完全够用）

注意：
  - MPS 对 4-bit 量化支持有限，可能需要 CPU 回退
  - 本脚本为独立参考脚本，超参直接内联而非走 config（降级方案）
"""

import sys
import json
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import get_run_config, get_sft_config, PROJECT_ROOT as ROOT


def check_dependencies():
    """检查 QLoRA 所需的额外依赖"""
    missing = []
    try:
        import bitsandbytes
    except ImportError:
        missing.append("bitsandbytes")
    try:
        from peft import LoraConfig
    except ImportError:
        missing.append("peft")

    if missing:
        print(f"缺少依赖: {', '.join(missing)}")
        print(f"安装命令: pip install {' '.join(missing)}")
        return False
    return True


def run_8b_lora_sft():
    """
    使用 QLoRA 对 Llama-3.1-8B 进行 SFT

    QLoRA 配置说明：
    - r=16: LoRA 秩。越大表达能力越强，但显存越多
    - lora_alpha=32: LoRA 缩放系数。通常设为 2×r
    - target_modules: 对哪些层加 LoRA。通常选注意力层
    - 4-bit NormalFloat (NF4) 量化: 比 INT4 精度更高
    """
    if not check_dependencies():
        return

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForSeq2Seq,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import Dataset

    # 配置
    model_name = "meta-llama/Llama-3.1-8B"
    output_dir = str(ROOT / "results/checkpoints/8b_lora_sft")
    data_path = str(ROOT / "data/sft_mix/train.jsonl")

    run_config = get_run_config()
    sft_config = get_sft_config()
    max_steps = run_config.get("max_steps", 100)

    print("=" * 60)
    print("Llama-3.1-8B QLoRA SFT")
    print("=" * 60)

    # 4-bit 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",           # NormalFloat4，比 INT4 精度高
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,       # 二次量化，进一步省内存
    )

    # 加载量化模型
    print(f"\n加载 4-bit 量化模型: {model_name}")
    print("  (首次下载约 5GB，需要 HF 账号和 Llama 3.1 许可)")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # 准备 LoRA
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,                                 # LoRA 秩
        lora_alpha=32,                        # LoRA alpha = 2×r
        target_modules=[                      # 只对注意力层加 LoRA
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    # 打印可训练参数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  总参数量: {total_params / 1e9:.2f}B")
    print(f"  可训练参数量: {trainable_params / 1e6:.1f}M ({trainable_params / total_params * 100:.2f}%)")

    # 加载训练数据
    print(f"\n加载 SFT 数据: {data_path}")
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    def tokenize_fn(example):
        messages = example["messages"]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            parts = [f"{msg['role']}: {msg['content']}" for msg in messages]
            text = "\n".join(parts)

        encoded = tokenizer(text, truncation=True, max_length=1024, padding=False)
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded

    dataset = Dataset.from_list(samples)
    tokenized = dataset.map(tokenize_fn, remove_columns=dataset.column_names, desc="Tokenizing")
    split = tokenized.train_test_split(test_size=0.05, seed=42)

    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,        # 有效 batch=32
        learning_rate=2e-4,                    # QLoRA 常用 lr
        warmup_ratio=0.03,
        weight_decay=0.0,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_steps=100,
        save_steps=100,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        max_steps=max_steps,
        report_to="none",
        save_total_limit=2,
        seed=42,
        dataloader_pin_memory=False,
        optim="paged_adamw_8bit",              # 8-bit AdamW，省内存
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, padding=True, max_length=1024,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=data_collator,
    )

    # 开始训练
    print(f"\n  训练数据: {len(split['train']):,} 条")
    print(f"  验证数据: {len(split['test']):,} 条")
    print(f"  Max steps: {max_steps}")
    print(f"  有效 batch size: 32")
    print(f"  学习率: 2e-4")
    print()

    trainer.train()

    # 保存 LoRA adapter
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nLoRA adapter 已保存到: {output_dir}")
    print("使用 peft.AutoPeftModelForCausalLM.from_pretrained() 加载")


if __name__ == "__main__":
    print("注意: 此脚本需要:")
    print("  1. pip install bitsandbytes peft")
    print("  2. Llama 3.1 模型访问权限 (huggingface.co/meta-llama)")
    print("  3. 足够的磁盘空间 (~10GB)")
    print()

    run_8b_lora_sft()
