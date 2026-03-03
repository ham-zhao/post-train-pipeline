"""
数据格式转换器
将各种格式的数据统一为 SFT/DPO 训练格式

SFT 统一格式:
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "source": "..."}

DPO 统一格式:
{"prompt": "...", "chosen": "...", "rejected": "...", "source": "..."}
"""

import json
from pathlib import Path
from typing import Optional
from tqdm import tqdm


def convert_to_sft_format(sample: dict, source: str = "unknown") -> Optional[dict]:
    """
    将单条样本转换为统一 SFT 格式
    处理多种输入格式：messages、instruction/output、prompt/completion 等
    """
    messages = []

    # 格式 1: 已有 messages 字段
    if "messages" in sample and isinstance(sample["messages"], list):
        for msg in sample["messages"]:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                role = msg["role"]
                content = msg["content"]
                if role in ("user", "human"):
                    messages.append({"role": "user", "content": str(content)})
                elif role in ("assistant", "gpt", "bot"):
                    messages.append({"role": "assistant", "content": str(content)})
                elif role == "system":
                    messages.append({"role": "system", "content": str(content)})

    # 格式 2: instruction + output (FLAN, Alpaca 等)
    elif "instruction" in sample or "input" in sample:
        instruction = sample.get("instruction", sample.get("input", ""))
        output = sample.get("output", sample.get("response", ""))
        # 如果有额外 input 字段，拼接到 instruction
        if "instruction" in sample and "input" in sample and sample["input"]:
            instruction = f"{instruction}\n\n{sample['input']}"
        if instruction and output:
            messages = [
                {"role": "user", "content": str(instruction)},
                {"role": "assistant", "content": str(output)},
            ]

    # 格式 3: prompt + completion
    elif "prompt" in sample and "completion" in sample:
        messages = [
            {"role": "user", "content": str(sample["prompt"])},
            {"role": "assistant", "content": str(sample["completion"])},
        ]

    # 格式 4: question + answer
    elif "question" in sample and "answer" in sample:
        messages = [
            {"role": "user", "content": str(sample["question"])},
            {"role": "assistant", "content": str(sample["answer"])},
        ]

    # 格式 5: text 字段（OpenAssistant-guanaco 等）
    elif "text" in sample:
        text = sample["text"]
        # 尝试解析 ### Human: ... ### Assistant: ... 格式
        if "### Human:" in text and "### Assistant:" in text:
            parts = text.split("### ")
            for part in parts:
                part = part.strip()
                if part.startswith("Human:"):
                    content = part[len("Human:"):].strip()
                    if content:
                        messages.append({"role": "user", "content": content})
                elif part.startswith("Assistant:"):
                    content = part[len("Assistant:"):].strip()
                    if content:
                        messages.append({"role": "assistant", "content": content})

    # 验证：至少包含一轮 user + assistant 对话
    if len(messages) < 2:
        return None

    has_user = any(m["role"] == "user" for m in messages)
    has_assistant = any(m["role"] == "assistant" for m in messages)
    if not (has_user and has_assistant):
        return None

    return {
        "messages": messages,
        "source": source,
    }


def convert_to_dpo_format(sample: dict, source: str = "tulu-3-preference") -> Optional[dict]:
    """
    将单条样本转换为统一 DPO 格式
    需要 prompt + chosen + rejected
    """
    prompt = None
    chosen = None
    rejected = None

    # 格式 1: 直接有 prompt/chosen/rejected
    if all(k in sample for k in ["prompt", "chosen", "rejected"]):
        prompt = sample["prompt"]
        chosen = sample["chosen"]
        rejected = sample["rejected"]

    # 格式 2: chosen_messages / rejected_messages（Tülu 3 偏好数据常用格式）
    elif "chosen" in sample and "rejected" in sample:
        chosen_data = sample["chosen"]
        rejected_data = sample["rejected"]

        # chosen/rejected 可能是 messages 列表
        if isinstance(chosen_data, list) and isinstance(rejected_data, list):
            # 提取 prompt（通常是 user 消息）
            prompt_parts = []
            chosen_response = ""
            rejected_response = ""

            for msg in chosen_data:
                if isinstance(msg, dict):
                    if msg.get("role") in ("user", "human"):
                        prompt_parts.append(msg.get("content", ""))
                    elif msg.get("role") in ("assistant", "gpt"):
                        chosen_response = msg.get("content", "")

            for msg in rejected_data:
                if isinstance(msg, dict):
                    if msg.get("role") in ("assistant", "gpt"):
                        rejected_response = msg.get("content", "")

            if prompt_parts and chosen_response and rejected_response:
                prompt = "\n".join(prompt_parts)
                chosen = chosen_response
                rejected = rejected_response
        else:
            # chosen/rejected 直接是字符串
            if "prompt" in sample:
                prompt = sample["prompt"]
            elif "question" in sample:
                prompt = sample["question"]
            chosen = str(chosen_data)
            rejected = str(rejected_data)

    # 格式 3: question + chosen_response + rejected_response
    elif "question" in sample:
        prompt = sample.get("question", "")
        chosen = sample.get("chosen_response", sample.get("chosen", ""))
        rejected = sample.get("rejected_response", sample.get("rejected", ""))

    # 验证
    if not prompt or not chosen or not rejected:
        return None

    # 确保都是字符串
    prompt = str(prompt).strip()
    chosen = str(chosen).strip()
    rejected = str(rejected).strip()

    if not prompt or not chosen or not rejected:
        return None

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "source": source,
    }


def batch_convert_sft(samples: list, source: str = "unknown") -> list:
    """批量转换 SFT 数据"""
    converted = []
    failed = 0

    for sample in tqdm(samples, desc=f"转换 SFT ({source})"):
        result = convert_to_sft_format(sample, source=source)
        if result:
            converted.append(result)
        else:
            failed += 1

    print(f"  {source}: 成功 {len(converted):,} / 失败 {failed:,}")
    return converted


def batch_convert_dpo(samples: list, source: str = "tulu-3-preference") -> list:
    """批量转换 DPO 数据"""
    converted = []
    failed = 0

    for sample in tqdm(samples, desc="转换 DPO"):
        result = convert_to_dpo_format(sample, source=source)
        if result:
            converted.append(result)
        else:
            failed += 1

    print(f"  DPO: 成功 {len(converted):,} / 失败 {failed:,}")
    return converted
