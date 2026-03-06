# Post-Training Pipeline：复现 Tülu 3 的 SFT → DPO 全流程

> 将一个"只会接龙"的 Base Model 变成"会对话、能安全拒绝"的 Chat Model

## Post-Training 全景图

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Post-Training Pipeline                          │
│                                                                     │
│  Base Model ──→ SFT ──→ DPO ──→ (RLVR) ──→ Chat Model            │
│  (只会接龙)    (学对话)  (学偏好)  (可验证RL)   (能对话+安全)        │
│                                                                     │
│  Qwen2.5-1.5B   57K条    30K对    [不实现]     最终模型              │
│                 指令数据  偏好数据  只讲原理                         │
└─────────────────────────────────────────────────────────────────────┘
```

## 项目目标

复现 AllenAI **Tülu 3** 论文的完整 Post-training 流程，理解：

1. **SFT（监督微调）**：用指令数据教模型"对话格式"
2. **DPO（直接偏好优化）**：用偏好对数据提升回答质量和安全性
3. **数据配比**：不同技能数据如何影响模型能力
4. **安全正交性**：安全数据和能力数据占不同参数空间
5. **消融实验**：每个组件的贡献量化

## 硬件与模型

| 配置 | 值 |
|------|-----|
| 机器 | MacBook Pro M4 Max |
| 内存 | 128GB 统一内存 |
| 加速 | PyTorch MPS backend |
| 主模型 | Qwen2.5-1.5B（全量微调） |
| 可选 | Llama-3.1-8B（QLoRA，隔夜运行） |

## 数据配比（对应 Tülu 3 论文 Table 2）

```
SFT 数据 57K 条:
┌──────────────┬───────┬─────────────────────┐
│ FLAN v2      │ 10K   │ ████████ 通用指令    │
│ WildGuardMix │ 10K   │ ████████ 安全拒绝    │
│ WildJailbreak│ 10K   │ ████████ 对抗安全    │
│ No Robots    │ 9.5K  │ ███████  人工对话    │
│ OpenAssistant│ 7.1K  │ ██████   多轮对话    │
│ NuminaMath   │ 5K    │ ████     数学推理    │
│ Persona IF   │ 5K    │ ████     指令遵循    │
│ CoCoNot      │ 5K    │ ████     防过度拒绝  │
│ Persona Math │ 3K    │ ██       合成数学    │
│ Persona Code │ 2K    │ ██       合成代码    │
└──────────────┴───────┴─────────────────────┘

DPO 数据 30K 对:
  allenai/llama-3.1-tulu-3-8b-preference-mixture
  每对含 chosen（好回复）+ rejected（差回复）
```

## 两档运行模式

```yaml
smoke_test:   # 20-30 分钟，验证代码
  SFT 2000条 / DPO 1000对 / 评估 hellaswag

full_run:     # 7-12 小时，完整实验
  SFT 57K条 / DPO 30K对 / 评估 5个benchmark
```

## 快速开始

```bash
# 1. 环境搭建
bash setup.sh

# 2. 下载数据
python scripts/run_download.py

# 3. SFT 训练
python scripts/run_sft.py

# 4. DPO 训练
python scripts/run_dpo.py

# 5. 评估对比
python scripts/run_eval.py
```

## 核心发现（smoke_test 模式，2000 SFT / 1000 DPO）

### 三方对比: Base vs SFT vs DPO

| 指标 | Base | SFT | DPO | 说明 |
|------|------|-----|-----|------|
| HellaSwag (acc_norm) | 0.63 | 0.63 | 0.63 | 通用能力保持不变 |
| Safety ASR↓ | 85.0% | 40.0% | 40.0% | SFT 大幅降低攻击成功率 |
| Over-refusal↓ | 0.0% | 0.0% | 10.0% | DPO 略增过度拒绝（正常） |

**关键结论**：
- SFT 将 ASR 从 85% 降至 40%（-45pp），验证了安全数据的有效性
- HellaSwag 在三阶段保持一致，说明 post-training 不损害通用语言能力
- DPO reward margin 从 -0.115 上升到 +0.833，模型成功学习了偏好方向

### SFT 训练指标

| 指标 | 起始 | 终止 | 变化 |
|------|------|------|------|
| Train Loss | 1.757 | 1.549 | -11.8% |
| Eval Loss | 1.852 | 1.830 | -1.2% |

### DPO 训练指标

| 指标 | 起始 | 终止 | 变化 |
|------|------|------|------|
| DPO Loss | 2.123 | 1.599 | -24.7% |
| Chosen Reward | 0.003 | 1.028 | +1.025 |
| Rejected Reward | 0.118 | 0.195 | +0.077 |
| Reward Margin | -0.115 | +0.833 | +0.948 |

### 消融实验（数据已准备，full_run 模式可执行）

| 去掉什么 | 预期影响 | 验证假设 |
|---------|---------|---------|
| 安全数据 (WildGuard+WildJail) | ASR 上升，能力不变 | 安全正交性 |
| 数学数据 (NuminaMath+PersonaMath) | 数学能力下降 | 数学贡献 |
| CoCoNot | Over-refusal 上升 | 防过度拒绝 |
| No Robots | 对话质量下降 | 人工数据价值 |
| FLAN v2 | 通用指令能力下降 | 通用数据基底 |

## 项目结构

```
post-train-pipeline/
├── configs/          # 训练、评估配置文件
├── data/             # 下载的数据集
├── notebooks/        # 8 个教学 Notebook
├── src/              # 源代码
│   ├── data_preparation/   # 数据下载、格式转换、去污染
│   ├── training/           # SFT、DPO 训练器
│   ├── evaluation/         # 评估、安全测试
│   └── utils/              # 配置加载、可视化
├── scripts/          # 一键运行脚本
├── results/          # 模型、评估结果、图表
└── docs/             # 交互日志、FAQ
```

## 参考

- [Tülu 3 论文](https://arxiv.org/abs/2411.15124)
- [Tülu 3 GitHub](https://github.com/allenai/open-instruct)
- [DPO 论文](https://arxiv.org/abs/2305.18290)
- [Qwen2.5 技术报告](https://arxiv.org/abs/2412.15115)
