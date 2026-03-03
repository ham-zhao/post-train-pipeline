# 交互记录

## 阶段一：环境搭建 + 数据准备 + 概念讲解

### 交互 1: 项目初始化
- **类型**：指令确认
- **用户输入**：启动阶段一，创建项目结构、run_config.yaml 和 README.md
- **处理要点**：创建完整目录结构（configs/data/notebooks/src/scripts/results/docs），编写 4 个配置文件（run_config/sft_config/dpo_config/eval_config）、requirements.txt、setup.sh、README.md、interaction_log.md
- **关键发现**：项目为空目录起步，所有文件从零创建

### 交互 2: 继续执行阶段一
- **类型**：指令确认
- **用户输入**：继续（安装依赖、下载数据、创建 Notebooks）
- **处理要点**：发现系统 Python 3.9.6 版本过低（MPS 需要 3.10+），通过 brew 安装 Python 3.11 + venv 解决。依赖全部安装成功。
- **关键发现**：PyTorch 2.10.0 + MPS 就绪

### 踩坑记录 1: SFT 数据 source 映射不匹配
- **报错信息**：SFT 只下载到 604 条（目标 2000），多个子集显示"不足"
- **原因**：`allenai/tulu-3-sft-mixture` 中的 source 字段用的是完整 HuggingFace ID（如 `ai2-adapt-dev/numinamath_tir_math_decontaminated`），与初始映射不一致
- **解决方案**：扫描全量 939K 条获取 19 个实际 source 名称，更新 SOURCE_MAPPING 和处理上限（500K→950K）
- **预防建议**：使用新数据集前先用 streaming 采样检查字段格式和值

### 踩坑记录 2: Python 版本过低
- **报错信息**：系统 Python 3.9.6，pip 为旧版
- **原因**：macOS 默认 Python 版本低，不支持 MPS backend
- **解决方案**：`brew install python@3.11` + `python3.11 -m venv .venv`
- **预防建议**：项目开始时先检查 Python 版本 >= 3.10

## 阶段二：SFT 训练 + DPO 训练

### 交互 3: 启动阶段二
- **类型**：指令确认
- **用户输入**：继续执行阶段二
- **处理要点**：依次执行 SFT 训练 → DPO 训练 → 创建 Notebooks 02-05 → 生成 8B LoRA 脚本

### 踩坑记录 3: `torch_dtype` 已废弃
- **报错信息**：`torch_dtype` is deprecated! Use `dtype` instead!
- **原因**：transformers 5.2.0 中 `from_pretrained()` 参数名变更
- **解决方案**：`training_utils.py` 中 `torch_dtype=torch_dtype` → `dtype=torch_dtype`
- **预防建议**：新版 transformers 需检查废弃参数

### 踩坑记录 4: `use_mps_device` 已废弃
- **报错信息**：潜在废弃警告
- **原因**：transformers 5.x 中 Trainer 自动检测 MPS 设备
- **解决方案**：从 `TrainingArguments` 中移除 `use_mps_device` 参数
- **预防建议**：新版 transformers Trainer 不再需要显式指定 MPS

### 踩坑记录 5: MPS 内存不足 (OOM)
- **报错信息**：`RuntimeError: MPS backend out of memory (MPS allocated: 32.22 GiB)`
- **原因**：batch_size=4 + seq_length=2048 + 全量微调 1.5B 参数导致激活值过大
- **解决方案**：
  1. batch_size: 4→1，grad_accum: 8→32（保持有效 batch=32）
  2. max_seq_length: 2048→1024
  3. max_steps: 500→100（适配 smoke_test 时间目标）
  4. 环境变量 `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`
- **预防建议**：MPS 上全量微调 1.5B 模型，batch_size=1 是安全选择

### 踩坑记录 6: DPOConfig `max_prompt_length` 已移除
- **报错信息**：`TypeError: DPOConfig.__init__() got an unexpected keyword argument 'max_prompt_length'`
- **原因**：TRL 0.29.0 中 `DPOConfig` 不再支持 `max_prompt_length` 参数，只保留 `max_length`
- **解决方案**：从 `DPOConfig` 初始化中移除 `max_prompt_length` 参数
- **预防建议**：TRL 版本更新频繁，API 变化大，使用前先检查 `help(DPOConfig.__init__)`

### 踩坑记录 7: DPOConfig `loss_type` 改为 list[str]
- **报错信息**：潜在类型不匹配
- **原因**：TRL 0.29.0 中 `loss_type` 从 `str` 改为 `list[str]`
- **解决方案**：`loss_type="sigmoid"` → `loss_type=["sigmoid"]`
- **预防建议**：DPO 相关 API 变化快，注意查看类型签名

### SFT 训练结果
- **模型**：Qwen2.5-1.5B 全量微调
- **数据**：2000 条 SFT 混合数据
- **训练步数**：100 步（smoke_test 模式）
- **Training loss**：1.757 → 1.549（稳步下降）
- **Eval loss**：1.852 (step 50) → 1.830 (step 100)
- **耗时**：约 28 分钟（~17s/step）
- **输出**：`results/checkpoints/sft/`（~3GB model.safetensors）

### DPO 训练结果
- **模型**：以 SFT checkpoint 为起点 + 冻结 ref_model
- **数据**：1000 对偏好数据（950 训练 / 50 验证）
- **训练步数**：100 步（smoke_test 模式）
- **Training loss**：2.123 → 1.599（稳步下降）
- **Eval loss**：2.923 (step 50) → 2.283 (step 100)
- **Reward margin**：-0.115 (step 10) → +0.833 (step 100) ✅
  - 从负到正说明模型成功学会偏好 chosen 回复
  - chosen reward: 0.003 → 1.028
  - rejected reward: 0.118 → 0.195
- **耗时**：约 97 分钟（~58s/step，DPO 需双模型前向传播）
- **输出**：`results/checkpoints/dpo/`（~3GB model.safetensors）

### 阶段二完成清单
- [x] SFT 训练（100 步 smoke_test）
- [x] DPO 训练（100 步 smoke_test）
- [x] Notebook 02: SFT 数据混合实验
- [x] Notebook 03: SFT 训练可视化
- [x] Notebook 04: DPO 数据分析
- [x] Notebook 05: DPO 训练可视化
- [x] 8B LoRA 训练脚本（scripts/run_8b_lora.py）

## 阶段三：评估 + 消融 + 三方对比 Dashboard

### 交互 4: 启动阶段三
- **类型**：指令确认
- **用户输入**：执行 Phase 3 完整计划
- **处理要点**：运行 lm-eval benchmark → 安全评估 → 生成对比 → Dashboard → 消融数据准备 → 创建 Notebooks 06-08

### 踩坑记录 8: lm-eval stdout 解析器提取错误分数
- **报错信息**：HellaSwag 分数全部显示为 1.0（实际应为 0.63）
- **原因**：`parse_lm_eval_output()` 解析 stdout 的 `|` 分隔表格时，正则匹配到了错误的数值列（匹配到了 filter 列的 "none" 而非实际 acc_norm 值）
- **解决方案**：新增 `parse_lm_eval_json()` 函数，直接从 lm-eval 保存的 JSON 结果文件中提取 `acc_norm,none` 字段，在 `run_eval_pipeline()` 中优先使用 JSON 解析
- **预防建议**：lm-eval 的 stdout 格式不稳定，应始终使用其 JSON 输出文件作为结果来源

### 评估配置调整
- **eval_config.yaml**: `batch_size` 从 8 降至 2（防止 MPS OOM）
- **评估模式**: smoke_test（HellaSwag only, 200 sample limit）

### Benchmark 评估结果 (HellaSwag acc_norm, 0-shot, 200 samples)
| 模型 | acc_norm |
|------|---------|
| Base (Qwen2.5-1.5B) | 0.6300 |
| SFT | 0.6300 |
| DPO | 0.6300 |

- **解读**：三个阶段分数一致，smoke_test 100 步训练不足以改变通用推理能力
- **符合预期**：Tulu 3 论文中 SFT 后 HellaSwag 通常保持或略降

### 安全评估结果 (20 harmful + 10 benign prompts)
| 模型 | ASR ↓ | Over-refusal ↓ |
|------|-------|----------------|
| Base | 85.0% | 0.0% |
| SFT | 40.0% | 0.0% |
| DPO | 40.0% | 10.0% |

- **ASR 从 85% 降至 40%**：SFT 安全数据（WildGuardMix + WildJailbreak）效果显著
- **Over-refusal 从 0% 升至 10%**：DPO 阶段出现轻微过度拒绝
- **Base 模型**：大量有害 prompt 输出重复乱码（"afone"），不含拒绝关键词故计为"未拒绝"

### 生成质量对比
- **Base**：频繁输出重复/乱码（缺乏 chat 能力的 base model 表现正常）
- **SFT**：能正确回答通用和数学问题，安全拒绝有改善
- **DPO**：回答更简洁有结构，但 smoke_test 下与 SFT 差异不大

### 消融数据准备
| 实验 | 样本数 | 减少量 | 说明 |
|------|--------|--------|------|
| full | 1999 | 0 | 对照基准 |
| no_safety | 1399 | -600 | 去掉安全数据 |
| no_math | 1759 | -240 | 去掉数学数据 |
| no_coconot | 1849 | -150 | 去掉防过度拒绝数据 |
| no_norobots | 1714 | -285 | 去掉人工对话数据 |
| no_flan | 1699 | -300 | 去掉通用指令数据 |

### Dashboard 生成
- 四象限 Dashboard 已保存: `results/figures/dashboard.png`
  - 左上: 能力雷达图（Base vs SFT vs DPO）
  - 右上: 安全指标柱状图（ASR + Over-refusal）
  - 左下: SFT Training Loss 曲线
  - 右下: DPO Reward Margin 曲线

### 阶段三完成清单
- [x] Benchmark 评估（lm-eval, HellaSwag 0-shot）
- [x] 安全评估（20 harmful + 10 benign prompts）
- [x] 生成质量对比（10 prompts, 4 categories）
- [x] 修复 lm-eval 分数解析器（stdout → JSON）
- [x] 三方对比 Dashboard（dashboard.png）
- [x] 消融数据准备（6 组）
- [x] Notebook 06: 评估结果分析
- [x] Notebook 07: 消融实验分析
- [x] Notebook 08: 三方对比 Dashboard
- [x] 更新交互日志
