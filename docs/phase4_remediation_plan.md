# Phase 4：合规修复与全量验收计划

## 问题

对照 `~/.claude/CLAUDE.md` 审计现有代码，发现 **19 处违规**，其中：
- Notebook 00-05 从未执行过（代码存在 ≠ 任务完成）
- 脚本缺少产出文件声明
- Notebook 缺少依赖文件校验
- README 结果表仍为占位符

## 方案

逐项修复所有违规，执行全部 Notebook，完成阶段验收。

## 预期

- 9 个 Notebook 全部有 output，无 traceback
- 7 个脚本全部有产出文件声明
- README 结果表填入实际数值
- 阶段验收清单 100% 通过

---

## 违规清单

| # | 规则 | 违规内容 | 严重度 |
|---|------|---------|--------|
| 1 | §1.1 增量执行 | Notebook 00-05 从未执行 | **阻断** |
| 2 | §2.2 数据契约 | 5 个脚本缺少产出文件声明 | 高 |
| 3 | §2.2 数据契约 | 4 个 Notebook 缺少依赖文件校验 | 高 |
| 4 | §3.1 Overview | README 结果表全为占位符 `-` | 高 |
| 5 | §2.3 Config 驱动 | run_8b_lora.py 硬编码超参 | 中 |

---

## 执行计划

### Step 1：脚本产出文件声明（§2.2）

为以下 5 个脚本添加产出文件声明：

| 脚本 | 产出文件 |
|------|---------|
| `scripts/run_download.py` | `data/sft_mix/train.jsonl`, `data/dpo_mix/train.jsonl`, `results/reports/data_stats.json` |
| `scripts/run_sft.py` | `results/checkpoints/sft/model.safetensors`, `results/checkpoints/sft/training_log.json` |
| `scripts/run_dpo.py` | `results/checkpoints/dpo/model.safetensors`, `results/checkpoints/dpo/training_log.json` |
| `scripts/run_eval.py` | `results/eval_results/{base,sft,dpo}/scores.json`, `results/eval_results/{base,sft,dpo}/safety.json`, `results/eval_results/generation_comparison.json`, `results/eval_results/summary.json` |
| `scripts/run_8b_lora.py` | `results/checkpoints/lora_8b/adapter_model.safetensors` |

**验收**：`grep -l "产出文件" scripts/*.py | wc -l` → 应为 5

### Step 2：Notebook 依赖文件校验（§2.2）

为以下 4 个 Notebook 在首个代码 cell 后添加 REQUIRED 校验：

| Notebook | 依赖文件 |
|----------|---------|
| `02_sft_data_mixing.ipynb` | `data/sft_mix/train.jsonl` |
| `03_sft_training.ipynb` | `results/checkpoints/sft/training_log.json` |
| `04_dpo_data_analysis.ipynb` | `data/dpo_mix/train.jsonl` |
| `05_dpo_training.ipynb` | `results/checkpoints/dpo/training_log.json` |

校验模板：
```python
REQUIRED_FILES = ["path/to/file"]
for f in REQUIRED_FILES:
    assert (PROJECT_ROOT / f).exists(), f"缺少: {f}，请先运行对应脚本"
```

**验收**：`grep -c "REQUIRED_FILES" notebooks/0{2,3,4,5}*.ipynb` → 每个文件 ≥ 1

### Step 3：执行 Notebook 00-05（§1.1）

按依赖顺序逐个执行，每个执行后验收：

| 顺序 | Notebook | 依赖 | 验收标准 |
|------|----------|------|---------|
| 1 | 00_post_training_overview | 无 | 所有 cell 有 output |
| 2 | 01_data_exploration | `data/sft_mix/train.jsonl` | 数据统计表 + 分布图 |
| 3 | 02_sft_data_mixing | `data/sft_mix/train.jsonl` | 配比饼图有输出 |
| 4 | 03_sft_training | `results/checkpoints/sft/training_log.json` | loss 曲线图有输出 |
| 5 | 04_dpo_data_analysis | `data/dpo_mix/train.jsonl` | 偏好数据统计有输出 |
| 6 | 05_dpo_training | `results/checkpoints/dpo/training_log.json` | reward margin 图有输出 |

执行命令（每个单独执行，失败则修复后重试）：
```bash
source .venv/bin/activate
jupyter nbconvert --to notebook --execute --inplace notebooks/00_post_training_overview.ipynb
# 验收：无 traceback
jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_exploration.ipynb
# 验收：无 traceback
# ... 逐个执行
```

**阻断规则**：Notebook N 执行失败 → 修复后重试 → 通过后才执行 N+1

### Step 4：更新 README 结果表（§3.1）

将占位符 `-` 替换为实际实验结果：

```markdown
| 指标 | Base | SFT | DPO | LIFT |
|------|------|-----|-----|------|
| HellaSwag (acc_norm) | 63.0% | 63.0% | 63.0% | +0.0% |
| Safety ASR↓ | 85.0% | 40.0% | 40.0% | -45.0% |
| Over-refusal↓ | 0.0% | 0.0% | 10.0% | +10.0% |
```

消融实验表：
```markdown
| 去掉什么 | 样本减少 | 验证结论 |
|---------|---------|---------|
| 安全数据 | -600 (30%) | 安全正交性：预测 ASR 显著上升 |
| 数学数据 | -240 (12%) | 预测数学 benchmark 下降 |
| CoCoNot | -150 (7.5%) | 预测 Over-refusal 上升 |
```

**验收**：`grep -c "^|.*-.*|$" README.md` → 应为 0（无占位符行）

### Step 5：修复 run_8b_lora.py 硬编码（§2.3）

将硬编码超参移至注释说明（该脚本为独立参考脚本，不经 config 驱动属可接受降级方案，但需注释说明）。

### Step 6：阶段验收

```
Phase 4 验收：
  [ ] 5 个脚本均有产出文件声明
  [ ] 4 个 Notebook 均有 REQUIRED_FILES 校验
  [ ] 9 个 Notebook 全部执行成功（有 output，无 traceback）
  [ ] README 结果表无占位符
  [ ] git commit + push 到 GitHub
```

---

## 依赖图

```
Step 1 (脚本声明) ──┐
                     ├──→ Step 3 (执行 NB 00-05) ──→ Step 6 (验收)
Step 2 (NB 校验)  ──┘                                    ↑
                                                          │
Step 4 (README) ──────────────────────────────────────────┘
Step 5 (8b_lora) ─────────────────────────────────────────┘
```

Step 1 和 Step 2 可并行。Step 3 依赖 Step 2 完成。Step 4、5 独立可并行。
