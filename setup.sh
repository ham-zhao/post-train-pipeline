#!/bin/bash
# =============================================================================
# Post-Training Pipeline 环境搭建脚本
# 用法: bash setup.sh
# =============================================================================

set -e

echo "=========================================="
echo "Post-Training Pipeline 环境搭建"
echo "=========================================="

# --- 检查 Python 版本 ---
echo ""
echo "[1/4] 检查 Python 版本..."
python3 --version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python 版本: $PYTHON_VERSION"

# --- 检查 MPS 支持 ---
echo ""
echo "[2/4] 检查 Apple Silicon MPS 支持..."
python3 -c "
import torch
print(f'PyTorch 版本: {torch.__version__}')
print(f'MPS 可用: {torch.backends.mps.is_available()}')
if torch.backends.mps.is_available():
    print('MPS backend 就绪，可用于 GPU 加速')
else:
    print('警告: MPS 不可用，将回退到 CPU')
"

# --- 安装依赖 ---
echo ""
echo "[3/4] 安装 Python 依赖..."
pip install -r requirements.txt

# --- 创建必要目录 ---
echo ""
echo "[4/4] 创建项目目录结构..."
mkdir -p data/{sft_mix,dpo_mix,eval}
mkdir -p results/{figures,checkpoints/{sft,dpo},eval_results,reports,logs/{sft,dpo}}
mkdir -p docs

# --- 完成 ---
echo ""
echo "=========================================="
echo "环境搭建完成！"
echo "=========================================="
echo ""
echo "下一步:"
echo "  1. 下载数据: python scripts/run_download.py"
echo "  2. SFT 训练: python scripts/run_sft.py"
echo "  3. DPO 训练: python scripts/run_dpo.py"
echo "  4. 评估对比: python scripts/run_eval.py"
echo ""
echo "配置文件: configs/run_config.yaml"
echo "  当前模式: $(grep 'run_mode:' configs/run_config.yaml | awk '{print $2}' | tr -d '\"')"
