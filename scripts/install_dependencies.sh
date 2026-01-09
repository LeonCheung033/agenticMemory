#!/bin/bash
# 手动安装依赖脚本
# 使用方法: bash scripts/install_dependencies.sh

set -e

echo "=========================================="
echo "安装 Agentic Memory 依赖"
echo "=========================================="

# 检查是否在conda环境中
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "警告: 未检测到conda环境，请先运行: conda activate agemem"
    read -p "是否继续? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "步骤 1/5: 安装PyTorch..."
# 检查是否有CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "检测到NVIDIA GPU，安装CUDA版本的PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "未检测到GPU，安装CPU版本的PyTorch..."
    pip install torch torchvision torchaudio
fi

echo ""
echo "步骤 2/5: 安装核心依赖..."
pip install transformers>=4.35.0 accelerate>=0.24.0 datasets>=2.14.0

echo ""
echo "步骤 3/5: 安装向量存储..."
pip install faiss-cpu chromadb>=0.4.0

echo ""
echo "步骤 4/5: 安装其他依赖..."
pip install \
    tiktoken>=0.5.0 \
    openai>=1.0.0 \
    wandb>=0.15.0 \
    tensorboard>=2.14.0 \
    pyyaml>=6.0 \
    numpy>=1.24.0 \
    scipy>=1.11.0 \
    scikit-learn>=1.3.0 \
    sentence-transformers>=2.2.0 \
    rouge-score>=0.1.2 \
    nltk>=3.8.0

echo ""
echo "步骤 5/5: 安装开发工具..."
pip install \
    pytest>=7.4.0 \
    pytest-cov>=4.1.0 \
    black>=23.0.0 \
    flake8>=6.0.0 \
    mypy>=1.5.0 \
    pre-commit>=3.5.0

echo ""
echo "下载NLTK数据..."
python -c "import nltk; nltk.download('punkt')" || echo "NLTK数据下载失败，可以稍后手动下载"

echo ""
echo "=========================================="
echo "依赖安装完成！"
echo "=========================================="
echo ""
echo "下一步:"
echo "1. pip install -e '.[dev]'  # 安装项目包"
echo "2. pre-commit install      # 安装pre-commit hooks"
