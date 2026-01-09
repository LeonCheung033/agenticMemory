#!/bin/bash
# 环境设置脚本
# 使用方法: bash scripts/setup_env.sh

set -e  # 遇到错误立即退出

echo "=========================================="
echo "Agentic Memory 环境设置"
echo "=========================================="

# 检查conda是否安装
if ! command -v conda &> /dev/null; then
    echo "错误: 未找到conda命令，请先安装Anaconda或Miniconda"
    exit 1
fi

echo "步骤 1/4: 创建conda环境..."
conda env create -f environment.yml

echo ""
echo "步骤 2/4: 激活环境..."
echo "请运行以下命令激活环境:"
echo "  conda activate agemem"

echo ""
echo "步骤 3/4: 安装开发依赖..."
echo "激活环境后，运行:"
echo "  pip install -e '.[dev]'"

echo ""
echo "步骤 4/4: 下载NLTK数据..."
echo "激活环境后，运行:"
echo "  python -c \"import nltk; nltk.download('punkt')\""

echo ""
echo "=========================================="
echo "环境设置完成！"
echo "=========================================="
echo ""
echo "下一步:"
echo "1. conda activate agemem"
echo "2. pip install -e '.[dev]'"
echo "3. python -c \"import nltk; nltk.download('punkt')\""
echo "4. pre-commit install"
