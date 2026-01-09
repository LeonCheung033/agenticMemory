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

echo "步骤 1/5: 创建conda环境（仅Python和pip）..."
conda env create -f environment.yml

echo ""
echo "步骤 2/5: 激活环境..."
echo "请运行以下命令激活环境，然后继续:"
echo "  conda activate agemem"
echo ""
read -p "环境已创建，是否现在安装依赖? (需要先激活环境) (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "步骤 3/5: 安装所有依赖..."
    bash scripts/install_dependencies.sh

    echo ""
    echo "步骤 4/5: 安装项目包..."
    pip install -e ".[dev]"

    echo ""
    echo "步骤 5/5: 安装pre-commit hooks..."
    pre-commit install

    echo ""
    echo "=========================================="
    echo "环境设置完成！"
    echo "=========================================="
else
    echo ""
    echo "请手动执行以下步骤:"
    echo "1. conda activate agemem"
    echo "2. bash scripts/install_dependencies.sh"
    echo "3. pip install -e '.[dev]'"
    echo "4. pre-commit install"
fi
