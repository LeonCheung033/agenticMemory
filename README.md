# Agentic Memory (AgeMem)

Agentic Memory: Learning Unified Long-Term and Short-Term Memory Management for Large Language Model Agents

## 快速开始

### 环境要求

- Python >= 3.10
- Conda (推荐) 或 pip
- CUDA 11.8+ (如果使用GPU)

### 安装步骤

#### 使用 Conda (推荐)

**方式1: 使用自动化脚本（推荐）**
```bash
bash scripts/setup_env.sh
```

**方式2: 手动安装**

1. **创建conda环境（仅Python和pip）**:
```bash
conda env create -f environment.yml
conda activate agemem
```

2. **安装所有依赖**:
```bash
bash scripts/install_dependencies.sh
```

3. **安装项目包**:
```bash
pip install -e ".[dev]"
```

4. **安装pre-commit hooks**:
```bash
pre-commit install
```

#### 使用 pip

```bash
pip install -r requirements.txt
pip install -e ".[dev]"
```

## 项目结构

```
agenticMemory/
├── src/agemem/          # 主包代码
├── configs/             # 配置文件
├── data/                # 数据目录
├── scripts/             # 脚本文件
├── tests/               # 测试代码
└── dev_docs/            # 开发文档（不对外暴露）
```

## 开发工作流

本项目使用 Git Flow 工作流：

- `main`: 生产分支
- `develop`: 开发分支
- `feature/*`: 功能分支
- `release/*`: 发布分支
- `hotfix/*`: 热修复分支

### 开始新功能开发

```bash
git checkout develop
git pull origin develop
git flow feature start your-feature-name
# ... 开发代码 ...
git flow feature finish your-feature-name
```

## 代码规范

- 使用 `black` 进行代码格式化
- 使用 `flake8` 进行代码检查
- 使用 `mypy` 进行类型检查
- 遵循 Conventional Commits 规范

运行代码检查:
```bash
black src/
flake8 src/
mypy src/
```

## 测试

运行测试:
```bash
pytest tests/ -v
```

运行测试并生成覆盖率报告:
```bash
pytest tests/ --cov=src/agemem --cov-report=html
```

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！
