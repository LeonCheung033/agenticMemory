from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="agemem",
    version="0.1.0",
    author="Agentic Memory Team",
    description="Agentic Memory: Unified Long-Term and Short-Term Memory Management for LLM Agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/agenticMemory",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "transformers>=4.35.0",
        "accelerate>=0.24.0",
        "datasets>=2.14.0",
        "faiss-cpu",
        "chromadb>=0.4.0",
        "tiktoken>=0.5.0",
        "openai>=1.0.0",
        "wandb>=0.15.0",
        "tensorboard>=2.14.0",
        "pyyaml>=6.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "sentence-transformers>=2.2.0",
        "rouge-score>=0.1.2",
        "nltk>=3.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.5.0",
        ],
    },
)
