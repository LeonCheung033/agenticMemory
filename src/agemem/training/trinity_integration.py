"""Trinity-RFT集成模块."""
# 导入AgeMem组件（这会触发workflow注册）
from .workflows.agemem_workflow import AgeMemWorkflow
from .reward import AgeMemRewardFunction

# 尝试导入Trinity-RFT的注册器（如果可用）
try:
    from trinity.common.workflows import WORKFLOWS  # noqa: F401

    # 如果Trinity-RFT可用，确保workflow已注册
    # 注意：装饰器已经在agemem_workflow.py中处理了注册
    TRINITY_AVAILABLE = True
except ImportError:
    # Trinity-RFT未安装，使用兼容的注册器
    TRINITY_AVAILABLE = False

__all__ = [
    "AgeMemWorkflow",
    "AgeMemRewardFunction",
    "TRINITY_AVAILABLE",
]
