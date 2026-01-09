"""工具基类."""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ToolSchema:
    """工具Schema定义（JSON Schema格式）."""

    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema格式的参数定义

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于LLM）."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class BaseTool(ABC):
    """工具基类."""

    def __init__(self):
        self._schema: Optional[ToolSchema] = None

    @property
    @abstractmethod
    def schema(self) -> ToolSchema:
        """返回工具的Schema."""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        执行工具.

        Args:
            **kwargs: 工具参数

        Returns:
            Dict[str, Any]: 执行结果
            {
                "success": bool,
                "result": {...} 或 "error": str
            }
        """
        pass

    def validate_args(self, args: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        验证参数.

        Args:
            args: 参数字典

        Returns:
            (is_valid, error_message)
        """
        schema = self.schema
        required = schema.parameters.get("required", [])

        # 检查必需参数
        for param in required:
            if param not in args:
                return False, f"Missing required parameter: {param}"

        return True, None
