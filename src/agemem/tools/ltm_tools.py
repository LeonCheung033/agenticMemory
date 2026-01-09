"""LTM工具实现."""
from typing import Dict, Any, Optional

from .base import BaseTool, ToolSchema
from ..memory.ltm import LTMManager


class AddMemoryTool(BaseTool):
    """Add_memory工具."""

    def __init__(self, ltm_manager: LTMManager):
        super().__init__()
        self.ltm = ltm_manager
        self._schema = ToolSchema(
            name="Add_memory",
            description="Adds new information to external memory store for future reference.",
            parameters={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The content to store in memory.",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional metadata tags to categorize and filter the memory.",
                        "additionalProperties": True,
                    },
                    "memory_type": {
                        "type": "string",
                        "description": "The type of memory being stored.",
                    },
                },
                "required": ["content"],
            },
        )

    @property
    def schema(self) -> ToolSchema:
        return self._schema

    def execute(
        self,
        content: str,
        metadata: Optional[Dict] = None,
        memory_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """执行Add_memory工具."""
        try:
            # 将memory_type添加到metadata
            if metadata is None:
                metadata = {}
            if memory_type:
                metadata["memory_type"] = memory_type

            memory_id = self.ltm.store(content, metadata)

            return {
                "success": True,
                "result": {
                    "memory_id": memory_id,
                    "message": f"Memory stored successfully with ID: {memory_id}",
                },
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class UpdateMemoryTool(BaseTool):
    """Update_memory工具."""

    def __init__(self, ltm_manager: LTMManager):
        super().__init__()
        self.ltm = ltm_manager
        self._schema = ToolSchema(
            name="Update_memory",
            description="Updates existing memory. Requires memory_id from prior retrieval.",
            parameters={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "The unique identifier of the memory to update. Must be obtained from a previous memory retrieval operation.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The new content to replace the existing memory content.",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Updated metadata for the memory.",
                        "additionalProperties": True,
                    },
                },
                "required": ["memory_id", "content"],
            },
        )

    @property
    def schema(self) -> ToolSchema:
        return self._schema

    def execute(
        self,
        memory_id: str,
        content: str,
        metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """执行Update_memory工具."""
        try:
            success = self.ltm.update(memory_id, content, metadata)

            return {
                "success": success,
                "result": {
                    "memory_id": memory_id,
                    "message": "Memory updated successfully" if success else "Update failed",
                },
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class DeleteMemoryTool(BaseTool):
    """Delete_memory工具."""

    def __init__(self, ltm_manager: LTMManager):
        super().__init__()
        self.ltm = ltm_manager
        self._schema = ToolSchema(
            name="Delete_memory",
            description="Removes memory from store. Requires confirmation.",
            parameters={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "The unique identifier of the memory to delete. Must be obtained from a previous memory retrieval operation.",
                    },
                    "confirmation": {
                        "type": "boolean",
                        "description": "Confirmation that this memory should be permanently deleted.",
                    },
                },
                "required": ["memory_id", "confirmation"],
            },
        )

    @property
    def schema(self) -> ToolSchema:
        return self._schema

    def execute(self, memory_id: str, confirmation: bool) -> Dict[str, Any]:
        """执行Delete_memory工具."""
        try:
            if not confirmation:
                return {
                    "success": False,
                    "error": "Deletion requires confirmation. Set confirmation=True to proceed.",
                }

            success = self.ltm.delete(memory_id)

            return {
                "success": success,
                "result": {
                    "memory_id": memory_id,
                    "message": "Memory deleted successfully" if success else "Deletion failed",
                },
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
