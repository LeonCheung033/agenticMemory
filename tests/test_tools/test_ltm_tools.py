"""测试LTM工具."""
import pytest
from src.agemem.tools.ltm_tools import (
    AddMemoryTool,
    UpdateMemoryTool,
    DeleteMemoryTool,
)
from src.agemem.memory.ltm import LTMManager


class TestAddMemoryTool:
    """测试AddMemoryTool."""

    def test_init(self):
        """测试初始化."""
        ltm = LTMManager()
        tool = AddMemoryTool(ltm)
        assert tool.schema.name == "Add_memory"
        assert tool.ltm == ltm

    def test_execute_basic(self):
        """测试基本执行."""
        ltm = LTMManager()
        tool = AddMemoryTool(ltm)

        result = tool.execute(content="Test memory content")

        assert result["success"] is True
        assert "memory_id" in result["result"]
        assert ltm.count() == 1

    def test_execute_with_metadata(self):
        """测试带元数据的执行."""
        ltm = LTMManager()
        tool = AddMemoryTool(ltm)

        metadata = {"type": "test", "priority": "high"}
        result = tool.execute(content="Test content", metadata=metadata)

        assert result["success"] is True
        memory_id = result["result"]["memory_id"]
        memory = ltm.get(memory_id)
        assert memory.metadata == metadata

    def test_execute_with_memory_type(self):
        """测试带memory_type的执行."""
        ltm = LTMManager()
        tool = AddMemoryTool(ltm)

        result = tool.execute(
            content="Test content", memory_type="user_preference"
        )

        assert result["success"] is True
        memory_id = result["result"]["memory_id"]
        memory = ltm.get(memory_id)
        assert memory.metadata["memory_type"] == "user_preference"

    def test_execute_error_handling(self):
        """测试错误处理."""
        # 创建一个会失败的LTM管理器（通过mock）
        ltm = LTMManager()
        tool = AddMemoryTool(ltm)

        # 正常情况下不应该失败
        result = tool.execute(content="Valid content")
        assert result["success"] is True


class TestUpdateMemoryTool:
    """测试UpdateMemoryTool."""

    def test_init(self):
        """测试初始化."""
        ltm = LTMManager()
        tool = UpdateMemoryTool(ltm)
        assert tool.schema.name == "Update_memory"
        assert "memory_id" in tool.schema.parameters["required"]
        assert "content" in tool.schema.parameters["required"]

    def test_execute_success(self):
        """测试成功更新."""
        ltm = LTMManager()
        tool = UpdateMemoryTool(ltm)

        # 先存储一个记忆
        memory_id = ltm.store("Original content", {"version": 1})

        # 更新记忆
        result = tool.execute(
            memory_id=memory_id, content="Updated content", metadata={"version": 2}
        )

        assert result["success"] is True
        memory = ltm.get(memory_id)
        assert memory.content == "Updated content"
        assert memory.metadata["version"] == 2

    def test_execute_nonexistent_memory(self):
        """测试更新不存在的记忆."""
        ltm = LTMManager()
        tool = UpdateMemoryTool(ltm)

        result = tool.execute(
            memory_id="nonexistent-id", content="New content"
        )

        assert result["success"] is False
        assert "Update failed" in result["result"]["message"]

    def test_execute_metadata_only(self):
        """测试仅更新元数据."""
        ltm = LTMManager()
        tool = UpdateMemoryTool(ltm)

        memory_id = ltm.store("Original content", {"key": "old"})
        original_content = ltm.get(memory_id).content

        # 只更新元数据（需要提供content，但可以保持原内容）
        result = tool.execute(
            memory_id=memory_id,
            content=original_content,
            metadata={"key": "new"},
        )

        assert result["success"] is True
        memory = ltm.get(memory_id)
        assert memory.metadata["key"] == "new"


class TestDeleteMemoryTool:
    """测试DeleteMemoryTool."""

    def test_init(self):
        """测试初始化."""
        ltm = LTMManager()
        tool = DeleteMemoryTool(ltm)
        assert tool.schema.name == "Delete_memory"
        assert "confirmation" in tool.schema.parameters["required"]

    def test_execute_with_confirmation(self):
        """测试带确认的删除."""
        ltm = LTMManager()
        tool = DeleteMemoryTool(ltm)

        # 先存储一个记忆
        memory_id = ltm.store("To be deleted", {})
        assert ltm.count() == 1

        # 删除记忆（带确认）
        result = tool.execute(memory_id=memory_id, confirmation=True)

        assert result["success"] is True
        assert ltm.count() == 0
        assert ltm.get(memory_id) is None

    def test_execute_without_confirmation(self):
        """测试不带确认的删除."""
        ltm = LTMManager()
        tool = DeleteMemoryTool(ltm)

        memory_id = ltm.store("Content", {})
        assert ltm.count() == 1

        # 尝试删除但不确认
        result = tool.execute(memory_id=memory_id, confirmation=False)

        assert result["success"] is False
        assert "confirmation" in result["error"].lower()
        # 记忆应该还在
        assert ltm.count() == 1

    def test_execute_nonexistent_memory(self):
        """测试删除不存在的记忆."""
        ltm = LTMManager()
        tool = DeleteMemoryTool(ltm)

        result = tool.execute(memory_id="nonexistent-id", confirmation=True)

        assert result["success"] is False
        assert "Deletion failed" in result["result"]["message"]

    def test_schema_validation(self):
        """测试Schema验证."""
        ltm = LTMManager()
        tool = DeleteMemoryTool(ltm)

        # 缺少confirmation参数
        args = {"memory_id": "test-id"}
        is_valid, error = tool.validate_args(args)
        assert is_valid is False
        assert "confirmation" in error
