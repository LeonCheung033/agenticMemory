"""测试基础记忆接口"""
import pytest
from datetime import datetime
from src.agemem.memory.base import MemoryItem, BaseMemory


class TestMemoryItem:
    """测试MemoryItem数据结构"""

    def test_create_memory_item(self):
        """测试创建MemoryItem"""
        item = MemoryItem(content="Test content", metadata={"type": "test"})
        assert item.content == "Test content"
        assert item.metadata == {"type": "test"}
        assert item.memory_id is not None
        assert item.access_count == 0

    def test_to_dict(self):
        """测试转换为字典"""
        item = MemoryItem(content="Test content", metadata={"type": "test"})
        data = item.to_dict()
        assert data["content"] == "Test content"
        assert data["metadata"] == {"type": "test"}
        assert "memory_id" in data
        assert "created_at" in data
        assert "updated_at" in data
        assert "access_count" in data

    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            "memory_id": "test-id",
            "content": "Test content",
            "metadata": {"type": "test"},
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
            "access_count": 5,
        }
        item = MemoryItem.from_dict(data)
        assert item.memory_id == "test-id"
        assert item.content == "Test content"
        assert item.metadata == {"type": "test"}
        assert item.access_count == 5

    def test_round_trip(self):
        """测试序列化往返"""
        original = MemoryItem(content="Test content", metadata={"type": "test", "key": "value"})
        data = original.to_dict()
        restored = MemoryItem.from_dict(data)
        assert restored.content == original.content
        assert restored.metadata == original.metadata
        assert restored.memory_id == original.memory_id


class TestBaseMemory:
    """测试BaseMemory抽象类"""

    def test_base_memory_is_abstract(self):
        """测试BaseMemory是抽象类，不能直接实例化"""
        with pytest.raises(TypeError):
            BaseMemory()
