"""测试LTM管理器."""
import pytest
import tempfile
import shutil
from pathlib import Path
from src.agemem.memory.ltm import LTMManager
from src.agemem.memory.base import MemoryItem


class TestLTMManager:
    """测试LTMManager."""

    def test_init(self):
        """测试初始化."""
        ltm = LTMManager()
        assert ltm.dimension == 384
        assert ltm.count() == 0
        assert ltm.embedding_model is not None

    def test_store(self):
        """测试存储记忆."""
        ltm = LTMManager()
        memory_id = ltm.store("Test memory content", {"type": "test"})

        assert memory_id is not None
        assert ltm.count() == 1

        memory = ltm.get(memory_id)
        assert memory is not None
        assert memory.content == "Test memory content"
        assert memory.metadata == {"type": "test"}
        assert memory.embedding is not None
        assert len(memory.embedding) == 384

    def test_retrieve(self):
        """测试检索记忆."""
        ltm = LTMManager()

        # 存储多个记忆
        ltm.store("I love machine learning", {"topic": "ML"})
        ltm.store("Python is a great language", {"topic": "programming"})
        ltm.store("The weather is nice today", {"topic": "weather"})

        # 检索相关记忆
        results = ltm.retrieve("machine learning algorithms", top_k=2)

        assert len(results) > 0
        # 第一个结果应该与查询最相似
        assert "machine learning" in results[0].content.lower()

    def test_retrieve_with_filters(self):
        """测试带过滤器的检索."""
        ltm = LTMManager()

        ltm.store("ML is great", {"topic": "ML", "domain": "AI"})
        ltm.store("Python is fun", {"topic": "programming", "domain": "CS"})
        ltm.store("Deep learning rocks", {"topic": "ML", "domain": "AI"})

        # 使用过滤器检索
        results = ltm.retrieve(
            "machine learning", top_k=5, filters={"domain": "AI"}
        )

        assert len(results) > 0
        for result in results:
            assert result.metadata.get("domain") == "AI"

    def test_update(self):
        """测试更新记忆."""
        ltm = LTMManager()
        memory_id = ltm.store("Original content", {"version": 1})

        # 更新内容
        success = ltm.update(memory_id, "Updated content", {"version": 2})

        assert success is True
        memory = ltm.get(memory_id)
        assert memory.content == "Updated content"
        assert memory.metadata["version"] == 2
        assert memory.embedding is not None  # 嵌入应该被更新

    def test_update_metadata_only(self):
        """测试仅更新元数据."""
        ltm = LTMManager()
        memory_id = ltm.store("Test content", {"key": "old"})
        original_content = ltm.get(memory_id).content
        original_embedding = ltm.get(memory_id).embedding.copy()

        # 只更新元数据
        success = ltm.update(memory_id, metadata={"key": "new"})

        assert success is True
        memory = ltm.get(memory_id)
        assert memory.content == original_content
        assert memory.metadata["key"] == "new"
        # 嵌入不应该改变
        assert memory.embedding == original_embedding

    def test_delete(self):
        """测试删除记忆."""
        ltm = LTMManager()
        memory_id = ltm.store("To be deleted", {})

        assert ltm.count() == 1

        success = ltm.delete(memory_id)

        assert success is True
        assert ltm.count() == 0
        assert ltm.get(memory_id) is None

    def test_delete_nonexistent(self):
        """测试删除不存在的记忆."""
        ltm = LTMManager()
        success = ltm.delete("nonexistent-id")
        assert success is False

    def test_get(self):
        """测试根据ID获取记忆."""
        ltm = LTMManager()
        memory_id = ltm.store("Test content", {})

        memory = ltm.get(memory_id)
        assert memory is not None
        assert memory.content == "Test content"

        # 获取不存在的记忆
        assert ltm.get("nonexistent-id") is None

    def test_count(self):
        """测试记忆计数."""
        ltm = LTMManager()
        assert ltm.count() == 0

        ltm.store("Memory 1", {})
        assert ltm.count() == 1

        ltm.store("Memory 2", {})
        assert ltm.count() == 2

        ltm.delete(list(ltm.memories.keys())[0])
        assert ltm.count() == 1

    def test_save_and_load(self):
        """测试保存和加载."""
        ltm = LTMManager()
        ltm.store("Memory 1", {"key": "value1"})
        ltm.store("Memory 2", {"key": "value2"})

        # 保存到临时目录
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = str(Path(tmpdir) / "test_index")

            ltm.save(index_path)

            # 创建新的LTM管理器并加载
            ltm2 = LTMManager(index_path=index_path)

            assert ltm2.count() == 2
            # 检查记忆内容是否正确
            memories = list(ltm2.memories.values())
            contents = [m.content for m in memories]
            assert "Memory 1" in contents
            assert "Memory 2" in contents

    def test_retrieve_empty(self):
        """测试空记忆库的检索."""
        ltm = LTMManager()
        results = ltm.retrieve("query", top_k=5)
        assert len(results) == 0

    def test_retrieve_access_count(self):
        """测试检索时更新访问计数."""
        ltm = LTMManager()
        memory_id = ltm.store("Test content", {})

        memory = ltm.get(memory_id)
        initial_count = memory.access_count

        # 检索记忆
        ltm.retrieve("test", top_k=1)

        # 访问计数应该增加
        memory = ltm.get(memory_id)
        assert memory.access_count > initial_count
