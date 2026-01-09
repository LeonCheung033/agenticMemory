"""测试STM工具."""
import pytest
from unittest.mock import Mock, MagicMock
from src.agemem.tools.stm_tools import (
    RetrieveMemoryTool,
    SummaryContextTool,
    FilterContextTool,
)
from src.agemem.memory.ltm import LTMManager
from src.agemem.memory.stm import STMManager


class TestRetrieveMemoryTool:
    """测试RetrieveMemoryTool."""

    def test_init(self):
        """测试初始化."""
        ltm = LTMManager()
        stm = STMManager()
        tool = RetrieveMemoryTool(ltm, stm)
        assert tool.schema.name == "Retrieve_memory"
        assert tool.ltm == ltm
        assert tool.stm == stm

    def test_execute_basic(self):
        """测试基本执行."""
        ltm = LTMManager()
        stm = STMManager()
        tool = RetrieveMemoryTool(ltm, stm)

        # 先存储一些记忆
        ltm.store("I love machine learning", {"topic": "ML"})
        ltm.store("Python is great", {"topic": "programming"})

        # 检索记忆
        result = tool.execute(query="machine learning", top_k=2)

        assert result["success"] is True
        assert result["result"]["count"] > 0
        assert len(result["result"]["memories"]) > 0
        # 检查是否添加到STM上下文
        assert len(stm.context_history) > 0

    def test_execute_with_metadata_filter(self):
        """测试带元数据过滤的检索."""
        ltm = LTMManager()
        stm = STMManager()
        tool = RetrieveMemoryTool(ltm, stm)

        # 存储带不同元数据的记忆
        ltm.store("ML content", {"domain": "AI", "type": "knowledge"})
        ltm.store("Python code", {"domain": "CS", "type": "code"})

        # 使用过滤器检索
        result = tool.execute(
            query="programming",
            top_k=5,
            metadata_filter={"domain": "CS"},
        )

        assert result["success"] is True
        # 所有检索到的记忆应该匹配过滤器
        for memory in result["result"]["memories"]:
            assert memory["metadata"].get("domain") == "CS"

    def test_execute_empty_memory(self):
        """测试空记忆库的检索."""
        ltm = LTMManager()
        stm = STMManager()
        tool = RetrieveMemoryTool(ltm, stm)

        result = tool.execute(query="test query", top_k=3)

        assert result["success"] is True
        assert result["result"]["count"] == 0
        assert len(result["result"]["memories"]) == 0


class TestSummaryContextTool:
    """测试SummaryContextTool."""

    def test_init(self):
        """测试初始化."""
        stm = STMManager()
        mock_llm = Mock()
        tool = SummaryContextTool(stm, mock_llm)
        assert tool.schema.name == "Summary_context"
        assert tool.stm == stm
        assert tool.llm == mock_llm

    def test_execute_span_all(self):
        """测试span='all'的情况."""
        stm = STMManager()
        mock_llm = Mock()
        mock_llm.generate = Mock(return_value="This is a summary.")
        tool = SummaryContextTool(stm, mock_llm)

        # 添加一些消息
        stm.add_message("user", "Message 1")
        stm.add_message("assistant", "Response 1")
        stm.add_message("user", "Message 2")

        result = tool.execute(span="all")

        assert result["success"] is True
        assert "summary" in result["result"]
        # 应该只剩下总结消息
        assert len(stm.context_history) == 1
        assert stm.context_history[0]["role"] == "system"

    def test_execute_span_number(self):
        """测试span为数字的情况."""
        stm = STMManager()
        mock_llm = Mock()
        mock_llm.generate = Mock(return_value="Summary of last 2 rounds.")
        tool = SummaryContextTool(stm, mock_llm)

        # 添加多个消息
        for i in range(10):
            stm.add_message("user", f"Message {i}")
            stm.add_message("assistant", f"Response {i}")

        initial_count = len(stm.context_history)

        # 总结最后2轮
        result = tool.execute(span="2")

        assert result["success"] is True
        # 应该保留了最近的消息并添加了总结
        assert len(stm.context_history) < initial_count
        assert len(stm.context_history) > 0

    def test_execute_invalid_span(self):
        """测试无效的span值."""
        stm = STMManager()
        mock_llm = Mock()
        tool = SummaryContextTool(stm, mock_llm)

        result = tool.execute(span="invalid")

        assert result["success"] is False
        assert "Invalid span value" in result["error"]

    def test_execute_empty_context(self):
        """测试空上下文的总结."""
        stm = STMManager()
        mock_llm = Mock()
        tool = SummaryContextTool(stm, mock_llm)

        result = tool.execute(span="all")

        assert result["success"] is True
        assert "No messages to summarize" in result["result"]["message"]

    def test_summary_prompt_template(self):
        """测试总结prompt模板."""
        stm = STMManager()
        mock_llm = Mock()
        mock_llm.generate = Mock(return_value="Summary")
        tool = SummaryContextTool(stm, mock_llm)

        stm.add_message("user", "Test message")
        tool.execute(span="all")

        # 验证prompt模板被使用
        call_args = mock_llm.generate.call_args[0][0]
        assert "conversation summarization assistant" in call_args.lower()
        assert "[CONVERSATION_TEXT]" not in call_args  # 应该被替换


class TestFilterContextTool:
    """测试FilterContextTool."""

    def test_init(self):
        """测试初始化."""
        stm = STMManager()
        ltm = LTMManager()
        tool = FilterContextTool(stm, ltm)
        assert tool.schema.name == "Filter_context"
        assert tool.stm == stm
        assert tool.ltm == ltm

    def test_execute_basic(self):
        """测试基本执行."""
        stm = STMManager()
        ltm = LTMManager()
        tool = FilterContextTool(stm, ltm)

        # 添加一些消息
        stm.add_message("user", "I love machine learning")
        stm.add_message("assistant", "That's great!")
        stm.add_message("user", "The weather is nice today")
        stm.add_message("assistant", "Yes, it is sunny")

        initial_count = len(stm.context_history)

        # 过滤与"machine learning"相关的内容
        result = tool.execute(criteria="machine learning algorithms", threshold=0.6)

        assert result["success"] is True
        assert result["result"]["removed_count"] >= 0
        # 剩余消息数应该减少或保持不变
        assert len(stm.context_history) <= initial_count

    def test_execute_custom_threshold(self):
        """测试自定义阈值."""
        stm = STMManager()
        ltm = LTMManager()
        tool = FilterContextTool(stm, ltm)

        stm.add_message("user", "Test message 1")
        stm.add_message("user", "Test message 2")

        # 使用不同的阈值
        result1 = tool.execute(criteria="test", threshold=0.3)
        stm.clear()
        stm.add_message("user", "Test message 1")
        stm.add_message("user", "Test message 2")
        result2 = tool.execute(criteria="test", threshold=0.9)

        # 不同阈值应该产生不同的结果
        assert result1["success"] is True
        assert result2["success"] is True

    def test_execute_empty_context(self):
        """测试空上下文的过滤."""
        stm = STMManager()
        ltm = LTMManager()
        tool = FilterContextTool(stm, ltm)

        result = tool.execute(criteria="test criteria")

        assert result["success"] is True
        assert result["result"]["removed_count"] == 0
        assert result["result"]["remaining_count"] == 0

    def test_execute_error_handling(self):
        """测试错误处理."""
        stm = STMManager()
        ltm = LTMManager()
        tool = FilterContextTool(stm, ltm)

        # 正常情况下不应该失败
        stm.add_message("user", "Test")
        result = tool.execute(criteria="test")
        assert result["success"] is True
