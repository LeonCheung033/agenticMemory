"""测试STM管理器."""
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from src.agemem.memory.stm import STMManager
from src.agemem.utils.embedding import EmbeddingModel


class TestSTMManager:
    """测试STMManager."""

    def test_init(self):
        """测试初始化."""
        stm = STMManager()
        assert stm.max_tokens == 8192
        assert len(stm.context_history) == 0
        assert stm.token_counter is not None

    def test_init_custom_params(self):
        """测试自定义参数初始化."""
        token_counter = Mock()
        stm = STMManager(max_tokens=4096, token_counter=token_counter)
        assert stm.max_tokens == 4096
        assert stm.token_counter == token_counter

    def test_add_message(self):
        """测试添加消息."""
        stm = STMManager()
        stm.add_message("user", "Hello, world!")

        assert len(stm.context_history) == 1
        msg = stm.context_history[0]
        assert msg["role"] == "user"
        assert msg["content"] == "Hello, world!"
        assert "tokens" in msg
        assert "timestamp" in msg
        assert "metadata" in msg

    def test_add_message_with_metadata(self):
        """测试添加带元数据的消息."""
        stm = STMManager()
        metadata = {"source": "test", "priority": "high"}
        stm.add_message("system", "System message", metadata=metadata)

        msg = stm.context_history[0]
        assert msg["metadata"] == metadata

    def test_get_token_count(self):
        """测试获取token计数."""
        stm = STMManager()
        assert stm.get_token_count() == 0

        stm.add_message("user", "Hello")
        count1 = stm.get_token_count()
        assert count1 > 0

        stm.add_message("assistant", "Hi there!")
        count2 = stm.get_token_count()
        assert count2 > count1

    def test_get_messages(self):
        """测试获取消息列表."""
        stm = STMManager()
        stm.add_message("user", "Message 1")
        stm.add_message("assistant", "Message 2")

        messages = stm.get_messages()
        assert len(messages) == 2
        assert messages[0]["content"] == "Message 1"
        assert messages[1]["content"] == "Message 2"
        # 应该返回副本，不是引用
        assert messages is not stm.context_history

    def test_clear(self):
        """测试清空上下文."""
        stm = STMManager()
        stm.add_message("user", "Message 1")
        stm.add_message("assistant", "Message 2")

        assert len(stm.context_history) == 2
        stm.clear()
        assert len(stm.context_history) == 0

    def test_remove_messages(self):
        """测试移除消息."""
        stm = STMManager()
        stm.add_message("user", "Message 1")
        stm.add_message("assistant", "Message 2")
        stm.add_message("user", "Message 3")

        # 移除索引1和2的消息
        stm.remove_messages([1, 2])

        assert len(stm.context_history) == 1
        assert stm.context_history[0]["content"] == "Message 1"

    def test_remove_messages_invalid_indices(self):
        """测试移除无效索引."""
        stm = STMManager()
        stm.add_message("user", "Message 1")

        # 尝试移除无效索引
        stm.remove_messages([10, -1])

        # 应该仍然有1条消息
        assert len(stm.context_history) == 1

    def test_filter_messages(self):
        """测试根据相似度过滤消息."""
        stm = STMManager()
        embedding_model = EmbeddingModel()

        # 添加一些消息
        stm.add_message("user", "I love machine learning")
        stm.add_message("assistant", "That's great!")
        stm.add_message("user", "The weather is nice today")
        stm.add_message("assistant", "Yes, it is sunny")

        # 创建过滤标准（与"machine learning"相关）
        criteria = "machine learning algorithms"
        criteria_embedding = embedding_model.encode(criteria, convert_to_numpy=True)

        # 过滤（保留相似度低于0.6的消息，即不相关的消息）
        removed = stm.filter_messages(
            criteria_embedding, threshold=0.6, embedding_model=embedding_model
        )

        # 应该移除一些消息
        assert removed > 0
        assert len(stm.context_history) < 4

    def test_filter_messages_no_embedding_model(self):
        """测试没有嵌入模型时的过滤."""
        stm = STMManager()
        stm.add_message("user", "Message 1")

        removed = stm.filter_messages(
            np.array([1.0, 2.0, 3.0]), threshold=0.6, embedding_model=None
        )

        # 应该没有移除任何消息
        assert removed == 0
        assert len(stm.context_history) == 1

    def test_summarize(self):
        """测试总结功能."""
        stm = STMManager()

        # 创建mock LLM
        mock_llm = Mock()
        mock_llm.generate = Mock(return_value="This is a summary of the conversation.")

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]

        prompt_template = "Summarize: [CONVERSATION_TEXT]"

        summary = stm.summarize(mock_llm, messages, prompt_template)

        assert summary == "This is a summary of the conversation."
        # 验证LLM被调用
        mock_llm.generate.assert_called_once()
        # 验证prompt包含对话内容
        call_args = mock_llm.generate.call_args[0][0]
        assert "Hello" in call_args
        assert "Hi there!" in call_args

    def test_summarize_empty_messages(self):
        """测试总结空消息列表."""
        stm = STMManager()
        mock_llm = Mock()
        mock_llm.generate = Mock(return_value="Empty summary")

        summary = stm.summarize(mock_llm, [], "Template: [CONVERSATION_TEXT]")

        assert summary == "Empty summary"
        # 验证prompt包含空内容
        call_args = mock_llm.generate.call_args[0][0]
        assert "unknown:" in call_args or len(call_args) > 0

    def test_token_count_accuracy(self):
        """测试token计数的准确性."""
        stm = STMManager()
        text = "Hello, world!"
        stm.add_message("user", text)

        # 手动计算token数
        manual_count = stm.token_counter.count(text)

        # 应该与消息中的token数一致
        assert stm.context_history[0]["tokens"] == manual_count
