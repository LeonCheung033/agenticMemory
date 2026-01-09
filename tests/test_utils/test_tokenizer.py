"""测试Token计数工具"""
import pytest
from src.agemem.utils.tokenizer import TokenCounter


class TestTokenCounter:
    """测试TokenCounter"""

    def test_init(self):
        """测试初始化"""
        counter = TokenCounter()
        assert counter.encoder is not None

    def test_count_simple_text(self):
        """测试简单文本计数"""
        counter = TokenCounter()
        text = "Hello world"
        count = counter.count(text)
        assert count > 0
        assert isinstance(count, int)

    def test_count_empty_text(self):
        """测试空文本计数"""
        counter = TokenCounter()
        count = counter.count("")
        assert count == 0

    def test_count_long_text(self):
        """测试长文本计数"""
        counter = TokenCounter()
        text = "This is a longer text with multiple sentences. " * 10
        count = counter.count(text)
        assert count > 10  # 应该有很多tokens

    def test_count_messages(self):
        """测试消息列表计数"""
        counter = TokenCounter()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "system", "content": "You are a helpful assistant."},
        ]
        total = counter.count_messages(messages)
        assert total > 0
        # 总token数应该大于单个消息的token数
        single_count = counter.count("Hello")
        assert total > single_count

    def test_count_messages_empty(self):
        """测试空消息列表"""
        counter = TokenCounter()
        total = counter.count_messages([])
        assert total == 0

    def test_count_consistency(self):
        """测试计数一致性"""
        counter = TokenCounter()
        text = "Test message"
        count1 = counter.count(text)
        count2 = counter.count(text)
        assert count1 == count2
