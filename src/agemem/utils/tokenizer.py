"""Token计数工具"""
import tiktoken
from typing import List, Dict, Any


class TokenCounter:
    """Token计数器"""

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        初始化tokenizer

        Args:
            model_name: 模型名称（用于选择编码器）
        """
        try:
            self.encoder = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # 如果模型不存在，使用cl100k_base（GPT-3.5/4的编码）
            self.encoder = tiktoken.get_encoding("cl100k_base")

    def count(self, text: str) -> int:
        """计算文本的token数"""
        return len(self.encoder.encode(text))

    def count_messages(self, messages: List[Dict[str, Any]]) -> int:
        """
        计算消息列表的总token数

        Args:
            messages: 消息列表，每个消息包含 'role' 和 'content'

        Returns:
            总token数
        """
        total = 0
        for msg in messages:
            # 计算role和content的token
            role_text = f"{msg.get('role', '')}: {msg.get('content', '')}"
            total += self.count(role_text)
        return total
