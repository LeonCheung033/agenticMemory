"""短期记忆（STM）管理器."""
from typing import List, Dict, Optional, Any
from datetime import datetime
import numpy as np

from ..utils.tokenizer import TokenCounter


class STMManager:
    """
    短期记忆管理器.

    管理对话上下文（消息列表），支持：
    - 添加消息
    - Token计数
    - 总结功能（使用论文中的prompt）
    """

    def __init__(
        self,
        max_tokens: int = 8192,
        token_counter: Optional[TokenCounter] = None,
    ):
        """
        初始化STM管理器.

        Args:
            max_tokens: 最大token数
            token_counter: Token计数器
        """
        self.max_tokens = max_tokens
        self.token_counter = token_counter or TokenCounter()

        # 上下文历史：消息列表
        # 每个消息格式：{"role": "user|assistant|system|tool", "content": "...", "tokens": int, "metadata": {...}}
        self.context_history: List[Dict[str, Any]] = []

    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        添加消息到上下文.

        Args:
            role: 消息角色（user, assistant, system, tool）
            content: 消息内容
            metadata: 元数据
        """
        tokens = self.token_counter.count(content)

        message = {
            "role": role,
            "content": content,
            "tokens": tokens,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
        }

        self.context_history.append(message)

    def get_token_count(self) -> int:
        """获取当前上下文的token总数."""
        return sum(msg.get("tokens", 0) for msg in self.context_history)

    def get_messages(self) -> List[Dict[str, Any]]:
        """获取所有消息."""
        return self.context_history.copy()

    def clear(self):
        """清空上下文."""
        self.context_history = []

    def remove_messages(self, indices: List[int]):
        """
        移除指定索引的消息.

        Args:
            indices: 要移除的消息索引列表（从后往前，避免索引变化）
        """
        # 从后往前删除，避免索引变化
        for idx in sorted(indices, reverse=True):
            if 0 <= idx < len(self.context_history):
                del self.context_history[idx]

    def filter_messages(
        self,
        criteria_embedding: np.ndarray,
        threshold: float = 0.6,
        embedding_model: Optional[Any] = None,
    ) -> int:
        """
        根据相似度过滤消息.

        根据论文公式：
        C_t' = {u_i ∈ C_t | sim(c, u_i) < θ}

        Args:
            criteria_embedding: 过滤标准的嵌入向量
            threshold: 相似度阈值θ（默认0.6）
            embedding_model: 嵌入模型（用于计算消息嵌入）

        Returns:
            int: 移除的消息数量
        """
        if not embedding_model:
            return 0

        filtered_context = []
        removed_count = 0

        for msg in self.context_history:
            msg_content = msg.get("content", "")
            msg_embedding = embedding_model.encode(
                msg_content, convert_to_numpy=True
            )

            # 计算余弦相似度
            # 由于向量已归一化，可以直接计算点积
            similarity = np.dot(
                criteria_embedding.flatten(), msg_embedding.flatten()
            )

            # 保留相似度低于阈值的消息（即不相关的消息保留）
            if similarity < threshold:
                filtered_context.append(msg)
            else:
                removed_count += 1

        self.context_history = filtered_context
        return removed_count

    def summarize(
        self,
        llm: Any,
        messages: List[Dict[str, Any]],
        prompt_template: str,
    ) -> str:
        """
        总结消息列表.

        使用论文附录中的总结prompt.

        Args:
            llm: LLM模型（需要有generate方法）
            messages: 要总结的消息列表
            prompt_template: 总结prompt模板

        Returns:
            str: 总结文本
        """
        # 构建对话文本
        conversation_text = "\n".join(
            [
                f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
                for msg in messages
            ]
        )

        # 填充prompt模板
        prompt = prompt_template.replace("[CONVERSATION_TEXT]", conversation_text)

        # 生成总结
        summary = llm.generate(prompt)

        return summary
