"""基础记忆接口和数据结构"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
import uuid


@dataclass
class MemoryItem:
    """
    记忆项数据结构

    根据论文：m = (c, enc(c), metadata)
    其中：
    - c: 内容字符串
    - enc(c): 嵌入向量
    - metadata: 元数据（timestamp, source, tags等）
    """

    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "access_count": self.access_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryItem":
        """从字典创建"""
        item = cls(
            memory_id=data["memory_id"],
            content=data["content"],
            embedding=data.get("embedding"),
            metadata=data.get("metadata", {}),
            access_count=data.get("access_count", 0),
        )
        if "created_at" in data:
            item.created_at = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data:
            item.updated_at = datetime.fromisoformat(data["updated_at"])
        return item


class BaseMemory(ABC):
    """基础记忆抽象类"""

    @abstractmethod
    def store(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        存储记忆

        Args:
            content: 记忆内容
            metadata: 元数据

        Returns:
            memory_id: 记忆ID
        """
        pass

    @abstractmethod
    def retrieve(
        self, query: str, top_k: int = 3, filters: Optional[Dict[str, Any]] = None
    ) -> List[MemoryItem]:
        """
        检索记忆

        根据论文公式：
        Retrieve(q, k) = TopK(M_t, sim(q, m_i), k)
        sim(q, m_i) = (enc(q)^T * enc(m_i)) / (||enc(q)|| * ||enc(m_i)||)

        Args:
            query: 查询文本
            top_k: 返回前k个
            filters: 元数据过滤器

        Returns:
            List[MemoryItem]: 检索到的记忆列表
        """
        pass

    @abstractmethod
    def update(
        self,
        memory_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        更新记忆

        根据论文公式：m_i <- (c', enc(c'), metadata')

        Args:
            memory_id: 记忆ID
            content: 新内容
            metadata: 新元数据

        Returns:
            bool: 是否成功
        """
        pass

    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        """
        删除记忆

        根据论文公式：M_{t+1} = M_t \\ {m_i}

        Args:
            memory_id: 记忆ID

        Returns:
            bool: 是否成功
        """
        pass

    @abstractmethod
    def get(self, memory_id: str) -> Optional[MemoryItem]:
        """根据ID获取记忆"""
        pass

    @abstractmethod
    def count(self) -> int:
        """返回记忆总数"""
        pass
