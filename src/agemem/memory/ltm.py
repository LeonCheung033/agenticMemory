"""长期记忆（LTM）管理器."""
from typing import List, Dict, Optional, Any
from datetime import datetime
import numpy as np
import faiss
from pathlib import Path
import json

from .base import BaseMemory, MemoryItem
from ..utils.embedding import EmbeddingModel


class LTMManager(BaseMemory):
    """
    长期记忆管理器.

    使用FAISS进行向量检索，支持：
    - 存储记忆（带嵌入）
    - 基于语义相似度的检索
    - 更新和删除记忆
    """

    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        index_path: Optional[str] = None,
        dimension: int = 384,  # all-MiniLM-L6-v2的维度
    ):
        """
        初始化LTM管理器.

        Args:
            embedding_model: 嵌入模型（如果为None，会创建默认模型）
            index_path: FAISS索引保存路径
            dimension: 嵌入向量维度
        """
        self.embedding_model = embedding_model or EmbeddingModel()
        self.dimension = dimension
        self.index_path = index_path

        # FAISS索引（使用内积，因为向量已归一化）
        self.index = faiss.IndexFlatIP(dimension)

        # 内存存储：memory_id -> MemoryItem
        self.memories: Dict[str, MemoryItem] = {}

        # 如果存在保存的索引，加载它
        if index_path and Path(f"{index_path}.index").exists():
            self.load(index_path)

    def store(
        self, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        存储记忆.

        根据论文：m_new = (c, enc(c), metadata)
        M_{t+1} = M_t ∪ {m_new}

        Args:
            content: 记忆内容
            metadata: 元数据

        Returns:
            memory_id: 新创建的记忆ID
        """
        # 生成嵌入
        embedding = self.embedding_model.encode(content, convert_to_numpy=True)
        embedding = embedding.flatten()  # 确保是1D数组

        # 创建记忆项
        memory_item = MemoryItem(
            content=content,
            embedding=embedding.tolist(),
            metadata=metadata or {},
        )

        # 添加到索引
        self.index.add(embedding.reshape(1, -1))

        # 存储到内存
        self.memories[memory_item.memory_id] = memory_item

        return memory_item.memory_id

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[MemoryItem]:
        """
        检索记忆.

        根据论文公式：
        Retrieve(q, k) = TopK(M_t, sim(q, m_i), k)

        Args:
            query: 查询文本
            top_k: 返回前k个
            filters: 元数据过滤器（可选）

        Returns:
            List[MemoryItem]: 检索到的记忆列表，按相似度排序
        """
        if len(self.memories) == 0:
            return []

        # 生成查询嵌入
        query_embedding = self.embedding_model.encode(
            query, convert_to_numpy=True
        )
        query_embedding = query_embedding.reshape(1, -1)

        # 搜索（返回top_k个最相似的）
        k = min(top_k, len(self.memories))
        similarities, indices = self.index.search(query_embedding, k)

        # 获取对应的记忆项
        results = []
        memory_list = list(self.memories.values())

        for i, idx in enumerate(indices[0]):
            if idx < len(memory_list):
                memory_item = memory_list[idx]

                # 应用元数据过滤器
                if filters:
                    if not self._match_filters(memory_item.metadata, filters):
                        continue

                # 更新访问计数
                memory_item.access_count += 1

                results.append(memory_item)

        return results

    def update(
        self,
        memory_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        更新记忆.

        根据论文：m_i <- (c', enc(c'), metadata')

        Args:
            memory_id: 记忆ID
            content: 新内容
            metadata: 新元数据

        Returns:
            bool: 是否成功
        """
        if memory_id not in self.memories:
            return False

        memory_item = self.memories[memory_id]

        # 更新内容
        if content is not None:
            memory_item.content = content
            # 重新生成嵌入
            new_embedding = self.embedding_model.encode(
                content, convert_to_numpy=True
            ).flatten()
            memory_item.embedding = new_embedding.tolist()

            # 更新FAISS索引（需要重建，因为FAISS不支持直接更新）
            # 这里简化处理：删除旧索引，添加新索引
            # 实际应用中可以使用更高效的索引结构
            self._rebuild_index()

        # 更新元数据
        if metadata is not None:
            memory_item.metadata.update(metadata)

        memory_item.updated_at = datetime.now()

        return True

    def delete(self, memory_id: str) -> bool:
        """
        删除记忆.

        根据论文：M_{t+1} = M_t \\ {m_i}

        Args:
            memory_id: 记忆ID

        Returns:
            bool: 是否成功
        """
        if memory_id not in self.memories:
            return False

        del self.memories[memory_id]

        # 重建索引（FAISS不支持直接删除）
        self._rebuild_index()

        return True

    def get(self, memory_id: str) -> Optional[MemoryItem]:
        """根据ID获取记忆."""
        return self.memories.get(memory_id)

    def count(self) -> int:
        """返回记忆总数."""
        return len(self.memories)

    def _match_filters(
        self, metadata: Dict[str, Any], filters: Dict[str, Any]
    ) -> bool:
        """检查元数据是否匹配过滤器."""
        for key, value in filters.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True

    def _rebuild_index(self):
        """重建FAISS索引."""
        self.index.reset()
        for memory_item in self.memories.values():
            if memory_item.embedding:
                embedding = np.array(memory_item.embedding).reshape(1, -1)
                self.index.add(embedding)

    def save(self, path: str):
        """保存索引和记忆到磁盘."""
        # 保存FAISS索引
        faiss.write_index(self.index, f"{path}.index")

        # 保存记忆数据
        memories_data = {
            memory_id: item.to_dict()
            for memory_id, item in self.memories.items()
        }
        with open(f"{path}.json", "w") as f:
            json.dump(memories_data, f, indent=2)

    def load(self, path: str):
        """从磁盘加载索引和记忆."""
        # 加载FAISS索引
        self.index = faiss.read_index(f"{path}.index")

        # 加载记忆数据
        with open(f"{path}.json", "r") as f:
            memories_data = json.load(f)

        self.memories = {
            memory_id: MemoryItem.from_dict(data)
            for memory_id, data in memories_data.items()
        }
