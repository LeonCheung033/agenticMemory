"""嵌入工具"""
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """嵌入模型封装"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        初始化嵌入模型

        Args:
            model_name: 模型名称或路径
        """
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def encode(
        self, texts: Union[str, List[str]], convert_to_numpy: bool = True
    ) -> Union[np.ndarray, List[List[float]]]:
        """
        编码文本为嵌入向量

        Args:
            texts: 单个文本或文本列表
            convert_to_numpy: 是否转换为numpy数组

        Returns:
            嵌入向量或向量列表
        """
        embeddings = self.model.encode(
            texts, convert_to_numpy=convert_to_numpy, normalize_embeddings=True  # 归一化以便计算余弦相似度
        )
        return embeddings

    def cosine_similarity(
        self, query_embedding: np.ndarray, memory_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        计算余弦相似度

        根据论文公式：
        sim(q, m_i) = (enc(q)^T * enc(m_i)) / (||enc(q)|| * ||enc(m_i)||)

        由于已经归一化，可以直接计算点积

        Args:
            query_embedding: 查询嵌入向量 (1, dim)
            memory_embeddings: 记忆嵌入向量 (n, dim)

        Returns:
            相似度数组 (n,)
        """
        # 归一化后的向量，余弦相似度 = 点积
        similarities = np.dot(memory_embeddings, query_embedding.T).flatten()
        return similarities
