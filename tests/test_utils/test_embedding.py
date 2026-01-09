"""测试嵌入工具"""
import pytest
import numpy as np
from src.agemem.utils.embedding import EmbeddingModel


class TestEmbeddingModel:
    """测试EmbeddingModel"""

    def test_init(self):
        """测试初始化"""
        model = EmbeddingModel()
        assert model.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert model.model is not None

    def test_encode_single_text(self):
        """测试编码单个文本"""
        model = EmbeddingModel()
        text = "This is a test sentence."
        embedding = model.encode(text)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)  # all-MiniLM-L6-v2的维度
        # 检查是否归一化（向量的L2范数应该接近1）
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01

    def test_encode_multiple_texts(self):
        """测试编码多个文本"""
        model = EmbeddingModel()
        texts = ["First sentence.", "Second sentence."]
        embeddings = model.encode(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, 384)
        # 检查每个向量是否归一化
        for emb in embeddings:
            norm = np.linalg.norm(emb)
            assert abs(norm - 1.0) < 0.01

    def test_cosine_similarity(self):
        """测试余弦相似度计算"""
        model = EmbeddingModel()

        # 编码两个相似的文本
        text1 = "I love programming"
        text2 = "I enjoy coding"
        text3 = "The weather is nice today"

        emb1 = model.encode(text1).reshape(1, -1)
        emb2 = model.encode(text2).reshape(1, -1)
        emb3 = model.encode(text3).reshape(1, -1)

        # 计算相似度
        sim_12 = model.cosine_similarity(emb1, emb2)
        sim_13 = model.cosine_similarity(emb1, emb3)

        # text1和text2应该比text1和text3更相似
        assert sim_12[0] > sim_13[0]
        # 相似度应该在[-1, 1]范围内
        assert -1 <= sim_12[0] <= 1
        assert -1 <= sim_13[0] <= 1

    def test_cosine_similarity_batch(self):
        """测试批量计算余弦相似度"""
        model = EmbeddingModel()

        query = "machine learning"
        texts = ["artificial intelligence", "deep learning", "cooking recipes"]

        query_emb = model.encode(query).reshape(1, -1)
        text_embs = model.encode(texts)

        similarities = model.cosine_similarity(query_emb, text_embs)

        assert len(similarities) == 3
        # 前两个应该比第三个更相似
        assert similarities[0] > similarities[2]
        assert similarities[1] > similarities[2]
