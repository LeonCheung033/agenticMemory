"""测试奖励函数."""
import pytest
from unittest.mock import Mock
from src.agemem.training.reward import (
    AgeMemRewardFunction,
    RewardConfig,
    TrajectoryInfo,
)
from src.agemem.agent.agent import AgentState
from src.agemem.memory.ltm import LTMManager
from src.agemem.memory.stm import STMManager


class TestRewardConfig:
    """测试RewardConfig."""

    def test_default_config(self):
        """测试默认配置."""
        config = RewardConfig()
        assert config.w_task == pytest.approx(1.0 / 3.0)
        assert config.w_context == pytest.approx(1.0 / 3.0)
        assert config.w_memory == pytest.approx(1.0 / 3.0)
        assert config.max_tokens == 8192
        assert config.max_rounds == 20

    def test_custom_config(self):
        """测试自定义配置."""
        config = RewardConfig(
            w_task=0.5, w_context=0.3, w_memory=0.2, max_tokens=4096, max_rounds=10
        )
        assert config.w_task == 0.5
        assert config.w_context == 0.3
        assert config.w_memory == 0.2
        assert config.max_tokens == 4096
        assert config.max_rounds == 10


class TestTrajectoryInfo:
    """测试TrajectoryInfo."""

    def test_init(self):
        """测试初始化."""
        info = TrajectoryInfo(query="test query")
        assert info.query == "test query"
        assert info.has_answer is False
        assert info.total_memories_added == 0
        assert len(info.retrieved_memories) == 0
        assert len(info.tool_call_history) == 0

    def test_init_with_values(self):
        """测试带值初始化."""
        info = TrajectoryInfo(
            query="test",
            predicted_answer="answer",
            has_answer=True,
            total_memories_added=5,
            high_quality_memories=3,
        )
        assert info.has_answer is True
        assert info.predicted_answer == "answer"
        assert info.total_memories_added == 5
        assert info.high_quality_memories == 3


class TestAgeMemRewardFunction:
    """测试AgeMemRewardFunction."""

    def test_init(self):
        """测试初始化."""
        reward_fn = AgeMemRewardFunction()
        assert reward_fn.config is not None
        assert reward_fn.llm_judge is None
        assert reward_fn.llm_relevance_scorer is None

    def test_init_with_config(self):
        """测试带配置初始化."""
        config = RewardConfig(w_task=0.5)
        reward_fn = AgeMemRewardFunction(config=config)
        assert reward_fn.config.w_task == 0.5

    def test_compute_task_reward_no_answer(self):
        """测试无答案的任务奖励."""
        reward_fn = AgeMemRewardFunction()
        info = TrajectoryInfo(query="test", has_answer=False)

        task_reward = reward_fn._compute_task_reward(info)
        assert task_reward == reward_fn.config.penalty_no_answer

    def test_compute_task_reward_with_answer(self):
        """测试有答案的任务奖励."""
        reward_fn = AgeMemRewardFunction()
        info = TrajectoryInfo(
            query="test", predicted_answer="answer", expected_answer="answer", has_answer=True
        )

        task_reward = reward_fn._compute_task_reward(info)
        # 没有judge时，使用简单匹配
        assert task_reward == 1.0

    def test_get_judge_score_without_judge(self):
        """测试没有judge时的评分."""
        reward_fn = AgeMemRewardFunction()
        score = reward_fn._get_judge_score("answer", "answer")
        assert score == 1.0

        score = reward_fn._get_judge_score("wrong", "answer")
        assert score == 0.0

    def test_get_judge_score_with_judge(self):
        """测试有judge时的评分."""
        mock_judge = Mock()
        mock_judge.chat = Mock(return_value="0.85")
        reward_fn = AgeMemRewardFunction(llm_judge=mock_judge)

        score = reward_fn._get_judge_score("answer", "expected")
        assert score == 0.85
        mock_judge.chat.assert_called_once()

    def test_parse_judge_score(self):
        """测试解析judge分数."""
        reward_fn = AgeMemRewardFunction()
        assert reward_fn._parse_judge_score("0.85") == 0.85
        assert reward_fn._parse_judge_score("The score is 0.75") == 0.75
        assert reward_fn._parse_judge_score("0.9") == 0.9
        assert reward_fn._parse_judge_score("invalid") == 0.0

    def test_compute_compression_reward(self):
        """测试压缩效率奖励."""
        config = RewardConfig(max_tokens=1000)
        reward_fn = AgeMemRewardFunction(config=config)

        # 使用500 tokens，应该得到0.5
        info = TrajectoryInfo(query="test", final_token_count=500)
        reward = reward_fn._compute_compression_reward(info)
        assert reward == pytest.approx(0.5)

        # 使用1000 tokens，应该得到0.0
        info.final_token_count = 1000
        reward = reward_fn._compute_compression_reward(info)
        assert reward == pytest.approx(0.0)

        # 使用0 tokens，应该得到1.0
        info.final_token_count = 0
        reward = reward_fn._compute_compression_reward(info)
        assert reward == pytest.approx(1.0)

    def test_compute_preventive_reward(self):
        """测试预防性管理奖励."""
        reward_fn = AgeMemRewardFunction()

        info = TrajectoryInfo(query="test", tool_calls_before_overflow=True)
        reward = reward_fn._compute_preventive_reward(info)
        assert reward == 1.0

        info.tool_calls_before_overflow = False
        reward = reward_fn._compute_preventive_reward(info)
        assert reward == 0.0

    def test_compute_preservation_reward(self):
        """测试信息保留奖励."""
        reward_fn = AgeMemRewardFunction()

        info = TrajectoryInfo(query="test", key_tokens_preserved=True)
        reward = reward_fn._compute_preservation_reward(info)
        assert reward == 1.0

        info.key_tokens_preserved = False
        reward = reward_fn._compute_preservation_reward(info)
        assert reward == 0.0

    def test_compute_storage_reward(self):
        """测试存储质量奖励."""
        reward_fn = AgeMemRewardFunction()

        # 没有记忆
        info = TrajectoryInfo(query="test", total_memories_added=0)
        reward = reward_fn._compute_storage_reward(info)
        assert reward == 0.0

        # 5个记忆，3个高质量
        info = TrajectoryInfo(query="test", total_memories_added=5, high_quality_memories=3)
        reward = reward_fn._compute_storage_reward(info)
        assert reward == pytest.approx(3.0 / 5.0)

        # 全部高质量
        info.high_quality_memories = 5
        reward = reward_fn._compute_storage_reward(info)
        assert reward == 1.0

    def test_compute_maintenance_reward(self):
        """测试维护奖励."""
        reward_fn = AgeMemRewardFunction()

        info = TrajectoryInfo(query="test", update_or_delete_performed=True)
        reward = reward_fn._compute_maintenance_reward(info)
        assert reward == 1.0

        info.update_or_delete_performed = False
        reward = reward_fn._compute_maintenance_reward(info)
        assert reward == 0.0

    def test_compute_relevance_reward(self):
        """测试语义相关性奖励."""
        reward_fn = AgeMemRewardFunction()

        # 没有检索记忆
        info = TrajectoryInfo(query="test", retrieved_memories=[])
        reward = reward_fn._compute_relevance_reward(info)
        assert reward == 0.0

        # 有检索记忆但没有scorer
        info.retrieved_memories = [{"content": "memory"}]
        reward = reward_fn._compute_relevance_reward(info)
        assert reward == 0.5  # 默认值

    def test_compute_penalty(self):
        """测试惩罚项计算."""
        config = RewardConfig(max_tokens=1000, max_rounds=10)
        reward_fn = AgeMemRewardFunction(config=config)

        # 无违规
        info = TrajectoryInfo(query="test", num_rounds=5, final_token_count=500)
        penalty = reward_fn._compute_penalty(info)
        assert penalty == 0.0

        # 轮数超限
        info.num_rounds = 15
        penalty = reward_fn._compute_penalty(info)
        assert penalty == config.penalty_rounds

        # Token超限
        info.num_rounds = 5
        info.final_token_count = 1500
        penalty = reward_fn._compute_penalty(info)
        assert penalty == config.penalty_overflow

        # 两者都超限
        info.num_rounds = 15
        info.final_token_count = 1500
        penalty = reward_fn._compute_penalty(info)
        assert penalty == config.penalty_rounds + config.penalty_overflow

    def test_compute_context_reward(self):
        """测试上下文管理奖励."""
        reward_fn = AgeMemRewardFunction()
        info = TrajectoryInfo(
            query="test",
            final_token_count=4000,  # 在8192的50%左右
            tool_calls_before_overflow=True,
            key_tokens_preserved=True,
        )

        reward = reward_fn._compute_context_reward(info)
        # 应该包含三个组件的加权和
        assert reward > 0.0
        assert reward <= 1.0

    def test_compute_memory_reward(self):
        """测试记忆管理奖励."""
        reward_fn = AgeMemRewardFunction()
        info = TrajectoryInfo(
            query="test",
            total_memories_added=5,
            high_quality_memories=3,
            update_or_delete_performed=True,
            retrieved_memories=[{"content": "memory"}],
        )

        reward = reward_fn._compute_memory_reward(info)
        # 应该包含三个组件的加权和
        assert reward > 0.0
        assert reward <= 1.0

    def test_compute_full_reward(self):
        """测试完整奖励计算."""
        reward_fn = AgeMemRewardFunction()
        info = TrajectoryInfo(
            query="test",
            predicted_answer="answer",
            expected_answer="answer",
            has_answer=True,
            final_token_count=4000,
            tool_calls_before_overflow=True,
            key_tokens_preserved=True,
            total_memories_added=5,
            high_quality_memories=3,
            update_or_delete_performed=True,
            retrieved_memories=[{"content": "memory"}],
            num_rounds=5,
        )

        result = reward_fn.compute(info)

        assert "total_reward" in result
        assert "task_reward" in result
        assert "context_reward" in result
        assert "memory_reward" in result
        assert "penalty" in result
        assert "components" in result

        # 检查组件
        assert "task" in result["components"]
        assert "context" in result["components"]
        assert "memory" in result["components"]

    def test_extract_trajectory_info(self):
        """测试轨迹信息提取."""
        reward_fn = AgeMemRewardFunction()

        # 创建Agent状态
        ltm = LTMManager()
        stm = STMManager()
        stm.add_message("user", "test query")
        agent_state = AgentState(context=stm, memory=ltm, task_spec={"query": "test query"})

        # 创建执行轨迹
        from src.agemem.agent.parser import ParsedResponse

        execution_trace = [
            {
                "parsed": ParsedResponse(
                    reasoning="test",
                    tool_calls=[{"name": "Add_memory", "arguments": {"content": "test memory"}}],
                ),
                "execution": {
                    "tool_results": [
                        {
                            "tool_name": "Add_memory",
                            "success": True,
                            "result": {"message": "Memory stored successfully with ID: test-id"},
                        }
                    ]
                },
            },
            {
                "parsed": ParsedResponse(reasoning="test", answer="final answer"),
                "execution": {"has_answer": True},
            },
        ]

        info = reward_fn.extract_trajectory_info(
            agent_state, execution_trace, query="test query", expected_answer="expected"
        )

        assert info.query == "test query"
        assert info.has_answer is True
        assert info.predicted_answer == "final answer"
        assert info.total_memories_added == 1
        assert info.num_rounds == 2

    def test_is_high_quality_memory(self):
        """测试高质量记忆判断."""
        reward_fn = AgeMemRewardFunction()

        # 长消息（应该被认为是高质量）
        result = {
            "result": {
                "message": "This is a long memory content that should be considered high quality"
            }
        }
        assert reward_fn._is_high_quality_memory(result) is True

        # 短消息（不应该被认为是高质量）
        result = {"result": {"message": "short"}}
        assert reward_fn._is_high_quality_memory(result) is False

    def test_check_key_tokens_preserved(self):
        """测试关键token保留检查."""
        reward_fn = AgeMemRewardFunction()

        # 创建包含query关键词的context
        stm = STMManager()
        stm.add_message("user", "test query about machine learning")
        stm.add_message("assistant", "Machine learning is a subset of AI")

        query = "test query about machine learning"
        preserved = reward_fn._check_key_tokens_preserved(query, stm)
        assert preserved is True

        # 创建不包含query关键词的context
        stm2 = STMManager()
        stm2.add_message("user", "completely different topic")
        preserved = reward_fn._check_key_tokens_preserved(query, stm2)
        assert preserved is False
