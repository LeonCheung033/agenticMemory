"""测试三阶段Rollout实现."""
import pytest
from unittest.mock import Mock

from src.agemem.agent.agent import AgeMemAgent
from src.agemem.memory.ltm import LTMManager
from src.agemem.memory.stm import STMManager
from src.agemem.training.workflows.stage_rollouts import (
    Stage1Rollout,
    Stage2Rollout,
    Stage3Rollout,
)


@pytest.fixture
def mock_llm():
    """创建mock LLM."""
    llm = Mock()
    llm.chat = Mock(return_value="Test response")
    return llm


@pytest.fixture
def agent(mock_llm):
    """创建Agent实例."""
    ltm = LTMManager(dimension=384)
    stm = STMManager(max_tokens=8192)
    return AgeMemAgent(ltm_manager=ltm, stm_manager=stm, llm_model=mock_llm, max_turns=20)


class TestStage1Rollout:
    """测试Stage 1 Rollout."""

    def test_initialization(self, agent):
        """测试初始化."""
        rollout = Stage1Rollout(agent=agent, max_turns=4)
        assert rollout.agent == agent
        assert rollout.max_turns == 4
        assert rollout.current_turn == 0
        assert not rollout.initialized

    def test_initialize(self, agent):
        """测试初始化方法."""
        rollout = Stage1Rollout(agent=agent, max_turns=4)
        context_info = "Context line 1\nContext line 2\nContext line 3"
        rollout.initialize(context_info)

        assert rollout.initialized
        assert len(rollout.context_messages) == 3
        assert rollout.context_messages[0] == "Context line 1"

    def test_step_before_initialization(self, agent):
        """测试未初始化时调用step."""
        rollout = Stage1Rollout(agent=agent, max_turns=4)
        with pytest.raises(RuntimeError, match="not initialized"):
            rollout.step(0)

    def test_step_within_bounds(self, agent):
        """测试在范围内执行step."""
        rollout = Stage1Rollout(agent=agent, max_turns=4)
        context_info = "Context line 1\nContext line 2"
        rollout.initialize(context_info)

        # Mock agent.step返回
        agent.step = Mock(return_value=(Mock(has_answer=Mock(return_value=False)), {}))

        result = rollout.step(0)
        assert result is True
        assert rollout.current_turn == 1

    def test_step_out_of_bounds(self, agent):
        """测试超出范围."""
        rollout = Stage1Rollout(agent=agent, max_turns=2)
        context_info = "Context line 1"
        rollout.initialize(context_info)

        result = rollout.step(2)  # 超出max_turns
        assert result is False

    def test_step_no_more_messages(self, agent):
        """测试没有更多消息时."""
        rollout = Stage1Rollout(agent=agent, max_turns=4)
        context_info = "Context line 1"
        rollout.initialize(context_info)

        agent.step = Mock(return_value=(Mock(has_answer=Mock(return_value=False)), {}))

        result = rollout.step(0)
        assert result is True

        result = rollout.step(1)  # 没有更多消息
        assert result is False

    def test_step_with_answer(self, agent):
        """测试Agent给出答案时提前结束."""
        rollout = Stage1Rollout(agent=agent, max_turns=4)
        context_info = "Context line 1"
        rollout.initialize(context_info)

        parsed = Mock(has_answer=Mock(return_value=True))
        agent.step = Mock(return_value=(parsed, {}))

        result = rollout.step(0)
        assert result is False


class TestStage2Rollout:
    """测试Stage 2 Rollout."""

    def test_initialization(self, agent):
        """测试初始化."""
        rollout = Stage2Rollout(agent=agent, max_turns=4)
        assert rollout.agent == agent
        assert rollout.max_turns == 4
        assert not rollout.initialized

    def test_initialize(self, agent):
        """测试初始化方法."""
        rollout = Stage2Rollout(agent=agent, max_turns=4)
        distractors = ["Distractor 1", "Distractor 2", "Distractor 3"]
        rollout.initialize(distractors)

        assert rollout.initialized
        assert rollout.distractors == distractors

    def test_step_before_initialization(self, agent):
        """测试未初始化时调用step."""
        rollout = Stage2Rollout(agent=agent, max_turns=4)
        with pytest.raises(RuntimeError, match="not initialized"):
            rollout.step(0)

    def test_step_within_bounds(self, agent):
        """测试在范围内执行step."""
        rollout = Stage2Rollout(agent=agent, max_turns=4)
        distractors = ["Distractor 1", "Distractor 2"]
        rollout.initialize(distractors)

        agent.step = Mock(return_value=(Mock(has_answer=Mock(return_value=False)), {}))

        result = rollout.step(0)
        assert result is True
        assert rollout.current_turn == 1

    def test_step_out_of_bounds(self, agent):
        """测试超出范围."""
        rollout = Stage2Rollout(agent=agent, max_turns=2)
        distractors = ["Distractor 1"]
        rollout.initialize(distractors)

        result = rollout.step(2)  # 超出max_turns
        assert result is False

    def test_step_no_more_distractors(self, agent):
        """测试没有更多干扰时."""
        rollout = Stage2Rollout(agent=agent, max_turns=4)
        distractors = ["Distractor 1"]
        rollout.initialize(distractors)

        agent.step = Mock(return_value=(Mock(has_answer=Mock(return_value=False)), {}))

        result = rollout.step(0)
        assert result is True

        result = rollout.step(1)  # 没有更多干扰
        assert result is False

    def test_step_with_answer(self, agent):
        """测试Agent给出答案时提前结束."""
        rollout = Stage2Rollout(agent=agent, max_turns=4)
        distractors = ["Distractor 1"]
        rollout.initialize(distractors)

        parsed = Mock(has_answer=Mock(return_value=True))
        agent.step = Mock(return_value=(parsed, {}))

        result = rollout.step(0)
        assert result is False


class TestStage3Rollout:
    """测试Stage 3 Rollout."""

    def test_initialization(self, agent):
        """测试初始化."""
        rollout = Stage3Rollout(agent=agent, max_turns=6)
        assert rollout.agent == agent
        assert rollout.max_turns == 6
        assert not rollout.completed
        assert not rollout.initialized

    def test_initialize(self, agent):
        """测试初始化方法."""
        rollout = Stage3Rollout(agent=agent, max_turns=6)
        query = "What is the answer?"
        rollout.initialize(query)

        assert rollout.initialized
        assert rollout.query == query
        assert not rollout.completed

    def test_step_before_initialization(self, agent):
        """测试未初始化时调用step."""
        rollout = Stage3Rollout(agent=agent, max_turns=6)
        with pytest.raises(RuntimeError, match="not initialized"):
            rollout.step(0)

    def test_step_within_bounds(self, agent):
        """测试在范围内执行step."""
        rollout = Stage3Rollout(agent=agent, max_turns=6)
        query = "What is the answer?"
        rollout.initialize(query)

        agent.step = Mock(return_value=(Mock(has_answer=Mock(return_value=False)), {}))

        result = rollout.step(0)
        assert result is True
        assert rollout.current_turn == 1

    def test_step_out_of_bounds(self, agent):
        """测试超出范围."""
        rollout = Stage3Rollout(agent=agent, max_turns=2)
        query = "What is the answer?"
        rollout.initialize(query)

        agent.step = Mock(return_value=(Mock(has_answer=Mock(return_value=False)), {}))

        result = rollout.step(0)
        assert result is True

        result = rollout.step(2)  # 超出max_turns
        assert result is False

    def test_step_with_answer(self, agent):
        """测试Agent给出答案时提前结束."""
        rollout = Stage3Rollout(agent=agent, max_turns=6)
        query = "What is the answer?"
        rollout.initialize(query)

        parsed = Mock(has_answer=Mock(return_value=True))
        agent.step = Mock(return_value=(parsed, {}))

        result = rollout.step(0)
        assert result is False
        assert rollout.completed

    def test_step_after_completed(self, agent):
        """测试完成后再次调用step."""
        rollout = Stage3Rollout(agent=agent, max_turns=6)
        query = "What is the answer?"
        rollout.initialize(query)

        parsed = Mock(has_answer=Mock(return_value=True))
        agent.step = Mock(return_value=(parsed, {}))

        result = rollout.step(0)
        assert result is False
        assert rollout.completed

        # 再次调用应该返回False
        result = rollout.step(1)
        assert result is False
