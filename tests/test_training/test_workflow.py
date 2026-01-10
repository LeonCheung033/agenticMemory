"""测试AgeMem Workflow实现."""
import pytest
from unittest.mock import Mock

from src.agemem.training.workflows.agemem_workflow import AgeMemWorkflow


@pytest.fixture
def mock_task():
    """创建mock Task."""
    task = Mock()
    task.workflow_args = {
        "max_stage1_turns": 4,
        "max_stage2_turns": 4,
        "max_stage3_turns": 6,
        "max_tokens": 8192,
        "embedding_dim": 384,
        "ltm_index_path": None,
    }
    task.reward_fn_args = {
        "w_task": 0.333,
        "w_context": 0.333,
        "w_memory": 0.333,
        "max_tokens": 8192,
        "max_rounds": 20,
        "penalty_no_answer": -1.0,
        "penalty_rounds": -1.0,
        "penalty_overflow": -0.5,
    }
    task.raw_task = {
        "query": "What is the answer?",
        "context_info": "Context line 1\nContext line 2",
        "expected_answer": "The answer is 42",
        "distractors": ["Distractor 1", "Distractor 2"],
    }
    return task


@pytest.fixture
def mock_model():
    """创建mock Model."""
    model = Mock()
    model.chat = Mock(return_value="Test response")
    return model


class TestAgeMemWorkflow:
    """测试AgeMemWorkflow."""

    def test_initialization(self, mock_task, mock_model):
        """测试初始化."""
        workflow = AgeMemWorkflow(
            task=mock_task,
            model=mock_model,
            auxiliary_models=None,
            use_openai_client=False,
        )

        assert workflow.max_stage1_turns == 4
        assert workflow.max_stage2_turns == 4
        assert workflow.max_stage3_turns == 6
        assert workflow.query == "What is the answer?"
        assert len(workflow.distractors) == 2

    def test_max_step_num(self, mock_task, mock_model):
        """测试最大步数计算."""
        workflow = AgeMemWorkflow(
            task=mock_task,
            model=mock_model,
            auxiliary_models=None,
            use_openai_client=False,
        )

        assert workflow.max_step_num == 14  # 4 + 4 + 6

    def test_get_stage_and_step(self, mock_task, mock_model):
        """测试阶段和步数计算."""
        workflow = AgeMemWorkflow(
            task=mock_task,
            model=mock_model,
            auxiliary_models=None,
            use_openai_client=False,
        )

        # Stage 1
        stage, stage_step = workflow._get_stage_and_step(0)
        assert stage == 1
        assert stage_step == 0

        stage, stage_step = workflow._get_stage_and_step(3)
        assert stage == 1
        assert stage_step == 3

        # Stage 2
        stage, stage_step = workflow._get_stage_and_step(4)
        assert stage == 2
        assert stage_step == 0

        stage, stage_step = workflow._get_stage_and_step(7)
        assert stage == 2
        assert stage_step == 3

        # Stage 3
        stage, stage_step = workflow._get_stage_and_step(8)
        assert stage == 3
        assert stage_step == 0

        stage, stage_step = workflow._get_stage_and_step(13)
        assert stage == 3
        assert stage_step == 5

    def test_step_stage1(self, mock_task, mock_model):
        """测试Stage 1执行."""
        workflow = AgeMemWorkflow(
            task=mock_task,
            model=mock_model,
            auxiliary_models=None,
            use_openai_client=False,
        )

        # Mock stage1_rollout.step
        workflow.stage1_rollout.step = Mock(return_value=True)

        result = workflow.step(0)
        assert result is True
        assert len(workflow.execution_trace) == 1
        assert workflow.execution_trace[0]["stage"] == 1

    def test_step_stage2(self, mock_task, mock_model):
        """测试Stage 2执行."""
        workflow = AgeMemWorkflow(
            task=mock_task,
            model=mock_model,
            auxiliary_models=None,
            use_openai_client=False,
        )

        # Mock stage2_rollout.step
        workflow.stage2_rollout.step = Mock(return_value=True)

        result = workflow.step(4)  # Stage 2第一步
        assert result is True
        assert len(workflow.execution_trace) == 1
        assert workflow.execution_trace[0]["stage"] == 2

    def test_step_stage3(self, mock_task, mock_model):
        """测试Stage 3执行."""
        workflow = AgeMemWorkflow(
            task=mock_task,
            model=mock_model,
            auxiliary_models=None,
            use_openai_client=False,
        )

        # Mock stage3_rollout.step
        workflow.stage3_rollout.step = Mock(return_value=True)

        result = workflow.step(8)  # Stage 3第一步
        assert result is True
        assert len(workflow.execution_trace) == 1
        assert workflow.execution_trace[0]["stage"] == 3

    def test_step_invalid_stage(self, mock_task, mock_model):
        """测试无效阶段."""
        workflow = AgeMemWorkflow(
            task=mock_task,
            model=mock_model,
            auxiliary_models=None,
            use_openai_client=False,
        )

        result = workflow.step(100)  # 超出所有阶段
        assert result is False

    def test_build_execution_trace_empty(self, mock_task, mock_model):
        """测试构建空执行轨迹."""
        workflow = AgeMemWorkflow(
            task=mock_task,
            model=mock_model,
            auxiliary_models=None,
            use_openai_client=False,
        )

        trace = workflow._build_execution_trace([])
        assert len(trace) > 0  # 应该使用内部轨迹

    def test_build_execution_trace_with_experiences(self, mock_task, mock_model):
        """测试从experiences构建执行轨迹."""
        workflow = AgeMemWorkflow(
            task=mock_task,
            model=mock_model,
            auxiliary_models=None,
            use_openai_client=False,
        )

        # 创建mock experiences
        exp1 = Mock()
        exp1.eid = Mock()
        exp1.eid.step = 0

        exp2 = Mock()
        exp2.eid = Mock()
        exp2.eid.step = 1

        experiences = [exp1, exp2]
        trace = workflow._build_execution_trace(experiences)

        assert len(trace) == 2
        assert trace[0]["step"] == 0
        assert trace[1]["step"] == 1

    def test_reward(self, mock_task, mock_model):
        """测试奖励计算."""
        workflow = AgeMemWorkflow(
            task=mock_task,
            model=mock_model,
            auxiliary_models=None,
            use_openai_client=False,
        )

        # Mock reward function
        workflow.reward_fn.compute = Mock(
            return_value={
                "total_reward": 0.85,
                "task_reward": 0.9,
                "context_reward": 0.8,
                "memory_reward": 0.85,
                "penalty": 0.0,
            }
        )
        workflow.reward_fn.extract_trajectory_info = Mock(return_value=Mock())

        # Mock agent state
        workflow.agent.get_state = Mock(return_value=Mock())

        experiences = []
        reward = workflow.reward(experiences)

        assert reward == 0.85
        workflow.reward_fn.compute.assert_called_once()
