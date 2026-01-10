"""AgeMem三阶段Workflow实现."""
from typing import List, Dict, Any, Optional, Tuple

# 尝试导入Trinity-RFT，如果不存在则使用类型存根
try:
    from trinity.common.workflows.step_wise_workflow import RewardPropagationWorkflow
    from trinity.common.workflows.workflow import Task
    from trinity.common.models.model import ModelWrapper
    from trinity.common.experience import Experience
    from trinity.common.workflows import WORKFLOWS

    TRINITY_AVAILABLE = True
except ImportError:
    # Trinity-RFT未安装，创建兼容的基类和类型存根
    TRINITY_AVAILABLE = False

    class RewardPropagationWorkflow:
        """兼容基类（当Trinity-RFT未安装时）."""

        def __init__(self, *, task, model, auxiliary_models=None, use_openai_client=True):
            """初始化兼容基类."""
            self.task = task
            self.model = model
            self.auxiliary_models = auxiliary_models or []
            self.use_openai_client = use_openai_client

        def step(self, step_num: int) -> bool:
            """执行一步."""
            raise NotImplementedError

        def reward(self, experiences: List[Any]) -> float:
            """计算奖励."""
            raise NotImplementedError

    class Task:
        """Task类型存根."""

        def __init__(self):
            """初始化Task."""
            self.workflow_args = {}
            self.reward_fn_args = {}
            self.raw_task = {}

    class ModelWrapper:
        """ModelWrapper类型存根."""

        pass

    class Experience:
        """Experience类型存根."""

        def __init__(self):
            """初始化Experience."""
            self.eid = type("EID", (), {"step": 0})()

    # 创建一个简单的注册器
    class WorkflowRegistry:
        """简单的workflow注册器."""

        _registry = {}

        @classmethod
        def register_module(cls, name: str):
            """注册装饰器."""

            def decorator(cls_obj):
                cls._registry[name] = cls_obj
                return cls_obj

            return decorator

    WORKFLOWS = WorkflowRegistry()

from ...agent.agent import AgeMemAgent
from ...memory.ltm import LTMManager
from ...memory.stm import STMManager
from ...training.reward import AgeMemRewardFunction, RewardConfig
from .stage_rollouts import Stage1Rollout, Stage2Rollout, Stage3Rollout


@WORKFLOWS.register_module(name="agemem_workflow")
class AgeMemWorkflow(RewardPropagationWorkflow):
    """
    AgeMem三阶段Workflow.

    实现论文Appendix B.3的完整训练流程：
    - Stage 1: LTM Construction
    - Stage 2: STM Control under Distractors
    - Stage 3: Integrated Reasoning and Memory Coordination
    """

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[ModelWrapper]] = None,
        use_openai_client: bool = True,
    ):
        """
        初始化AgeMem Workflow.

        Args:
            task: Trinity Task对象（或兼容对象）
            model: Rollout模型
            auxiliary_models: 辅助模型（用于judge等）
            use_openai_client: 是否使用OpenAI client
        """
        super().__init__(
            task=task,
            model=model,
            auxiliary_models=auxiliary_models,
            use_openai_client=use_openai_client,
        )

        # 从task中提取配置
        workflow_args = task.workflow_args if hasattr(task, "workflow_args") else {}
        self.max_stage1_turns = workflow_args.get("max_stage1_turns", 4)
        self.max_stage2_turns = workflow_args.get("max_stage2_turns", 4)
        self.max_stage3_turns = workflow_args.get("max_stage3_turns", 6)

        # 初始化记忆管理器
        self.ltm = LTMManager(
            index_path=workflow_args.get("ltm_index_path"),
            dimension=workflow_args.get("embedding_dim", 384),
        )
        self.stm = STMManager(
            max_tokens=workflow_args.get("max_tokens", 8192),
        )

        # 确定LLM模型
        if use_openai_client and hasattr(self, "client"):
            llm_model = self.client
        else:
            llm_model = model

        # 初始化Agent
        self.agent = AgeMemAgent(
            ltm_manager=self.ltm,
            stm_manager=self.stm,
            llm_model=llm_model,
            max_turns=max(self.max_stage1_turns, self.max_stage2_turns, self.max_stage3_turns),
        )

        # 初始化奖励函数
        reward_fn_args = task.reward_fn_args if hasattr(task, "reward_fn_args") else {}
        reward_config = None
        if reward_fn_args:
            reward_config = RewardConfig(
                w_task=reward_fn_args.get("w_task", 1.0 / 3.0),
                w_context=reward_fn_args.get("w_context", 1.0 / 3.0),
                w_memory=reward_fn_args.get("w_memory", 1.0 / 3.0),
                max_tokens=reward_fn_args.get("max_tokens", 8192),
                max_rounds=reward_fn_args.get("max_rounds", 20),
                penalty_no_answer=reward_fn_args.get("penalty_no_answer", -1.0),
                penalty_rounds=reward_fn_args.get("penalty_rounds", -1.0),
                penalty_overflow=reward_fn_args.get("penalty_overflow", -0.5),
            )

        self.reward_fn = AgeMemRewardFunction(
            config=reward_config,
            llm_judge=auxiliary_models[0]
            if auxiliary_models and len(auxiliary_models) > 0
            else None,
            llm_relevance_scorer=auxiliary_models[1]
            if auxiliary_models and len(auxiliary_models) > 1
            else None,
        )

        # 阶段rollout实现
        self.stage1_rollout = Stage1Rollout(agent=self.agent, max_turns=self.max_stage1_turns)
        self.stage2_rollout = Stage2Rollout(agent=self.agent, max_turns=self.max_stage2_turns)
        self.stage3_rollout = Stage3Rollout(agent=self.agent, max_turns=self.max_stage3_turns)

        # 任务数据
        self.raw_task = task.raw_task if hasattr(task, "raw_task") else {}
        self.query = self.raw_task.get("query", "")
        self.context_info = self.raw_task.get("context_info", "")
        self.expected_answer = self.raw_task.get("expected_answer")
        self.distractors = self.raw_task.get("distractors", [])

        # 初始化阶段
        self._initialize_stages()

        # 执行轨迹（用于奖励计算）
        self.execution_trace: List[Dict[str, Any]] = []

    def _initialize_stages(self):
        """初始化三个阶段."""
        # Stage 1: 使用context_info初始化
        if self.context_info:
            self.stage1_rollout.initialize(self.context_info)

        # Stage 2: 使用distractors初始化
        if self.distractors:
            self.stage2_rollout.initialize(self.distractors)

        # Stage 3: 使用query初始化
        if self.query:
            self.stage3_rollout.initialize(self.query)

    @property
    def max_step_num(self) -> int:
        """返回最大步数（三阶段总和）."""
        return self.max_stage1_turns + self.max_stage2_turns + self.max_stage3_turns

    def step(self, step_num: int) -> bool:
        """
        执行一步（对应论文算法中的一个turn）.

        Args:
            step_num: 当前步数

        Returns:
            bool: 是否继续执行
        """
        # 确定当前阶段
        stage, stage_step = self._get_stage_and_step(step_num)

        # 记录执行信息
        step_info = {
            "step": step_num,
            "stage": stage,
            "stage_step": stage_step,
        }

        # 执行对应阶段的step
        if stage == 1:
            should_continue = self.stage1_rollout.step(stage_step)
            step_info["stage_name"] = "LTM_Construction"
        elif stage == 2:
            should_continue = self.stage2_rollout.step(stage_step)
            step_info["stage_name"] = "STM_Control"
        elif stage == 3:
            should_continue = self.stage3_rollout.step(stage_step)
            step_info["stage_name"] = "Integrated_Reasoning"
        else:
            return False

        # 记录执行轨迹
        step_info["should_continue"] = should_continue
        self.execution_trace.append(step_info)

        return should_continue

    def _get_stage_and_step(self, step_num: int) -> Tuple[int, int]:
        """
        根据总步数确定当前阶段和阶段内步数.

        Args:
            step_num: 总步数

        Returns:
            (stage, stage_step): 阶段号和阶段内步数
        """
        if step_num < self.max_stage1_turns:
            return (1, step_num)
        elif step_num < self.max_stage1_turns + self.max_stage2_turns:
            return (2, step_num - self.max_stage1_turns)
        else:
            return (3, step_num - self.max_stage1_turns - self.max_stage2_turns)

    def reward(self, experiences: List[Experience]) -> float:
        """
        计算轨迹级奖励（论文中的R(τ)）.

        Args:
            experiences: 所有步骤的经验列表

        Returns:
            float: 轨迹级奖励
        """
        # 获取最终Agent状态
        task_spec = {
            "query": self.query,
            "context_info": self.context_info,
            "expected_answer": self.expected_answer,
        }
        agent_state = self.agent.get_state(task_spec)

        # 构建执行轨迹（从experiences中提取，或使用内部轨迹）
        execution_trace = self._build_execution_trace(experiences)

        # 提取轨迹信息
        trajectory_info = self.reward_fn.extract_trajectory_info(
            agent_state=agent_state,
            execution_trace=execution_trace,
            query=self.query,
            expected_answer=self.expected_answer,
        )

        # 计算奖励
        reward_dict = self.reward_fn.compute(
            trajectory_info=trajectory_info,
            agent_state=agent_state,
        )

        return reward_dict["total_reward"]

    def _build_execution_trace(self, experiences: List[Experience]) -> List[Dict[str, Any]]:
        """
        从experiences构建执行轨迹.

        Args:
            experiences: Trinity Experience列表（或兼容对象）

        Returns:
            List[Dict]: 执行轨迹
        """
        # 如果experiences为空，使用内部轨迹
        if not experiences:
            return self.execution_trace

        # 按step分组
        steps: Dict[int, List[Any]] = {}
        for exp in experiences:
            if hasattr(exp, "eid") and hasattr(exp.eid, "step"):
                step = exp.eid.step
            elif isinstance(exp, dict):
                step = exp.get("step", 0)
            else:
                step = 0

            if step not in steps:
                steps[step] = []
            steps[step].append(exp)

        # 构建轨迹
        trace = []
        for step_num in sorted(steps.keys()):
            step_exps = steps[step_num]
            # 从experience中提取信息
            trace.append(
                {
                    "step": step_num,
                    "experiences": step_exps,
                    "execution": {
                        "tool_results": self._extract_tool_results(step_exps),
                    },
                }
            )

        return trace

    def _extract_tool_results(self, experiences: List[Any]) -> List[Dict[str, Any]]:
        """
        从experiences中提取工具调用结果.

        Args:
            experiences: 经验列表

        Returns:
            List[Dict]: 工具结果列表
        """
        tool_results = []
        for exp in experiences:
            # 尝试从experience中提取工具调用信息
            if hasattr(exp, "tool_calls"):
                tool_results.extend(exp.tool_calls)
            elif isinstance(exp, dict):
                if "tool_calls" in exp:
                    tool_results.extend(exp["tool_calls"])
                elif "execution" in exp and "tool_results" in exp["execution"]:
                    tool_results.extend(exp["execution"]["tool_results"])

        return tool_results
