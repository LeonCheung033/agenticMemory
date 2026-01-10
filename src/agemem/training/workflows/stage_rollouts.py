"""三阶段Rollout实现."""
from typing import List, Dict, Any
from dataclasses import dataclass

from ...agent.agent import AgeMemAgent


@dataclass
class StageResult:
    """阶段执行结果."""

    completed: bool
    experiences: List[Any]  # 经验列表
    final_state: Dict[str, Any]


class Stage1Rollout:
    """
    Stage 1: LTM Construction.

    根据论文Algorithm 1实现
    """

    def __init__(self, agent: AgeMemAgent, max_turns: int = 4):
        """
        初始化Stage 1 Rollout.

        Args:
            agent: AgeMem Agent实例
            max_turns: 最大轮数
        """
        self.agent = agent
        self.max_turns = max_turns
        self.current_turn = 0
        self.context_messages: List[str] = []  # 来自I_q的消息
        self.initialized = False

    def initialize(self, context_info: str):
        """
        初始化Stage 1.

        Args:
            context_info: 上下文信息（I_q）
        """
        # 解析context_info为消息列表
        # （简化：按行分割，实际应该更智能）
        self.context_messages = [line.strip() for line in context_info.split("\n") if line.strip()]

        # 重置Agent（不重置STM，因为Stage 1是初始阶段）
        task_spec = {
            "query": "",  # Stage 1没有query
            "context_info": context_info,
            "expected_answer": None,
        }
        self.agent.reset(task_spec, reset_stm=False)
        self.initialized = True
        self.current_turn = 0

    def step(self, stage_step: int) -> bool:
        """
        执行Stage 1的一步.

        Args:
            stage_step: 阶段内步数

        Returns:
            bool: 是否继续
        """
        if not self.initialized:
            raise RuntimeError("Stage1Rollout not initialized. Call initialize() first.")

        if stage_step >= self.max_turns:
            return False

        # 获取上下文信息（从I_q中采样）
        if stage_step < len(self.context_messages):
            message = self.context_messages[stage_step]
        else:
            # 如果没有更多消息，结束
            return False

        # 执行Agent step
        parsed, exec_info = self.agent.step(user_input=message)

        # 检查是否提前结束（Agent给出答案）
        if parsed.has_answer():
            return False

        self.current_turn = stage_step + 1
        return True


class Stage2Rollout:
    """
    Stage 2: STM Control under Distractors.

    根据论文Algorithm 2实现
    """

    def __init__(self, agent: AgeMemAgent, max_turns: int = 4):
        """
        初始化Stage 2 Rollout.

        Args:
            agent: AgeMem Agent实例
            max_turns: 最大轮数
        """
        self.agent = agent
        self.max_turns = max_turns
        self.distractors: List[str] = []
        self.initialized = False
        self.current_turn = 0

    def initialize(self, distractors: List[str]):
        """
        初始化Stage 2.

        Args:
            distractors: 干扰信息列表
        """
        self.distractors = distractors

        # 重置Agent（重置STM，保留LTM）
        task_spec = {
            "query": "",  # Stage 2没有query
            "context_info": "",
            "expected_answer": None,
        }
        self.agent.reset(task_spec, reset_stm=True)  # 关键：重置STM
        self.initialized = True
        self.current_turn = 0

    def step(self, stage_step: int) -> bool:
        """
        执行Stage 2的一步.

        Args:
            stage_step: 阶段内步数

        Returns:
            bool: 是否继续
        """
        if not self.initialized:
            raise RuntimeError("Stage2Rollout not initialized. Call initialize() first.")

        if stage_step >= self.max_turns:
            return False

        # 注入干扰信息
        if stage_step < len(self.distractors):
            distractor = self.distractors[stage_step]
        else:
            # 如果没有更多干扰，结束
            return False

        # 执行Agent step
        parsed, exec_info = self.agent.step(user_input=distractor)

        # 检查是否提前结束
        if parsed.has_answer():
            return False

        self.current_turn = stage_step + 1
        return True


class Stage3Rollout:
    """
    Stage 3: Integrated Reasoning and Memory Coordination.

    根据论文Algorithm 3实现
    """

    def __init__(self, agent: AgeMemAgent, max_turns: int = 6):
        """
        初始化Stage 3 Rollout.

        Args:
            agent: AgeMem Agent实例
            max_turns: 最大轮数
        """
        self.agent = agent
        self.max_turns = max_turns
        self.query = ""
        self.completed = False
        self.initialized = False
        self.current_turn = 0

    def initialize(self, query: str):
        """
        初始化Stage 3.

        Args:
            query: 查询问题
        """
        self.query = query
        self.completed = False

        # 重置Agent（不重置STM，因为STM从Stage 2延续）
        task_spec = {
            "query": query,
            "context_info": "",
            "expected_answer": None,
        }
        self.agent.reset(task_spec, reset_stm=False)  # 关键：不重置STM
        self.initialized = True
        self.current_turn = 0

    def step(self, stage_step: int) -> bool:
        """
        执行Stage 3的一步.

        Args:
            stage_step: 阶段内步数

        Returns:
            bool: 是否继续
        """
        if not self.initialized:
            raise RuntimeError("Stage3Rollout not initialized. Call initialize() first.")

        if self.completed or stage_step >= self.max_turns:
            return False

        # 第一步：query已经在reset时添加，直接执行
        # 后续步骤：继续执行
        parsed, exec_info = self.agent.step()

        # 检查是否获得答案
        if parsed.has_answer():
            self.completed = True
            return False

        self.current_turn = stage_step + 1
        return True
