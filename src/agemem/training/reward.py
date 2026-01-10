"""奖励函数实现."""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import re

from ..memory.stm import STMManager
from ..agent.agent import AgentState


@dataclass
class RewardConfig:
    """奖励函数配置."""

    # 权重
    w_task: float = 1.0 / 3.0
    w_context: float = 1.0 / 3.0
    w_memory: float = 1.0 / 3.0

    # 上下文管理子权重
    alpha_compression: float = 1.0 / 3.0
    alpha_preventive: float = 1.0 / 3.0
    alpha_preservation: float = 1.0 / 3.0

    # 记忆管理子权重
    beta_storage: float = 1.0 / 3.0
    beta_maintenance: float = 1.0 / 3.0
    beta_relevance: float = 1.0 / 3.0

    # 惩罚系数
    penalty_no_answer: float = -1.0
    penalty_rounds: float = -1.0
    penalty_overflow: float = -0.5

    # 限制
    max_tokens: int = 8192
    max_rounds: int = 20


@dataclass
class TrajectoryInfo:
    """轨迹信息（用于奖励计算）."""

    # 任务信息
    query: str
    expected_answer: Optional[str] = None
    predicted_answer: Optional[str] = None
    has_answer: bool = False

    # 上下文信息
    final_token_count: int = 0
    tool_calls_before_overflow: bool = False
    key_tokens_preserved: bool = False

    # 记忆信息
    total_memories_added: int = 0
    high_quality_memories: int = 0
    update_or_delete_performed: bool = False
    retrieved_memories: List[Any] = field(default_factory=list)

    # 交互信息
    num_rounds: int = 0

    # 工具调用历史
    tool_call_history: List[Dict[str, Any]] = field(default_factory=list)


class AgeMemRewardFunction:
    """
    AgeMem奖励函数.

    实现论文Appendix B.2的完整奖励公式
    """

    def __init__(
        self,
        config: Optional[RewardConfig] = None,
        llm_judge: Optional[Any] = None,
        llm_relevance_scorer: Optional[Any] = None,
    ):
        """
        初始化奖励函数.

        Args:
            config: 奖励配置
            llm_judge: LLM judge模型（用于任务评分）
            llm_relevance_scorer: LLM相关性评分模型
        """
        self.config = config or RewardConfig()
        self.llm_judge = llm_judge
        self.llm_relevance_scorer = llm_relevance_scorer

    def compute(
        self,
        trajectory_info: TrajectoryInfo,
        agent_state: Optional[AgentState] = None,
    ) -> Dict[str, Any]:
        """
        计算轨迹级奖励.

        Args:
            trajectory_info: 轨迹信息
            agent_state: Agent状态（可选，用于额外信息）

        Returns:
            Dict: {
                "total_reward": float,
                "task_reward": float,
                "context_reward": float,
                "memory_reward": float,
                "penalty": float,
                "components": {...}  # 详细组件
            }
        """
        # 计算各组件奖励
        task_reward = self._compute_task_reward(trajectory_info)
        context_reward = self._compute_context_reward(trajectory_info)
        memory_reward = self._compute_memory_reward(trajectory_info, agent_state)
        penalty = self._compute_penalty(trajectory_info)

        # 计算总奖励
        total_reward = (
            self.config.w_task * task_reward
            + self.config.w_context * context_reward
            + self.config.w_memory * memory_reward
            + penalty
        )

        return {
            "total_reward": total_reward,
            "task_reward": task_reward,
            "context_reward": context_reward,
            "memory_reward": memory_reward,
            "penalty": penalty,
            "components": {
                "task": {
                    "judge_score": (
                        self._get_judge_score(
                            trajectory_info.predicted_answer,
                            trajectory_info.expected_answer,
                        )
                        if trajectory_info.has_answer
                        else None
                    ),
                    "has_answer": trajectory_info.has_answer,
                },
                "context": {
                    "compression": self._compute_compression_reward(trajectory_info),
                    "preventive": self._compute_preventive_reward(trajectory_info),
                    "preservation": self._compute_preservation_reward(trajectory_info),
                },
                "memory": {
                    "storage": self._compute_storage_reward(trajectory_info),
                    "maintenance": self._compute_maintenance_reward(trajectory_info),
                    "relevance": self._compute_relevance_reward(trajectory_info, agent_state),
                },
            },
        }

    def _compute_task_reward(self, info: TrajectoryInfo) -> float:
        """
        计算任务完成奖励.

        公式:R_task = S_judge(A_pred, A_q) if has_answer else P_no_answer
        """
        if not info.has_answer:
            return self.config.penalty_no_answer

        judge_score = self._get_judge_score(info.predicted_answer, info.expected_answer)

        return judge_score

    def _get_judge_score(self, predicted: Optional[str], expected: Optional[str]) -> float:
        """
        使用LLM judge获取评分.

        Args:
            predicted: 预测答案
            expected: 期望答案

        Returns:
            float: 评分（0-1）
        """
        if not self.llm_judge:
            # 如果没有judge，使用简单匹配
            if predicted and expected:
                return 1.0 if predicted.strip() == expected.strip() else 0.0
            return 0.0

        # 使用LLM judge（需要实现judge接口）
        # 根据论文Appendix B.3的prompt模板
        expected_str = expected or "N/A"
        predicted_str = predicted or "N/A"
        judge_prompt = (
            "You are an expert judge evaluating the correctness of answers to questions.\n"
            "Given the following information:\n"
            "- Question: [QUESTION]\n"
            f"- Ground-truth Answer: {expected_str}\n"
            f"- Agent's Answer: {predicted_str}\n"
            "\n"
            "Please evaluate the generated answer on a scale of 0.0 to 1.0:\n"
            "- 1.0: Perfect match or equivalent correct answer\n"
            "- 0.8-0.9: Mostly correct with minor differences\n"
            "- 0.6-0.7: Partially correct or close approximation\n"
            "- 0.4-0.5: Some correct elements but significant errors\n"
            "- 0.2-0.3: Mostly incorrect with few correct elements\n"
            "- 0.0-0.1: Completely incorrect or irrelevant\n"
            "\n"
            'Respond with only a number between 0.0 and 1.0 (e.g., "0.85").'
        )

        try:
            # 调用LLM judge（需要根据实际接口实现）
            # 优先检查chat方法（更常见）
            if hasattr(self.llm_judge, "chat") and callable(getattr(self.llm_judge, "chat", None)):
                response = self.llm_judge.chat([{"role": "user", "content": judge_prompt}])
                if not isinstance(response, str):
                    if hasattr(response, "response_text"):
                        response = response.response_text
                    else:
                        response = str(response)
            elif hasattr(self.llm_judge, "generate") and callable(
                getattr(self.llm_judge, "generate", None)
            ):
                response = self.llm_judge.generate(judge_prompt)
            else:
                response = str(self.llm_judge(judge_prompt))

            # 确保response是字符串
            if not isinstance(response, str):
                response = str(response)

            # 解析评分
            score = self._parse_judge_score(response)
            return max(0.0, min(1.0, score))  # 限制在[0, 1]
        except Exception:
            # 出错时返回0
            return 0.0

    def _parse_judge_score(self, response: str) -> float:
        """解析judge响应中的分数."""
        # 尝试提取数字
        numbers = re.findall(r"\d+\.?\d*", response)
        if numbers:
            return float(numbers[0])
        return 0.0

    def _compute_context_reward(self, info: TrajectoryInfo) -> float:
        """
        计算上下文管理奖励.

        公式: R_context = α1*R_compression + α2*R_preventive + α3*R_preservation
        """
        compression = self._compute_compression_reward(info)
        preventive = self._compute_preventive_reward(info)
        preservation = self._compute_preservation_reward(info)

        return (
            self.config.alpha_compression * compression
            + self.config.alpha_preventive * preventive
            + self.config.alpha_preservation * preservation
        )

    def _compute_compression_reward(self, info: TrajectoryInfo) -> float:
        """
        计算压缩效率奖励.

        公式:R_compression = max(0, 1 - T_used / T_max)
        """
        if self.config.max_tokens == 0:
            return 1.0

        ratio = info.final_token_count / self.config.max_tokens
        return max(0.0, 1.0 - ratio)

    def _compute_preventive_reward(self, info: TrajectoryInfo) -> float:
        """
        计算预防性管理奖励.

        公式:R_preventive = 1 if tool invoked before overflow else 0
        """
        return 1.0 if info.tool_calls_before_overflow else 0.0

    def _compute_preservation_reward(self, info: TrajectoryInfo) -> float:
        """
        计算信息保留奖励.

        公式:R_preservation = 1 if key tokens preserved else 0
        """
        return 1.0 if info.key_tokens_preserved else 0.0

    def _compute_memory_reward(
        self, info: TrajectoryInfo, agent_state: Optional[AgentState] = None
    ) -> float:
        """
        计算记忆管理奖励.

        公式:R_memory = β1*R_storage + β2*R_maintenance + β3*R_relevance
        """
        storage = self._compute_storage_reward(info)
        maintenance = self._compute_maintenance_reward(info)
        relevance = self._compute_relevance_reward(info, agent_state)

        return (
            self.config.beta_storage * storage
            + self.config.beta_maintenance * maintenance
            + self.config.beta_relevance * relevance
        )

    def _compute_storage_reward(self, info: TrajectoryInfo) -> float:
        """
        计算存储质量奖励.

        公式:R_storage = N_high_quality / max(1, N_total)
        """
        if info.total_memories_added == 0:
            return 0.0

        return info.high_quality_memories / max(1, info.total_memories_added)

    def _compute_maintenance_reward(self, info: TrajectoryInfo) -> float:
        """
        计算维护奖励.

        公式:R_maintenance = 1 if update/delete performed else 0
        """
        return 1.0 if info.update_or_delete_performed else 0.0

    def _compute_relevance_reward(
        self, info: TrajectoryInfo, agent_state: Optional[AgentState] = None
    ) -> float:
        """
        计算语义相关性奖励.

        公式:R_relevance = S_LLM(R, q)
        """
        if not info.retrieved_memories or not info.query:
            return 0.0

        if not self.llm_relevance_scorer:
            # 如果没有scorer，使用简单启发式
            return 0.5  # 默认中等相关性

        # 使用LLM评估相关性（需要实现）
        # 这里简化处理，实际应该调用LLM
        return 0.5

    def _compute_penalty(self, info: TrajectoryInfo) -> float:
        """
        计算惩罚项.

        公式:P_penalty = P_rounds*1[N_rounds>N_max] + P_overflow*1[T_used>T_max]
        """
        penalty = 0.0

        # 轮数超限惩罚
        if info.num_rounds > self.config.max_rounds:
            penalty += self.config.penalty_rounds

        # Token超限惩罚
        if info.final_token_count > self.config.max_tokens:
            penalty += self.config.penalty_overflow

        return penalty

    def extract_trajectory_info(
        self,
        agent_state: AgentState,
        execution_trace: List[Dict[str, Any]],
        query: str,
        expected_answer: Optional[str] = None,
    ) -> TrajectoryInfo:
        """
        从Agent执行轨迹中提取信息.

        Args:
            agent_state: Agent最终状态
            execution_trace: 执行轨迹
            query: 查询
            expected_answer: 期望答案

        Returns:
            TrajectoryInfo: 轨迹信息
        """
        info = TrajectoryInfo(query=query, expected_answer=expected_answer)

        # 提取答案
        for step in reversed(execution_trace):
            parsed = step.get("parsed")
            if parsed and hasattr(parsed, "has_answer") and parsed.has_answer():
                info.predicted_answer = parsed.answer
                info.has_answer = True
                break

        # 提取上下文信息
        info.final_token_count = agent_state.context.get_token_count()

        # 检查是否在溢出前调用了工具
        # （简化：检查是否有Summary或Filter调用）
        for step in execution_trace:
            exec_info = step.get("execution", {})
            tool_results = exec_info.get("tool_results", [])
            for result in tool_results:
                tool_name = result.get("tool_name", "")
                if tool_name in ["Summary_context", "Filter_context"]:
                    info.tool_calls_before_overflow = True
                    break

        # 提取记忆信息
        # 统计Add_memory调用
        for step in execution_trace:
            exec_info = step.get("execution", {})
            tool_results = exec_info.get("tool_results", [])
            for result in tool_results:
                tool_name = result.get("tool_name", "")
                if tool_name == "Add_memory":
                    info.total_memories_added += 1
                    # 判断是否为高质量（简化：根据内容长度等）
                    # 实际应该使用LLM判断
                    if self._is_high_quality_memory(result):
                        info.high_quality_memories += 1
                elif tool_name in ["Update_memory", "Delete_memory"]:
                    info.update_or_delete_performed = True
                elif tool_name == "Retrieve_memory":
                    # 记录检索到的记忆
                    retrieved = result.get("result", {}).get("memories", [])
                    info.retrieved_memories.extend(retrieved)

        # 提取交互信息
        info.num_rounds = len(execution_trace)

        # 提取工具调用历史
        for step in execution_trace:
            exec_info = step.get("execution", {})
            tool_results = exec_info.get("tool_results", [])
            info.tool_call_history.extend(tool_results)

        # 检查关键token保留（简化实现）
        info.key_tokens_preserved = self._check_key_tokens_preserved(query, agent_state.context)

        return info

    def _is_high_quality_memory(self, tool_result: Dict[str, Any]) -> bool:
        """判断记忆是否为高质量（简化实现）."""
        # 实际应该使用LLM判断
        # 这里使用简单启发式：内容长度、是否有metadata等
        result = tool_result.get("result", {})
        message = result.get("message", "")
        # 尝试从result中提取content
        if not message:
            # 如果没有message，尝试从其他字段提取
            content = result.get("content", "")
            if content:
                return len(content) > 20
        return len(message) > 20  # 简单判断

    def _check_key_tokens_preserved(self, query: str, context: STMManager) -> bool:
        """检查关键token是否保留（简化实现）."""
        # 提取query中的关键token（实体、时间等）
        # 检查context中是否包含这些token
        # 这里简化处理
        messages = context.get_messages()
        context_text = " ".join([msg.get("content", "") for msg in messages])

        # 简单检查：query中的关键词是否在context中
        query_words = set(query.lower().split())
        context_words = set(context_text.lower().split())

        # 如果至少50%的关键词保留，认为保留成功
        overlap = len(query_words & context_words)
        return overlap / max(1, len(query_words)) >= 0.5
