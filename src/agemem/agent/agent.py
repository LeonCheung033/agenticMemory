"""Agent主类."""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from ..memory.ltm import LTMManager
from ..memory.stm import STMManager
from ..tools.base import BaseTool
from ..tools.ltm_tools import AddMemoryTool, UpdateMemoryTool, DeleteMemoryTool
from ..tools.stm_tools import RetrieveMemoryTool, SummaryContextTool, FilterContextTool
from .prompt import SystemPromptGenerator
from .parser import ResponseParser, ParsedResponse


@dataclass
class AgentState:
    """Agent状态（对应论文中的s_t）."""

    context: STMManager  # C_t
    memory: LTMManager  # M_t
    task_spec: Dict[str, Any]  # T: {query, context_info, expected_answer}

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于序列化）."""
        return {
            "context_tokens": self.context.get_token_count(),
            "memory_count": self.memory.count(),
            "task_query": self.task_spec.get("query", ""),
        }


class AgeMemAgent:
    """
    Agentic Memory Agent.

    实现论文中的Agent，支持：
    - 结构化推理和工具调用
    - 三阶段记忆管理
    - 与LLM交互
    """

    def __init__(
        self,
        ltm_manager: LTMManager,
        stm_manager: STMManager,
        llm_model: Any,  # LLM模型（需要支持chat接口）
        max_turns: int = 20,
    ):
        """
        初始化Agent.

        Args:
            ltm_manager: 长期记忆管理器
            stm_manager: 短期记忆管理器
            llm_model: LLM模型（需要有chat方法）
            max_turns: 最大对话轮数
        """
        self.ltm = ltm_manager
        self.stm = stm_manager
        self.llm = llm_model
        self.max_turns = max_turns

        # 初始化工具
        self.tools = self._initialize_tools()

        # 工具映射（name -> tool）
        self.tool_map = {tool.schema.name: tool for tool in self.tools}

        # 提示词生成器
        self.prompt_generator = SystemPromptGenerator(self.tools)
        self.system_prompt = self.prompt_generator.generate_full_prompt()

        # 响应解析器
        self.parser = ResponseParser()

        # 对话历史
        self.conversation_history: List[Dict[str, str]] = []

    def _initialize_tools(self) -> List[BaseTool]:
        """初始化所有记忆管理工具."""
        return [
            AddMemoryTool(self.ltm),
            UpdateMemoryTool(self.ltm),
            DeleteMemoryTool(self.ltm),
            RetrieveMemoryTool(self.ltm, self.stm),
            SummaryContextTool(self.stm, self.llm),
            FilterContextTool(self.stm, self.ltm),
        ]

    def reset(self, task_spec: Dict[str, Any], reset_stm: bool = True):
        """
        重置Agent状态（用于三阶段训练）.

        Args:
            task_spec: 任务规范 {query, context_info, expected_answer}
            reset_stm: 是否重置STM（Stage 2需要重置）
        """
        if reset_stm:
            self.stm.clear()

        # 重置对话历史
        self.conversation_history = []

        # 添加系统提示词
        self.conversation_history.append({"role": "system", "content": self.system_prompt})

        # 添加任务信息
        if task_spec.get("context_info"):
            self.conversation_history.append({"role": "user", "content": task_spec["context_info"]})

        if task_spec.get("query"):
            self.conversation_history.append({"role": "user", "content": task_spec["query"]})

    def get_state(self, task_spec: Dict[str, Any]) -> AgentState:
        """
        获取当前状态（论文中的s_t）.

        Args:
            task_spec: 任务规范

        Returns:
            AgentState: 当前状态
        """
        return AgentState(context=self.stm, memory=self.ltm, task_spec=task_spec)

    def step(self, user_input: Optional[str] = None) -> Tuple[ParsedResponse, Dict[str, Any]]:
        """
        执行一步Agent推理.

        Args:
            user_input: 可选的用户输入（用于Stage 1和Stage 2）

        Returns:
            (parsed_response, execution_info)
        """
        # 添加用户输入（如果有）
        if user_input:
            self.conversation_history.append({"role": "user", "content": user_input})
            self.stm.add_message("user", user_input)

        # 调用LLM生成响应
        response_text = self._call_llm()

        # 解析响应
        parsed = self.parser.parse(response_text)

        # 验证格式
        is_valid, error = self.parser.validate(parsed)
        if not is_valid:
            # 格式错误，返回错误信息
            execution_info = {"success": False, "error": error, "raw_response": response_text}
            return parsed, execution_info

        # 添加Assistant响应到历史
        self.conversation_history.append({"role": "assistant", "content": response_text})
        self.stm.add_message("assistant", response_text)

        # 执行工具调用（如果有）
        execution_info = {
            "success": True,
            "tool_results": [],
            "has_answer": parsed.has_answer(),
        }

        if parsed.has_tool_calls():
            tool_results = self._execute_tools(parsed.tool_calls)
            execution_info["tool_results"] = tool_results

            # 将工具结果添加到上下文
            for result in tool_results:
                tool_message = f"Tool {result['tool_name']} executed: {result.get('message', '')}"
                self.conversation_history.append({"role": "tool", "content": tool_message})
                self.stm.add_message("tool", tool_message, result)

        return parsed, execution_info

    def _call_llm(self) -> str:
        """
        调用LLM生成响应.

        Returns:
            str: LLM响应文本
        """
        # 构建消息列表（从conversation_history）
        messages = self.conversation_history.copy()

        # 调用LLM（假设llm有chat方法）
        if hasattr(self.llm, "chat"):
            response = self.llm.chat(messages)
            if isinstance(response, str):
                return response
            elif hasattr(response, "response_text"):
                return response.response_text
            elif isinstance(response, dict) and "content" in response:
                return response["content"]
            else:
                # 尝试提取第一个消息的内容
                if isinstance(response, list) and len(response) > 0:
                    first_msg = response[0]
                    if hasattr(first_msg, "response_text"):
                        return first_msg.response_text
                    elif isinstance(first_msg, dict):
                        return first_msg.get("content", str(first_msg))

        # 如果都不匹配，返回字符串表示
        return str(response)

    def _execute_tools(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        执行工具调用.

        Args:
            tool_calls: 工具调用列表

        Returns:
            List[Dict]: 工具执行结果列表
        """
        results = []

        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            arguments = tool_call.get("arguments", {})

            if tool_name not in self.tool_map:
                results.append({"tool_name": tool_name, "success": False, "error": f"Unknown tool: {tool_name}"})
                continue

            tool = self.tool_map[tool_name]

            # 验证参数
            is_valid, error = tool.validate_args(arguments)
            if not is_valid:
                results.append({"tool_name": tool_name, "success": False, "error": error})
                continue

            # 执行工具
            try:
                result = tool.execute(**arguments)
                results.append(
                    {
                        "tool_name": tool_name,
                        "success": result.get("success", False),
                        "result": result.get("result"),
                        "error": result.get("error"),
                        "message": result.get("result", {}).get("message", ""),
                    }
                )
            except Exception as e:
                results.append({"tool_name": tool_name, "success": False, "error": str(e)})

        return results

    def run_until_answer(
        self, task_spec: Dict[str, Any], reset_stm: bool = True
    ) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """
        运行Agent直到获得答案（用于Stage 3）.

        Args:
            task_spec: 任务规范
            reset_stm: 是否重置STM

        Returns:
            (final_answer, execution_trace)
        """
        self.reset(task_spec, reset_stm=reset_stm)

        execution_trace = []

        for turn in range(self.max_turns):
            parsed, exec_info = self.step()
            execution_trace.append({"turn": turn, "parsed": parsed, "execution": exec_info})

            # 如果获得答案，返回
            if parsed.has_answer():
                return parsed.answer, execution_trace

            # 如果没有工具调用，可能出错了
            if not parsed.has_tool_calls():
                break

        # 达到最大轮数，返回最后的状态
        return None, execution_trace
