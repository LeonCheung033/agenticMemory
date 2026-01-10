"""测试Agent主类."""
import pytest
from unittest.mock import Mock, MagicMock
from src.agemem.agent.agent import AgeMemAgent, AgentState
from src.agemem.memory.ltm import LTMManager
from src.agemem.memory.stm import STMManager


class TestAgeMemAgent:
    """测试AgeMemAgent."""

    @pytest.fixture
    def mock_llm(self):
        """创建Mock LLM."""
        llm = Mock()
        llm.chat = Mock(return_value="<think>Test reasoning</think><answer>Test answer</answer>")
        return llm

    @pytest.fixture
    def agent(self, mock_llm):
        """创建Agent实例."""
        ltm = LTMManager()
        stm = STMManager()
        return AgeMemAgent(ltm, stm, mock_llm, max_turns=5)

    def test_init(self, agent):
        """测试初始化."""
        assert agent.ltm is not None
        assert agent.stm is not None
        assert len(agent.tools) == 6  # 6个工具
        assert len(agent.tool_map) == 6
        assert agent.system_prompt is not None

    def test_initialize_tools(self, agent):
        """测试工具初始化."""
        assert len(agent.tools) == 6
        tool_names = [tool.schema.name for tool in agent.tools]
        assert "Add_memory" in tool_names
        assert "Update_memory" in tool_names
        assert "Delete_memory" in tool_names
        assert "Retrieve_memory" in tool_names
        assert "Summary_context" in tool_names
        assert "Filter_context" in tool_names

    def test_reset(self, agent):
        """测试重置."""
        task_spec = {
            "query": "test query",
            "context_info": "test context",
            "expected_answer": None,
        }
        agent.reset(task_spec, reset_stm=True)

        assert len(agent.conversation_history) > 0
        assert agent.conversation_history[0]["role"] == "system"
        assert agent.stm.get_token_count() == 0  # STM已清空

    def test_reset_without_stm_reset(self, agent):
        """测试不重置STM的重置."""
        # 先添加一些消息
        agent.stm.add_message("user", "test message")
        assert agent.stm.get_token_count() > 0

        task_spec = {"query": "test"}
        agent.reset(task_spec, reset_stm=False)

        # STM应该保留
        assert agent.stm.get_token_count() > 0

    def test_get_state(self, agent):
        """测试获取状态."""
        task_spec = {"query": "test query"}
        state = agent.get_state(task_spec)

        assert isinstance(state, AgentState)
        assert state.context == agent.stm
        assert state.memory == agent.ltm
        assert state.task_spec == task_spec

    def test_step_with_tool_call(self, agent, mock_llm):
        """测试带工具调用的step."""
        # Mock LLM返回带tool_call的响应
        mock_llm.chat.return_value = """<think>I need to add memory</think>
<tool_call>[{"name": "Add_memory", "arguments": {"content": "test memory"}}]</tool_call>"""

        agent.reset({"query": "test"}, reset_stm=True)
        parsed, exec_info = agent.step()

        assert exec_info["success"] is True
        assert parsed.has_tool_calls()
        assert len(exec_info["tool_results"]) > 0
        assert exec_info["tool_results"][0]["tool_name"] == "Add_memory"

    def test_step_with_answer(self, agent, mock_llm):
        """测试带答案的step."""
        # Mock LLM返回带answer的响应
        mock_llm.chat.return_value = """<think>I have the answer</think>
<answer>The answer is 42</answer>"""

        agent.reset({"query": "test"}, reset_stm=True)
        parsed, exec_info = agent.step()

        assert exec_info["success"] is True
        assert parsed.has_answer()
        assert parsed.answer == "The answer is 42"

    def test_step_with_user_input(self, agent):
        """测试带用户输入的step."""
        agent.reset({"query": "test"}, reset_stm=True)
        parsed, exec_info = agent.step(user_input="Additional context")

        # 检查用户输入是否添加到历史
        user_messages = [msg for msg in agent.conversation_history if msg["role"] == "user"]
        assert len(user_messages) > 0

    def test_step_invalid_format(self, agent, mock_llm):
        """测试格式无效的响应."""
        # Mock LLM返回无效格式
        mock_llm.chat.return_value = "Invalid response without tags"

        agent.reset({"query": "test"}, reset_stm=True)
        parsed, exec_info = agent.step()

        assert exec_info["success"] is False
        assert "error" in exec_info

    def test_execute_tools(self, agent):
        """测试工具执行."""
        tool_calls = [
            {"name": "Add_memory", "arguments": {"content": "test memory"}},
            {"name": "Unknown_tool", "arguments": {}},
        ]

        results = agent._execute_tools(tool_calls)

        assert len(results) == 2
        assert results[0]["success"] is True
        assert results[1]["success"] is False
        assert "Unknown tool" in results[1]["error"]

    def test_execute_tools_invalid_args(self, agent):
        """测试无效参数的工具调用."""
        tool_calls = [
            {"name": "Add_memory", "arguments": {}},  # 缺少required参数
        ]

        results = agent._execute_tools(tool_calls)

        assert len(results) == 1
        assert results[0]["success"] is False
        assert "Missing required parameter" in results[0]["error"]

    def test_run_until_answer(self, agent, mock_llm):
        """测试运行直到获得答案."""
        # Mock LLM先返回tool_call，然后返回answer
        call_count = 0

        def mock_chat(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return """<think>I need memory</think>
<tool_call>[{"name": "Add_memory", "arguments": {"content": "test"}}]</tool_call>"""
            else:
                return """<think>I have answer</think>
<answer>Final answer</answer>"""

        mock_llm.chat.side_effect = mock_chat

        task_spec = {"query": "test query"}
        answer, trace = agent.run_until_answer(task_spec, reset_stm=True)

        assert answer == "Final answer"
        assert len(trace) > 0

    def test_run_until_answer_max_turns(self, agent, mock_llm):
        """测试达到最大轮数."""
        # Mock LLM始终返回tool_call
        mock_llm.chat.return_value = """<think>Need tool</think>
<tool_call>[{"name": "Add_memory", "arguments": {"content": "test"}}]</tool_call>"""

        task_spec = {"query": "test"}
        answer, trace = agent.run_until_answer(task_spec, reset_stm=True)

        assert answer is None
        assert len(trace) == agent.max_turns

    def test_call_llm_various_formats(self, agent):
        """测试不同LLM响应格式."""
        # 测试字符串响应
        agent.llm.chat = Mock(return_value="<think>test</think>")
        result = agent._call_llm()
        assert isinstance(result, str)

        # 测试带response_text属性的对象
        mock_response = Mock()
        mock_response.response_text = "<think>test</think>"
        agent.llm.chat = Mock(return_value=mock_response)
        result = agent._call_llm()
        assert result == "<think>test</think>"

        # 测试字典响应
        agent.llm.chat = Mock(return_value={"content": "<think>test</think>"})
        result = agent._call_llm()
        assert result == "<think>test</think>"

        # 测试列表响应
        mock_msg = Mock()
        mock_msg.response_text = "<think>test</think>"
        agent.llm.chat = Mock(return_value=[mock_msg])
        result = agent._call_llm()
        assert result == "<think>test</think>"


class TestAgentState:
    """测试AgentState."""

    def test_to_dict(self):
        """测试状态转换为字典."""
        ltm = LTMManager()
        stm = STMManager()
        task_spec = {"query": "test query"}

        state = AgentState(context=stm, memory=ltm, task_spec=task_spec)
        state_dict = state.to_dict()

        assert "context_tokens" in state_dict
        assert "memory_count" in state_dict
        assert "task_query" in state_dict
        assert state_dict["task_query"] == "test query"
