"""测试系统提示词生成器."""
import pytest
from unittest.mock import Mock
from src.agemem.agent.prompt import SystemPromptGenerator
from src.agemem.tools.base import BaseTool, ToolSchema


class MockTool(BaseTool):
    """用于测试的Mock工具."""

    def __init__(self, name: str, description: str):
        super().__init__()
        self._schema = ToolSchema(
            name=name,
            description=description,
            parameters={
                "type": "object",
                "properties": {
                    "param1": {"type": "string", "description": "First parameter"},
                    "param2": {"type": "integer", "description": "Second parameter"},
                },
                "required": ["param1"],
            },
        )

    @property
    def schema(self) -> ToolSchema:
        return self._schema

    def execute(self, **kwargs) -> dict:
        return {"success": True}


class TestSystemPromptGenerator:
    """测试SystemPromptGenerator."""

    def test_init(self):
        """测试初始化."""
        tools = [MockTool("test_tool", "Test tool description")]
        generator = SystemPromptGenerator(tools)
        assert len(generator.tools) == 1

    def test_generate_tools_section(self):
        """测试工具部分生成."""
        tools = [
            MockTool("Tool1", "Description 1"),
            MockTool("Tool2", "Description 2"),
        ]
        generator = SystemPromptGenerator(tools)
        section = generator.generate_tools_section()

        assert "## Available Tools:" in section
        assert "Tool1" in section
        assert "Description 1" in section
        assert "Tool2" in section
        assert "Description 2" in section
        assert "Parameters:" in section

    def test_format_parameters(self):
        """测试参数格式化."""
        tools = [MockTool("test_tool", "Test")]
        generator = SystemPromptGenerator(tools)
        params = {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "First param"},
                "param2": {"type": "integer", "description": "Second param"},
            },
            "required": ["param1"],
        }

        formatted = generator._format_parameters(params)
        assert "param1" in formatted
        assert "First param" in formatted
        assert "(required)" in formatted
        assert "param2" in formatted
        assert "(optional)" in formatted

    def test_generate_full_prompt(self):
        """测试完整提示词生成."""
        tools = [MockTool("test_tool", "Test tool")]
        generator = SystemPromptGenerator(tools)
        prompt = generator.generate_full_prompt()

        # 检查关键部分
        assert "intelligent assistant" in prompt.lower()
        assert "## Available Tools:" in prompt
        assert "## Problem-Solving Workflow" in prompt
        assert "## Response Format (Strict)" in prompt
        assert "## Guidelines" in prompt
        assert "<think>" in prompt
        assert "<tool_call>" in prompt
        assert "<answer>" in prompt

    def test_prompt_contains_all_tools(self):
        """测试提示词包含所有工具."""
        tools = [
            MockTool("Add_memory", "Add memory"),
            MockTool("Retrieve_memory", "Retrieve memory"),
        ]
        generator = SystemPromptGenerator(tools)
        prompt = generator.generate_full_prompt()

        assert "Add_memory" in prompt
        assert "Retrieve_memory" in prompt

    def test_prompt_format_consistency(self):
        """测试提示词格式一致性."""
        tools = [MockTool("test", "test")]
        generator = SystemPromptGenerator(tools)
        prompt = generator.generate_full_prompt()

        # 检查格式要求
        assert prompt.startswith("You are an intelligent assistant")
        assert "Let's start!" in prompt
