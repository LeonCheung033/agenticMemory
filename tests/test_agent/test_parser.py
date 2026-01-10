"""测试响应解析器."""
import pytest
from src.agemem.agent.parser import ResponseParser, ParsedResponse


class TestResponseParser:
    """测试ResponseParser."""

    def test_parse_reasoning_only(self):
        """测试只解析reasoning."""
        parser = ResponseParser()
        text = "<think>This is my reasoning</think>"
        parsed = parser.parse(text)

        assert parsed.reasoning == "This is my reasoning"
        assert not parsed.has_tool_calls()
        assert not parsed.has_answer()

    def test_parse_reasoning_and_tool_call(self):
        """测试解析reasoning和tool_call."""
        parser = ResponseParser()
        text = """<think>I need to retrieve memory</think>
<tool_call>[{"name": "Retrieve_memory", "arguments": {"query": "test"}}]</tool_call>"""
        parsed = parser.parse(text)

        assert parsed.reasoning == "I need to retrieve memory"
        assert parsed.has_tool_calls()
        assert len(parsed.tool_calls) == 1
        assert parsed.tool_calls[0]["name"] == "Retrieve_memory"

    def test_parse_reasoning_and_answer(self):
        """测试解析reasoning和answer."""
        parser = ResponseParser()
        text = """<think>I have the answer</think>
<answer>The answer is 42</answer>"""
        parsed = parser.parse(text)

        assert parsed.reasoning == "I have the answer"
        assert parsed.has_answer()
        assert parsed.answer == "The answer is 42"
        assert not parsed.has_tool_calls()

    def test_parse_multiple_tool_calls(self):
        """测试解析多个工具调用."""
        parser = ResponseParser()
        text = """<think>I need multiple tools</think>
<tool_call>[{"name": "Add_memory", "arguments": {"content": "test"}}, {"name": "Retrieve_memory", "arguments": {"query": "test"}}]</tool_call>"""
        parsed = parser.parse(text)

        assert len(parsed.tool_calls) == 2
        assert parsed.tool_calls[0]["name"] == "Add_memory"
        assert parsed.tool_calls[1]["name"] == "Retrieve_memory"

    def test_parse_multiple_reasoning_blocks(self):
        """测试多个reasoning块（取最后一个）."""
        parser = ResponseParser()
        text = """<think>First reasoning</think>
<tool_call>[{"name": "Add_memory", "arguments": {"content": "test"}}]</tool_call>
<think>Second reasoning</think>
<answer>Final answer</answer>"""
        parsed = parser.parse(text)

        assert parsed.reasoning == "Second reasoning"
        assert parsed.has_answer()
        assert parsed.answer == "Final answer"

    def test_parse_invalid_json(self):
        """测试无效JSON处理."""
        parser = ResponseParser()
        text = """<think>Reasoning</think>
<tool_call>invalid json</tool_call>"""
        parsed = parser.parse(text)

        assert parsed.reasoning == "Reasoning"
        assert not parsed.has_tool_calls()  # 应该返回空列表

    def test_validate_missing_reasoning(self):
        """测试验证缺少reasoning."""
        parser = ResponseParser()
        parsed = ParsedResponse(reasoning=None, tool_calls=[{"name": "test"}])

        is_valid, error = parser.validate(parsed)
        assert not is_valid
        assert "Missing" in error

    def test_validate_both_tool_call_and_answer(self):
        """测试验证同时有tool_call和answer（互斥规则）."""
        parser = ResponseParser()
        parsed = ParsedResponse(
            reasoning="test",
            tool_calls=[{"name": "test"}],
            answer="answer",
        )

        is_valid, error = parser.validate(parsed)
        assert not is_valid
        assert "both" in error.lower() or "Cannot" in error

    def test_validate_no_tool_call_or_answer(self):
        """测试验证既没有tool_call也没有answer."""
        parser = ResponseParser()
        parsed = ParsedResponse(reasoning="test", tool_calls=[], answer=None)

        is_valid, error = parser.validate(parsed)
        assert not is_valid
        assert "either" in error.lower() or "Must" in error

    def test_validate_valid_with_tool_call(self):
        """测试验证有效的tool_call响应."""
        parser = ResponseParser()
        parsed = ParsedResponse(
            reasoning="test",
            tool_calls=[{"name": "test"}],
            answer=None,
        )

        is_valid, error = parser.validate(parsed)
        assert is_valid
        assert error is None

    def test_validate_valid_with_answer(self):
        """测试验证有效的answer响应."""
        parser = ResponseParser()
        parsed = ParsedResponse(reasoning="test", tool_calls=[], answer="answer")

        is_valid, error = parser.validate(parsed)
        assert is_valid
        assert error is None

    def test_parse_with_whitespace(self):
        """测试带空白的解析."""
        parser = ResponseParser()
        text = """<think>
    This is reasoning
    with multiple lines
</think>
<answer>
    The answer
</answer>"""
        parsed = parser.parse(text)

        assert "reasoning" in parsed.reasoning
        assert "answer" in parsed.answer

    def test_is_complete(self):
        """测试is_complete方法."""
        # 完整的响应（有reasoning和tool_call）
        parsed1 = ParsedResponse(reasoning="test", tool_calls=[{"name": "test"}])
        assert parsed1.is_complete()

        # 完整的响应（有reasoning和answer）
        parsed2 = ParsedResponse(reasoning="test", answer="answer")
        assert parsed2.is_complete()

        # 不完整的响应（只有reasoning）
        parsed3 = ParsedResponse(reasoning="test")
        assert not parsed3.is_complete()

        # 不完整的响应（没有reasoning）
        parsed4 = ParsedResponse(tool_calls=[{"name": "test"}])
        assert not parsed4.is_complete()
