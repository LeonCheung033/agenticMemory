"""响应解析器模块."""
import re
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ParsedResponse:
    """解析后的响应结构."""

    reasoning: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    answer: Optional[str] = None
    raw_text: str = ""

    def has_tool_calls(self) -> bool:
        """是否有工具调用."""
        return len(self.tool_calls) > 0

    def has_answer(self) -> bool:
        """是否有最终答案."""
        return self.answer is not None

    def is_complete(self) -> bool:
        """响应是否完整（有reasoning和answer或tool_calls）."""
        return self.reasoning is not None and (self.has_answer() or self.has_tool_calls())


class ResponseParser:
    """解析Agent结构化响应."""

    # 正则表达式模式
    REASONING_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

    def parse(self, response_text: str) -> ParsedResponse:
        """
        解析响应文本.

        Args:
            response_text: Agent的原始响应文本

        Returns:
            ParsedResponse: 解析后的结构化响应
        """
        parsed = ParsedResponse(raw_text=response_text)

        # 提取reasoning（可能有多个，取最后一个）
        reasoning_matches = self.REASONING_PATTERN.findall(response_text)
        if reasoning_matches:
            parsed.reasoning = reasoning_matches[-1].strip()

        # 提取tool_calls
        tool_call_matches = self.TOOL_CALL_PATTERN.findall(response_text)
        if tool_call_matches:
            # 取最后一个tool_call块
            tool_call_text = tool_call_matches[-1].strip()
            parsed.tool_calls = self._parse_tool_calls(tool_call_text)

        # 提取answer
        answer_matches = self.ANSWER_PATTERN.findall(response_text)
        if answer_matches:
            parsed.answer = answer_matches[-1].strip()

        return parsed

    def _parse_tool_calls(self, tool_call_text: str) -> List[Dict[str, Any]]:
        """
        解析工具调用JSON.

        Args:
            tool_call_text: tool_call标签内的文本

        Returns:
            List[Dict]: 工具调用列表
        """
        try:
            # 尝试解析JSON数组
            tool_calls = json.loads(tool_call_text)
            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]
            return tool_calls
        except json.JSONDecodeError:
            # 如果解析失败，尝试提取单个工具调用
            try:
                # 尝试提取单个JSON对象
                single_call = json.loads(tool_call_text)
                return [single_call]
            except json.JSONDecodeError:
                # 解析失败，返回空列表
                return []

    def validate(self, parsed: ParsedResponse) -> Tuple[bool, Optional[str]]:
        """
        验证解析结果是否符合格式要求.

        Args:
            parsed: 解析后的响应

        Returns:
            (is_valid, error_message)
        """
        # 必须有reasoning
        if not parsed.reasoning:
            return False, "Missing <think> block"

        # 不能同时有tool_calls和answer（互斥规则）
        if parsed.has_tool_calls() and parsed.has_answer():
            return False, "Cannot have both <tool_call> and <answer> in the same response"

        # 必须有tool_calls或answer之一
        if not parsed.has_tool_calls() and not parsed.has_answer():
            return False, "Must have either <tool_call> or <answer> after <think>"

        return True, None
