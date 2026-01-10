"""系统提示词生成模块."""
from typing import List, Dict, Any

from ..tools.base import BaseTool


class SystemPromptGenerator:
    """生成Agent系统提示词."""

    def __init__(self, tools: List[BaseTool]):
        """
        初始化提示词生成器.

        Args:
            tools: 可用工具列表（6个记忆管理工具）
        """
        self.tools = tools

    def generate_tools_section(self) -> str:
        """
        生成工具定义部分.

        返回格式：
        ## Available Tools:
        [TOOL_SCHEMAS]
        """
        tool_schemas = []
        for tool in self.tools:
            schema = tool.schema.to_dict()
            tool_schemas.append(
                f"- **{schema['name']}**: {schema['description']}\n"
                f"  - Parameters: {self._format_parameters(schema['parameters'])}"
            )

        return "## Available Tools:\n" + "\n".join(tool_schemas)

    def _format_parameters(self, params: Dict[str, Any]) -> str:
        """格式化参数定义."""
        props = params.get("properties", {})
        required = params.get("required", [])

        param_list = []
        for name, desc in props.items():
            req_mark = " (required)" if name in required else " (optional)"
            param_list.append(f"`{name}`: {desc.get('description', '')}{req_mark}")

        return ", ".join(param_list)

    def generate_full_prompt(self) -> str:
        """
        生成完整系统提示词.

        严格按照论文附录A.1的格式（第122-166行）
        """
        tools_section = self.generate_tools_section()

        prompt = f"""You are an intelligent assistant that solves complex problems by managing context and memory with tools when needed.

{tools_section}

## Problem-Solving Workflow
You must follow a structured reasoning and action process for every task:
1. **Think & Plan**  
   Always start with a <think>...</think> block.  
   Inside it, explain your reasoning, plan your next step, and decide whether you need to call a tool or provide a final answer.
2. **Tool Calls**  
   If you decide to use one or more tools, follow your <think> block with a <tool_call>...</tool_call> block.  
   - You may call **one or multiple tools** in a single step.  
   - List multiple tool calls as elements of a JSON array.  
   - Each tool call must include "name" and "arguments".  
   - Example:
     <tool_call>[{{"name": "Retrieve_memory", "arguments": {{"query": "math problem solving strategies", "top_k": 3}}}}, {{"name": "Add_memory", "arguments": {{"content": "Strategy summary for reuse", "memory_type": "problem_solving"}}}}]</tool_call>
3. **Final Answer**  
   When you no longer need tools and are ready to present your final output, follow your last <think> block with an <answer>...</answer> block containing the full response.
4. **Mutual Exclusivity Rule**  
   After **each <think> block**, you must choose exactly **one** of the following:
   - a "<tool_call>" block (if you need tools), **or**
   - an "<answer>" block (if you are ready to respond).  
   You must **never** include both "<tool_call>" and "<answer>" immediately after the same "<think>" block.
5. **Iterative Solving**  
   You may repeat this sequence as needed:  
   "<think>" -> "<tool_call>" -> "<think>" -> "<tool_call>" ... -> "<think>" -> "<answer>"  
   until the problem is completely solved.

## Response Format (Strict)
Your full output must follow these rules:
- Every reasoning step must appear inside <think> tags.  
- Every tool usage must appear inside one <tool_call> tag (even if it includes multiple tool invocations).  
- The final solution must be wrapped in <answer> tags.  
- No text should appear outside these tags.

## Guidelines
- Always start with reasoning (<think>).
- After each reasoning step, decide: call tool(s) or answer.
- You can call multiple tools within one <tool_call> JSON array.
- Be concise, logical, and explicit in reasoning.
- Manage memory actively: retrieve, add, update, summarize, filter, or delete as needed.
- Use <answer> only once when the final solution is ready.

Let's start!"""

        return prompt
