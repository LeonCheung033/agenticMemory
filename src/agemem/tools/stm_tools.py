"""STM工具实现."""
from typing import Dict, Any, Optional, List
import numpy as np

from .base import BaseTool, ToolSchema
from ..memory.ltm import LTMManager
from ..memory.stm import STMManager


class RetrieveMemoryTool(BaseTool):
    """Retrieve_memory工具（STM工具，从LTM检索到STM）."""

    def __init__(self, ltm_manager: LTMManager, stm_manager: STMManager):
        super().__init__()
        self.ltm = ltm_manager
        self.stm = stm_manager
        self._schema = ToolSchema(
            name="Retrieve_memory",
            description="Retrieves relevant memories and adds them to current context.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant memories. Should describe what kind of information or context is needed.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "The maximum number of memories to retrieve. Defaults to 3.",
                    },
                    "metadata_filter": {
                        "type": "object",
                        "description": "Optional metadata filters to narrow down memory search (e.g., {'type': 'user_info', 'domain': 'math'}).",
                        "additionalProperties": True,
                    },
                },
                "required": ["query"],
            },
        )

    @property
    def schema(self) -> ToolSchema:
        return self._schema

    def execute(
        self,
        query: str,
        top_k: int = 3,
        metadata_filter: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """执行Retrieve_memory工具."""
        try:
            # 从LTM检索
            memories = self.ltm.retrieve(query, top_k, filters=metadata_filter)

            # 添加到STM上下文
            retrieved_contents = []
            for memory in memories:
                content = f"[Memory] {memory.content}"
                self.stm.add_message(
                    "system",
                    content,
                    {
                        "memory_id": memory.memory_id,
                        "source": "ltm_retrieval",
                    },
                )
                retrieved_contents.append(
                    {
                        "memory_id": memory.memory_id,
                        "content": memory.content,
                        "metadata": memory.metadata,
                    }
                )

            return {
                "success": True,
                "result": {
                    "memories": retrieved_contents,
                    "count": len(retrieved_contents),
                    "message": f"Retrieved and added {len(retrieved_contents)} memories to context",
                },
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class SummaryContextTool(BaseTool):
    """Summary_context工具."""

    # 论文附录中的总结prompt
    SUMMARY_PROMPT_TEMPLATE = """You are a conversation summarization assistant. 
Your goal is to compress the given conversation span into a concise summary that preserves all important information, intentions, decisions, and unresolved questions. 
The summary will later be used to replace the original conversation in the context, so make sure nothing essential is lost.

Instructions:
1. Read the provided conversation rounds carefully.
2. Identify the main topics, actions, results, and open issues.
3. Write a clear, factual summary in natural language.
4. Do NOT include greetings, filler text, or redundant phrasing.

Input:
- Conversation content: [CONVERSATION_TEXT]

Output:
- A concise yet comprehensive summary of the above conversation span.

Let's start the conversation summarization."""

    def __init__(self, stm_manager: STMManager, llm: Any):
        super().__init__()
        self.stm = stm_manager
        self.llm = llm
        self._schema = ToolSchema(
            name="Summary_context",
            description="Summarizes conversation rounds to reduce tokens while preserving key information.",
            parameters={
                "type": "object",
                "properties": {
                    "span": {
                        "type": "string",
                        "description": "The range of conversation rounds to summarize. Can be 'all' for entire context, or a number (e.g., '5') for the last N rounds. A system, user, assistant and 'tool' message are considered as one round.",
                    },
                },
                "required": ["span"],
            },
        )

    @property
    def schema(self) -> ToolSchema:
        return self._schema

    def execute(self, span: str) -> Dict[str, Any]:
        """执行Summary_context工具."""
        try:
            # 解析span参数
            if span.lower() == "all":
                # 总结所有非系统消息
                to_summarize = [
                    msg
                    for msg in self.stm.context_history
                    if msg.get("role") != "system"
                ]
                keep_recent = 0
            else:
                # 总结最后N轮
                try:
                    n_rounds = int(span)
                    # 计算需要保留的消息数（每轮包含system/user/assistant/tool）
                    keep_recent = n_rounds * 4  # 保守估计
                    to_summarize = (
                        self.stm.context_history[:-keep_recent]
                        if keep_recent < len(self.stm.context_history)
                        else []
                    )
                except ValueError:
                    return {
                        "success": False,
                        "error": f"Invalid span value: {span}. Must be 'all' or a number.",
                    }

            if not to_summarize:
                return {
                    "success": True,
                    "result": {
                        "message": "No messages to summarize",
                        "current_tokens": self.stm.get_token_count(),
                    },
                }

            # 使用论文中的总结prompt
            summary = self.stm.summarize(
                self.llm, to_summarize, self.SUMMARY_PROMPT_TEMPLATE
            )

            # 替换旧消息为总结
            summary_tokens = self.stm.token_counter.count(summary)
            summary_msg = {
                "role": "system",
                "content": summary,
                "tokens": summary_tokens,
                "metadata": {"type": "summary", "original_count": len(to_summarize)},
            }

            if span.lower() == "all":
                self.stm.context_history = [summary_msg]
            else:
                recent = self.stm.context_history[-keep_recent:]
                self.stm.context_history = [summary_msg] + recent

            return {
                "success": True,
                "result": {
                    "summary": summary,
                    "remaining_tokens": self.stm.get_token_count(),
                    "message": f"Summarized {len(to_summarize)} messages",
                },
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class FilterContextTool(BaseTool):
    """Filter_context工具."""

    def __init__(self, stm_manager: STMManager, ltm_manager: LTMManager):
        super().__init__()
        self.stm = stm_manager
        self.ltm = ltm_manager  # 用于获取嵌入模型
        self._schema = ToolSchema(
            name="Filter_context",
            description="Filters out irrelevant or outdated content from the conversation context to improve task-solving efficiency.",
            parameters={
                "type": "object",
                "properties": {
                    "criteria": {
                        "type": "string",
                        "description": "The criteria for content removal. Can be keywords, phrases, or descriptions of content types to remove (e.g., 'the birthday of John', 'the age of Mary').",
                    },
                },
                "required": ["criteria"],
            },
        )

    @property
    def schema(self) -> ToolSchema:
        return self._schema

    def execute(self, criteria: str, threshold: float = 0.6) -> Dict[str, Any]:
        """
        执行Filter_context工具.

        根据论文公式：
        C_t' = {u_i ∈ C_t | sim(c, u_i) < θ}
        默认θ=0.6
        """
        try:
            # 生成criteria的嵌入
            criteria_embedding = self.ltm.embedding_model.encode(
                criteria, convert_to_numpy=True
            )

            # 过滤消息
            removed_count = self.stm.filter_messages(
                criteria_embedding,
                threshold=threshold,
                embedding_model=self.ltm.embedding_model,
            )

            return {
                "success": True,
                "result": {
                    "removed_count": removed_count,
                    "remaining_count": len(self.stm.context_history),
                    "remaining_tokens": self.stm.get_token_count(),
                    "message": f"Filtered out {removed_count} messages based on criteria",
                },
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
