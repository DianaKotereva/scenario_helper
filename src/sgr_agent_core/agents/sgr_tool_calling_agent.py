from typing import Literal, Type

from openai import AsyncOpenAI, pydantic_function_tool

from sgr_agent_core.agent_config import AgentConfig
from sgr_agent_core.base_agent import BaseAgent
from sgr_agent_core.models import AgentStatesEnum
from sgr_agent_core.tools import (
    BaseTool,
    FinalAnswerTool,
    ReasoningTool,
    SystemBaseTool,
)


class SGRToolCallingAgent(BaseAgent):
    """Agent that uses OpenAI native function calling to select and execute
    tools based on SGR like a reasoning scheme."""

    name: str = "sgr_tool_calling_agent"

    def __init__(
        self,
        task_messages: list,
        openai_client: AsyncOpenAI,
        agent_config: AgentConfig,
        toolkit: list[Type[BaseTool]],
        *,
        def_name: str | None = None,
        reasoning_tool_cls: type[SystemBaseTool] = ReasoningTool,
        **kwargs: dict,
    ):
        super().__init__(
            task_messages=task_messages,
            openai_client=openai_client,
            agent_config=agent_config,
            toolkit=toolkit,
            def_name=def_name,
            **kwargs,
        )
        self.tool_choice: Literal["required"] = "required"
        self.ReasoningTool: type[SystemBaseTool] = reasoning_tool_cls
        self._reasoning_parse_retries: int = 2

    async def _reasoning_phase(self) -> ReasoningTool:
        phase_id = f"{self._context.iteration}-reasoning"
        last_completion = None
        reasoning: ReasoningTool | None = None
        for attempt in range(1, self._reasoning_parse_retries + 2):
            messages = await self._prepare_context()
            if attempt > 1:
                messages = messages + [
                    {
                        "role": "system",
                        "content": (
                            f"RETRY {attempt}: In reasoning phase call ONLY tool "
                            f"'{self.ReasoningTool.tool_name}' with valid arguments."
                        ),
                    }
                ]
            tools = [pydantic_function_tool(self.ReasoningTool, name=self.ReasoningTool.tool_name)]
            self._log_llm_call(
                phase="reasoning",
                request_payload={
                    "attempt": attempt,
                    "messages": messages,
                    "tools": [{"name": self.ReasoningTool.tool_name}],
                    "tool_choice": self.tool_choice,
                    "llm_kwargs": self.config.llm.to_openai_client_kwargs(),
                },
            )
            async with self.openai_client.chat.completions.stream(
                messages=messages,
                tools=tools,
                tool_choice=self.tool_choice,
                **self.config.llm.to_openai_client_kwargs(),
            ) as stream:
                async for event in stream:
                    if event.type == "chunk":
                        self.streaming_generator.add_chunk(event.chunk, phase_id)
                final_completion = await stream.get_final_completion()
            last_completion = final_completion
            self._log_llm_call(
                phase="reasoning",
                request_payload={"attempt": attempt, "messages_count": len(messages), "tool_choice": self.tool_choice},
                response_payload=getattr(final_completion, "model_dump", lambda **_: {"raw": str(final_completion)})(
                    mode="json"
                ),
            )
            try:
                reasoning = final_completion.choices[0].message.tool_calls[0].function.parsed_arguments
            except (IndexError, AttributeError, TypeError):
                reasoning = None
            if isinstance(reasoning, ReasoningTool):
                break

        if not isinstance(reasoning, ReasoningTool):
            # Fail-closed: do not crash, provide deterministic failed reasoning.
            reasoning = ReasoningTool(
                reasoning_steps=[
                    "Model returned invalid reasoning tool payload after retries",
                    "Cannot continue safely with tool execution",
                ],
                current_situation="Reasoning tool call parse failed",
                plan_status="Aborted due to invalid LLM tool payload",
                enough_data=False,
                remaining_steps=["Return failure answer requesting retry"],
                task_completed=False,
            )
            self._log_llm_call(
                phase="reasoning",
                request_payload={"attempt": "final_fallback"},
                response_payload=getattr(last_completion, "model_dump", lambda **_: {"raw": str(last_completion)})(
                    mode="json"
                )
                if last_completion is not None
                else None,
                error="reasoning_parsed_arguments_none_after_retries",
            )
        self.streaming_generator.add_tool_call(phase_id, reasoning)
        self.conversation.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "id": phase_id,
                        "function": {
                            "name": reasoning.tool_name,
                            "arguments": reasoning.model_dump_json(),
                        },
                    }
                ],
            }
        )
        tool_call_result = await reasoning(self._context, self.config)
        self.streaming_generator.add_tool_result(phase_id, tool_call_result, reasoning.tool_name)
        self.conversation.append({"role": "tool", "content": tool_call_result, "tool_call_id": phase_id})
        self._log_reasoning(reasoning)
        return reasoning

    async def _select_action_phase(self, reasoning: ReasoningTool) -> BaseTool:
        phase_id = f"{self._context.iteration}-action"
        completion = None
        tool = None
        max_retries = 2
        for attempt in range(1, max_retries + 2):
            messages = await self._prepare_context()
            if attempt > 1:
                messages = messages + [
                    {
                        "role": "system",
                        "content": "RETRY: return exactly one valid tool call with parseable arguments.",
                    }
                ]
            tools = await self._prepare_tools()
            self._log_llm_call(
                phase="action_select",
                request_payload={
                    "attempt": attempt,
                    "messages": messages,
                    "tools_count": len(tools),
                    "tool_choice": self.tool_choice,
                    "llm_kwargs": self.config.llm.to_openai_client_kwargs(),
                },
            )
            async with self.openai_client.chat.completions.stream(
                messages=messages,
                tools=tools,
                tool_choice=self.tool_choice,
                **self.config.llm.to_openai_client_kwargs(),
            ) as stream:
                async for event in stream:
                    if event.type == "chunk":
                        self.streaming_generator.add_chunk(event.chunk, phase_id)
                completion = await stream.get_final_completion()
            self._log_llm_call(
                phase="action_select",
                request_payload={"attempt": attempt, "messages_count": len(messages), "tools_count": len(tools)},
                response_payload=getattr(completion, "model_dump", lambda **_: {"raw": str(completion)})(mode="json"),
            )
            try:
                tool = completion.choices[0].message.tool_calls[0].function.parsed_arguments
            except (IndexError, AttributeError, TypeError):
                tool = None
            if isinstance(tool, BaseTool):
                break

        if not isinstance(tool, BaseTool):
            final_content = ""
            if completion is not None:
                try:
                    final_content = completion.choices[0].message.content or ""
                except Exception:
                    final_content = ""
            if not final_content:
                final_content = "Task could not be completed due to invalid model tool output."
            self._log_llm_call(
                phase="action_select",
                request_payload={"attempt": "final_fallback"},
                response_payload=getattr(completion, "model_dump", lambda **_: {"raw": str(completion)})(mode="json")
                if completion is not None
                else None,
                error="action_parsed_arguments_none_after_retries",
            )
            tool = FinalAnswerTool(
                reasoning="Fallback after invalid/empty action tool payload",
                completed_steps=["Model returned invalid action tool payload"],
                answer=final_content,
                status=AgentStatesEnum.FAILED,
            )
        if not isinstance(tool, BaseTool):
            raise ValueError("Selected tool is not a valid BaseTool instance")

        self.conversation.append(
            {
                "role": "assistant",
                "content": reasoning.remaining_steps[0] if reasoning.remaining_steps else "Completing",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": phase_id,
                        "function": {
                            "name": tool.tool_name,
                            "arguments": tool.model_dump_json(),
                        },
                    }
                ],
            }
        )
        self.streaming_generator.add_tool_call(phase_id, tool)
        return tool

    async def _action_phase(self, tool: BaseTool) -> str:
        phase_id = f"{self._context.iteration}-action"
        result = await tool(self._context, self.config, **self.tool_configs.get(tool.tool_name, {}))
        self.conversation.append({"role": "tool", "content": result, "tool_call_id": phase_id})
        self.streaming_generator.add_tool_result(phase_id, result, tool.tool_name)
        self._log_tool_execution(tool, result)
        return result
