"""Customer Agent — AgentExecutor bridge between A2A SDK and LangGraph."""

from __future__ import annotations

import logging
from uuid import uuid4

from langchain_core.messages import HumanMessage

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Part, TextPart

from customer_agent.graph import build_graph

logger = logging.getLogger(__name__)


class CustomerAgentExecutor(AgentExecutor):
    """Bridges A2A RequestContext to the Customer LangGraph agent."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        question = self._extract_question(context)
        context_id = context.context_id or str(uuid4())
        task_id = context.task_id or str(uuid4())

        # Propagate or generate trace metadata
        metadata = context.message.metadata or {} if context.message else {}
        trace_id = metadata.get("trace_id", str(uuid4()))
        depth = int(metadata.get("delegation_depth", 0))

        logger.info(
            "CustomerAgent executing | task=%s context=%s trace=%s depth=%d",
            task_id, context_id, trace_id, depth,
        )

        updater = TaskUpdater(event_queue, task_id, context_id)
        await updater.submit()
        await updater.start_work()

        try:
            # Build a per-request graph so the tool closure captures this request's IDs
            graph = build_graph(
                trace_id=trace_id,
                context_id=context_id,
                depth=depth,
            )

            result = await graph.ainvoke(
                {"messages": [HumanMessage(content=question)]},
                config={"configurable": {"thread_id": context_id}},
            )

            # Strategy: prefer the raw ToolMessage content (the full Law Agent
            # specialist analysis) over the final AIMessage. Small local LLMs
            # (Ollama) tend to summarize tool results instead of reproducing them
            # verbatim, so we bypass that summary and return the actual analysis.
            from langchain_core.messages import AIMessage, ToolMessage

            messages = result.get("messages", [])

            # 1. Collect all ToolMessage contents (specialist analyses)
            tool_contents = [
                msg.content for msg in messages
                if isinstance(msg, ToolMessage) and msg.content
            ]

            # 2. Get the final AIMessage for context/intro
            ai_intro = ""
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.content:
                    # Only use it as intro if it's clearly not a full analysis
                    # (i.e. short wrapper text from the LLM)
                    if len(msg.content) < 500:
                        ai_intro = msg.content
                    break

            if tool_contents:
                # Use the specialist analysis as the main body
                specialist_analysis = "\n\n---\n\n".join(tool_contents)
                if ai_intro:
                    answer = f"{ai_intro}\n\n{specialist_analysis}"
                else:
                    answer = specialist_analysis
            else:
                # Fallback: use the final AIMessage
                answer = ""
                for msg in reversed(messages):
                    content = getattr(msg, "content", "")
                    if content and not isinstance(msg, HumanMessage):
                        answer = content
                        break

            if not answer:
                answer = "I was unable to process your legal question at this time."

            await updater.add_artifact(
                parts=[Part(root=TextPart(text=answer))],
                name="legal_response",
            )
            await updater.complete()

        except Exception as exc:
            logger.exception("CustomerAgent execution error: %s", exc)
            await updater.failed(
                updater.new_agent_message(
                    parts=[Part(root=TextPart(text=f"Request failed: {exc}"))]
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id or str(uuid4())
        context_id = context.context_id or str(uuid4())
        updater = TaskUpdater(event_queue, task_id, context_id)
        await updater.cancel()

    @staticmethod
    def _extract_question(context: RequestContext) -> str:
        if context.message and context.message.parts:
            parts = []
            for part in context.message.parts:
                inner = getattr(part, "root", part)
                text = getattr(inner, "text", None)
                if text:
                    parts.append(text)
            return "\n".join(parts)
        return ""