"""
LangGraph orchestrator — ports magento_chatbot/agents/orchestrator.py for the
multi-tenant semantic-search backend.

Flow per request:
  START → router (keyword / semantic intent)
       → agent (LLM with pre-bound per-agent tool list)
       → tools (execute, inject customer_id / quote_id)
       → back to agent (loop) until no more tool calls
       → END

Per-request state carries message list, current agent choice, customer/store
context, and the raw user query (resilient to message restoration from the
Postgres checkpointer).
"""

from __future__ import annotations

import logging
import operator
import re
from typing import Annotated, List, Optional, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph

from backend.app.magento.chatbot.agents.llm_factory import build_llm
from backend.app.magento.chatbot.agents.prompts import AGENT_PROMPTS
from backend.app.magento.chatbot.agents.request_context import RequestContext
from backend.app.magento.chatbot.agents.tools import (
    DIRECT_RETURN_TOOLS,
    build_tools_by_agent,
)
from backend.app.magento.chatbot.services.config import MAX_CHAT_HISTORY_MESSAGES
from backend.app.magento.chatbot.services.intent_router import get_intent_router

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    current_agent: str
    user_query: str


SKU_PATTERN = re.compile(r"\b[A-Z]{1,4}\d{1,5}(?:-[A-Za-z0-9]+)+\b", re.IGNORECASE)

QTY_UPDATE_PATTERNS = [
    re.compile(r"(?:update|change|set|modify)\s+(?:the\s+)?(?:qty|quantity)", re.I),
    re.compile(r"(?:update|change|set|modify)\s+\d+\s+(?:qty|quantity)", re.I),
    re.compile(r"(?:qty|quantity)\s+(?:to\s+)?\d+", re.I),
]

PROFILE_KEYWORDS = [
    "my profile", "my email", "my name", "my phone", "my number",
    "my address", "my addresses", "my details", "my account",
    "personal details", "contact info", "billing address", "shipping address",
    "phone number", "email id", "email address",
]
ORDER_KEYWORDS = [
    "order", "orders", "my order", "order history", "track",
    "delivery", "last order", "recent order", "order details", "shipping status", "tracking",
]
CART_KEYWORDS = [
    "cart", "add to cart", "remove from cart", "view cart", "checkout",
    "buy", "purchase", "shopping cart", "my cart", "basket",
    "clear cart", "empty cart",
]


class AgentOrchestrator:
    def __init__(self, ctx: RequestContext):
        self.ctx = ctx
        self.llm = build_llm(
            provider=ctx.llm_provider,
            model=ctx.llm_model,
            api_key=ctx.llm_api_key,
        )

        tools_by_agent = build_tools_by_agent(ctx)
        self._tools_by_agent = tools_by_agent
        self._all_tools = [t for tools in tools_by_agent.values() for t in tools]
        self._tool_map = {t.name: t for t in self._all_tools}

        self._llm_by_agent = {
            name: self.llm.bind_tools(tools) for name, tools in tools_by_agent.items()
        }

        self._token_usage = {"input": 0, "output": 0, "total": 0}

    # ── node: router ────────────────────────────────────────────────────────

    def _route(self, state: AgentState) -> dict:
        query = (state.get("user_query") or "").lower().strip()
        if not query:
            for msg in reversed(state.get("messages") or []):
                if isinstance(msg, HumanMessage):
                    query = (msg.content or "").lower().strip()
                    break

        has_sku = bool(SKU_PATTERN.search(query))
        is_qty_update = any(p.search(query) for p in QTY_UPDATE_PATTERNS)
        has_cart_verb = any(v in query for v in ("add", "remove", "delete", "update", "change qty", "quantity", "qty", "clear cart", "empty cart"))

        if any(kw in query for kw in ORDER_KEYWORDS):
            agent = "order"
        elif any(kw in query for kw in PROFILE_KEYWORDS):
            agent = "profile"
        elif any(kw in query for kw in CART_KEYWORDS) or (has_sku and has_cart_verb) or is_qty_update:
            agent = "cart"
        else:
            agent = "product"

        return {"current_agent": agent}

    # ── node: agent (LLM) ───────────────────────────────────────────────────

    async def _agent_node(self, state: AgentState) -> dict:
        current_agent = state.get("current_agent") or "product"
        messages = list(state.get("messages") or [])
        user_query = (state.get("user_query") or "").strip()

        # Auto-chain: when a tool returned "FOUND_PRODUCTS: ... SKUs: ...", force
        # the details lookup so the LLM doesn't regurgitate raw SKUs.
        if messages and isinstance(messages[-1], ToolMessage):
            content = str(messages[-1].content or "")
            if "FOUND_PRODUCTS:" in content and "SKUs:" in content:
                m = re.search(r"SKUs:\s*(.+?)(?:\n|$)", content, re.S)
                if m:
                    forced = AIMessage(
                        content="",
                        tool_calls=[{
                            "name": "get_product_details_by_skus",
                            "args": {"skus": m.group(1).strip()},
                            "id": "auto_product_details",
                        }],
                    )
                    return {"messages": [forced]}

        # Semantic intent router — fire on the first pass only.
        is_first_pass = bool(messages) and isinstance(messages[-1], HumanMessage)
        if is_first_pass and user_query:
            try:
                tool_name, detected_agent = get_intent_router().classify(
                    user_query, llm_api_key=self.ctx.llm_api_key
                )
            except Exception:
                tool_name, detected_agent = None, None

            if tool_name and detected_agent == current_agent:
                # Zero-arg intents → force the tool.
                if tool_name in {
                    "get_customer_email", "get_customer_name", "get_customer_phone",
                    "get_customer_addresses", "get_customer_profile",
                    "view_cart", "clear_cart",
                    "get_customer_orders", "get_last_order",
                }:
                    forced = AIMessage(
                        content="",
                        tool_calls=[{"name": tool_name, "args": {}, "id": f"{tool_name}_intent"}],
                    )
                    return {"messages": [forced]}
                if tool_name == "find_products":
                    forced = AIMessage(
                        content="",
                        tool_calls=[{
                            "name": tool_name,
                            "args": {"query": user_query},
                            "id": "find_products_intent",
                        }],
                    )
                    return {"messages": [forced]}

        # Build the message list for the LLM: system prompt + trimmed history + current turn.
        last_human_idx = 0
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], HumanMessage):
                last_human_idx = i
                break

        previous = []
        for msg in messages[:last_human_idx]:
            if isinstance(msg, HumanMessage):
                previous.append(msg)
            elif isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
                previous.append(msg)
        if len(previous) > MAX_CHAT_HISTORY_MESSAGES:
            previous = previous[-MAX_CHAT_HISTORY_MESSAGES:]

        current_turn = messages[last_human_idx:]
        chat_context = [SystemMessage(content=AGENT_PROMPTS[current_agent])] + previous + current_turn

        llm_with_tools = self._llm_by_agent[current_agent]
        response = await llm_with_tools.ainvoke(chat_context)

        # Accumulate tokens for persistence later.
        usage = getattr(response, "usage_metadata", None) or {}
        self._token_usage["input"] += int(usage.get("input_tokens") or 0)
        self._token_usage["output"] += int(usage.get("output_tokens") or 0)
        self._token_usage["total"] += int(usage.get("total_tokens") or (usage.get("input_tokens", 0) + usage.get("output_tokens", 0)))

        return {"messages": [response]}

    # ── node: tools ─────────────────────────────────────────────────────────

    async def _tool_node(self, state: AgentState) -> dict:
        messages = state.get("messages") or []
        last = messages[-1]
        if not getattr(last, "tool_calls", None):
            return {"messages": []}

        results: list[ToolMessage] = []
        executed: set[str] = set()

        for tc in last.tool_calls:
            name = tc["name"]
            args = dict(tc.get("args") or {})
            tool_id = tc.get("id", name)

            # Dedupe duplicate cart operations (esp. add_to_cart).
            if name in {"add_to_cart", "add_product_by_name"}:
                has_name_variant = any(other.get("name") == "add_product_by_name" for other in last.tool_calls)
                if name == "add_to_cart" and has_name_variant:
                    results.append(ToolMessage(content="", tool_call_id=tool_id, name=name))
                    continue
                if "add_to_cart" in executed or "add_product_by_name" in executed:
                    results.append(ToolMessage(content="", tool_call_id=tool_id, name=name))
                    continue

            tool = self._tool_map.get(name)
            if not tool:
                results.append(ToolMessage(content=f"Unknown tool: {name}", tool_call_id=tool_id, name=name))
                continue

            try:
                out = await tool.ainvoke(args)
            except Exception as exc:
                logger.exception("tool %s failed: %s", name, exc)
                out = f"Error running {name}: {exc}"
            executed.add(name)
            results.append(ToolMessage(content=str(out), tool_call_id=tool_id, name=name))

        return {"messages": results}

    def _should_continue(self, state: AgentState) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("router", self._route)
        graph.add_node("agent", self._agent_node)
        graph.add_node("tools", self._tool_node)
        graph.add_edge(START, "router")
        graph.add_edge("router", "agent")
        graph.add_conditional_edges("agent", self._should_continue, {"tools": "tools", END: END})
        graph.add_edge("tools", "agent")
        return graph

    # ── public entry ────────────────────────────────────────────────────────

    async def chat(
        self,
        query: str,
        *,
        thread_id: str,
        history: Optional[List[BaseMessage]] = None,
        checkpointer=None,
    ) -> dict:
        """Execute one chat turn and return {content, current_agent, token_usage, direct_tool}.

        `direct_tool` (if not None) is the name of a DIRECT_RETURN_TOOLS tool whose raw
        output should be surfaced to the user — bypassing the LLM final response.
        """
        base_messages: list[BaseMessage] = list(history or [])
        base_messages.append(HumanMessage(content=query))

        workflow = self._build_graph()
        compiled = workflow.compile(checkpointer=checkpointer) if checkpointer else workflow.compile()

        initial_state: AgentState = {
            "messages": base_messages,
            "current_agent": "product",
            "user_query": query,
        }
        config = {"configurable": {"thread_id": thread_id}}
        final = await compiled.ainvoke(initial_state, config)
        final_messages = final.get("messages", [])

        current_turn_start = 0
        for i in range(len(final_messages) - 1, -1, -1):
            msg = final_messages[i]
            if isinstance(msg, HumanMessage) and (msg.content or "") == query:
                current_turn_start = i
                break
        current_turn = final_messages[current_turn_start:]

        direct_tool_name = None
        direct_tool_output = None
        for msg in current_turn:
            if isinstance(msg, ToolMessage) and msg.name in DIRECT_RETURN_TOOLS and msg.content:
                direct_tool_name = msg.name
                direct_tool_output = msg.content
                break

        final_ai = None
        for msg in reversed(current_turn):
            if isinstance(msg, AIMessage) and msg.content:
                final_ai = msg.content
                break

        content = direct_tool_output or final_ai or "I'm here to help — what would you like to know?"

        return {
            "content": content,
            "current_agent": final.get("current_agent") or "product",
            "token_usage": dict(self._token_usage),
            "direct_tool": direct_tool_name,
            "final_messages": final_messages,
        }
