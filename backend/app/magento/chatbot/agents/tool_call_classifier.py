"""
Tool-calling LLM intent router (Phase 3.1).

One LLM call, twelve tools, one selection. The Magento side reads the
returned `tool_name` + `arguments` and dispatches the corresponding
agent — same execution path as today's heuristic / LLM classifier, only
the *classification* step has changed.

Why this exists:
    The legacy stack runs a regex heuristic first, then an LLM JSON-mode
    fallback when confidence falls below 0.80. The regex is brittle
    against phrasing variations and the JSON-mode fallback re-implements
    schema enforcement that the providers' native tool-calling APIs
    already do for us. This module replaces both layers with one
    LangChain `bind_tools()` invocation that returns a structured
    tool_call.

What stays unchanged:
    * Agent execution (every Magento agent class is untouched)
    * Streaming pipeline (this runs before agent dispatch, never streamed)
    * Chip-driven structured actions (Chat.php's dispatchIntent path
      bypasses classification altogether)
    * Cost-tracking contract (we write to `token_usage_tracking` with
      `query_type='chat_tool_call'` so the admin dashboard sees the
      cost separately from the legacy `chat_intent`)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage 

from backend.app.magento.chatbot.agents.llm_factory import build_llm
from backend.app.magento.chatbot.agents.tools import ALL_TOOLS, TOOL_BY_NAME
from backend.app.services.token_usage_service import track_usage
from backend.app.utils.llm_logger import log_llm_interaction

# Reuse the rerank-service pricing table — it's the canonical per-model
# pricing tier used by every other LLM call in the backend. Keeps cost
# accounting consistent across `chat_intent`, `chat_tool_call`,
# `chat_answer`, `product_rerank`, etc.
from backend.app.services.llm_rerank_service import MODEL_PRICING, get_token_usage

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """
You are an e-commerce intent router.

Rules:

1. You MUST call exactly one tool.
2. Never answer the customer.
3. Prefer specific tools over generic ones.
4. Use conversation history to resolve references like:
   "that one"
   "the cheaper one"
   "tell me more"
5. Use general_chat ONLY if no other tool applies.
6. Use detected signals as authoritative hints.
7. Distinguish QUESTION vs BROWSE for product queries:
   - QUESTION (inquisitive — wants a text answer about products):
     "are your products X?", "do you have anything X?", "what's the
     X on your Y?", "is this safe for Z?", "can these be used X?",
     "how do I clean / care for these?", "what are these made of?"
     → answer_product_question
   - BROWSE (imperative — wants product cards to click):
     "show me X", "find X", "I'm looking for X", "I want a X",
     "cheapest X", "anything from <brand>", "products in <category>"
     → search_products
   The test: if a text answer with no product cards would satisfy
   the customer, route to answer_product_question.
8. Always prioritize the page_context if customer didn't include specific information.

"""


def _render_match_signals(signals: dict[str, Any]) -> str:
    """Turn the compact match_signals dict into a short natural-language
    block for the tool-call system prompt.

    Format:
        Detected matches in the customer's message (use these as hints
        for which tool to pick and what args to pass):
          - category: Solar Fountains (id: 47)
          - brand: Altico
          - attributes: color=Red, material=Stainless Steel

    Empty / missing sections are omitted. Returns empty string when no
    signal is present — caller skips appending the block entirely.
    """
    lines: list[str] = []

    category = signals.get("category")
    if isinstance(category, dict):
        cat_id = category.get("id")
        cat_name = (category.get("name") or "").strip()
        if cat_id and cat_name:
            lines.append(f"  - category: {cat_name} (id: {cat_id})")

    brand = (signals.get("brand") or "").strip() if isinstance(signals.get("brand"), str) else ""
    if brand:
        lines.append(f"  - brand: {brand}")

    attributes = signals.get("attributes") or {}
    if isinstance(attributes, dict) and attributes:
        rendered = ", ".join(
            f"{k}={v}"
            for k, v in attributes.items()
            if isinstance(k, str) and isinstance(v, str) and v.strip()
        )
        if rendered:
            lines.append(f"  - attributes: {rendered}")

    if not lines:
        return ""

    return (
        "Detected matches in the customer's message (use these as hints "
        "for which tool to pick and what args to pass — they are "
        "authoritative for THIS merchant's catalogue):\n"
        + "\n".join(lines)
    )


def select_tool(
    *,
    query: str,
    conversation_history: Optional[list[dict[str, str]]] = None,
    customer_context: Optional[dict[str, Any]] = None,
    match_signals: Optional[dict[str, Any]] = None,
    page_context: Optional[dict[str, Any]] = None,
    provider: str = "google",
    model: str = "gemini-2.5-flash",
    api_key: Optional[str] = None,
    client_id: str = "anonymous",
) -> dict[str, Any]:
    """Run one tool-calling LLM invocation; return the picked tool + args.

    Return shape (always a dict, never raises to the caller):

        {
            "tool_name":   str | None,       # picked tool, or None on hard-fail
            "arguments":   dict,             # may be empty
            "confidence":  float,            # 0.0 on fallback, 1.0 on real pick
            "usage":       {input, output, cost},
            "model":       str,
            "duration_ms": int,
            "error":       str | None,
        }

    Failures (LLM crash, no tool call returned, unknown tool name) fall
    back to {tool_name: "general_chat", confidence: 0.0, error: "..."}
    so the Magento Router can either still dispatch to the
    GenericChatAgent or — preferred — fall back to its heuristic
    classifier. The contract: this function NEVER throws.
    """
    conversation_history = conversation_history or []
    customer_context = customer_context or {}
    match_signals = match_signals or {}
    page_context = page_context or {}

    # Build a tiny context preamble for the system prompt — gives the
    # LLM enough situational awareness to route auth-gated tools sanely
    # (e.g. don't pick `manage_cart` for a brand-new guest with empty
    # context). Auth check itself still happens on the Magento side.
    ctx_lines = []
    if customer_context.get("is_logged_in"):
        name = (customer_context.get("customer_name") or "").strip()
        ctx_lines.append(
            f"The customer is logged in{f' (first name: {name})' if name else ''}."
        )
    else:
        ctx_lines.append("The customer is a guest (not logged in).")
    if customer_context.get("store_code"):
        ctx_lines.append(f"Active store view: {customer_context['store_code']}.")

    # Page context — the product / category the customer is viewing. This is
    # what lets the router resolve deictic questions ("do they freeze in
    # winter?", "is this in stock?") to a concrete product even though the
    # message names none.
    if page_context.get("type") == "product":
        name = (page_context.get("name") or "").strip()
        sku = (page_context.get("sku") or "").strip()
        label = name or sku
        if label:
            line = f"The customer is currently viewing the product: {label}"
            if sku:
                line += f" (SKU: {sku})"
            line += (
                ". If the customer refers to a product with words like 'this', "
                "'these', 'it', 'they', or asks about a product without naming "
                "one, assume they mean THIS product: route usage / attribute / "
                "care / safety questions to answer_product_question, and stock / "
                "price / variant questions to get_product_detail with this "
                "product's SKU in the `skus` argument."
            )
            ctx_lines.append(line)
    elif page_context.get("type") == "category":
        cat_name = (page_context.get("name") or "").strip()
        if cat_name:
            ctx_lines.append(f"The customer is currently browsing the category: {cat_name}.")

    context_block = "\n".join(ctx_lines)

    # Matched signals (structured filter rebuild 2026-05-22+).
    #
    # Magento ran BrandVocabulary + CategoryVocabulary + AttributeVocabulary
    # over the customer's message before calling us and shipped only the
    # MATCHES — not the entire vocabulary. We render them here as a short
    # natural-language hint the LLM can use to pick its tool + arguments
    # correctly without having to recognise merchant-specific names from
    # generic English (which cheap routing models like flash-lite are bad
    # at). Skipped entirely when no signal matched, keeping the prompt
    # minimal in the common case.
    signal_block = _render_match_signals(match_signals)

    # Prompt-prefix cacheability (2026-06-19): the SystemMessage is kept
    # byte-for-byte STATIC — only `_SYSTEM_PROMPT` — so that, together with
    # the bound tool schemas (also constant across requests), it forms a
    # stable ~5k-token prefix the provider can cache instead of re-processing
    # it every call. The per-request pieces (customer / page context and the
    # detected match-signals) therefore do NOT go in the system instruction;
    # they ride in the user turn instead, prefixed onto the current message.
    # The LLM sees identical information either way — only its position moved,
    # off the cacheable head and onto the variable tail.
    messages: list[Any] = [
        SystemMessage(content=_SYSTEM_PROMPT),
    ]
    for turn in conversation_history[-6:]:
        role = (turn.get("role") or "").strip().lower()
        content = (turn.get("content") or "").strip()
        if not content:
            continue

        if role == "user":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))

    # Current turn — per-request context + detected signals are prefixed onto
    # the customer's message so the system prefix above stays cache-stable.
    preamble_parts = [p for p in (context_block, signal_block) if p]
    if preamble_parts:
        final_user_content = (
            "\n\n".join(preamble_parts)
            + "\n\n---\nCustomer message: "
            + query
        )
    else:
        final_user_content = query
    messages.append(HumanMessage(content=final_user_content))

    t0 = time.perf_counter()
    response = None
    error_msg: Optional[str] = None

    try:
        llm = build_llm(
            provider=provider,
            model=model,
            api_key=api_key,
            # Routing is a deterministic task — sample at 0 temperature so
            # the same message produces the same tool selection across
            # retries / shadow-mode A/B logging.
            temperature=0.0,
        )
        # tool_choice="any" forces a tool call (LangChain >= 0.3 abstracts
        # the per-provider syntax — Gemini, OpenAI, and Anthropic all
        # accept it). When unsupported the LLM may return text only;
        # we treat that as a no-tool-picked fallback below.
        try:
            llm_with_tools = llm.bind_tools(ALL_TOOLS, tool_choice="any")
        except (TypeError, ValueError):
            # Older provider integrations don't accept tool_choice — fall
            # back to default behaviour where the LLM may still call a
            # tool, just not forced.
            llm_with_tools = llm.bind_tools(ALL_TOOLS)
        response = llm_with_tools.invoke(messages)
    except Exception as exc:
        error_msg = str(exc)
        logger.warning("[tool_call] LLM invoke failed: %s", error_msg)

    duration_ms = int((time.perf_counter() - t0) * 1000)

    # Extract the tool call.
    tool_calls = getattr(response, "tool_calls", None) or []
    picked_name: Optional[str] = None
    picked_args: dict[str, Any] = {}
    confidence = 0.0

    if tool_calls:
        first = tool_calls[0]
        # LangChain normalises tool_call shape across providers to
        # {name, args, id}. Defensive against future shape drift.
        picked_name = first.get("name") if isinstance(first, dict) else getattr(first, "name", None)
        picked_args = (first.get("args") if isinstance(first, dict) else getattr(first, "args", None)) or {}
        if picked_name and picked_name in TOOL_BY_NAME:
            # Confidence is binary for tool-call: the LLM picked a known
            # tool deterministically. We surface 1.0 to the Router so it
            # treats this as authoritative; downstream PHP can still
            # second-guess via shadow-mode disagreement logging.
            confidence = 1.0
        else:
            # Unknown tool name → treat as fallback.
            logger.warning("[tool_call] LLM returned unknown tool: %s", picked_name)
            picked_name = None

    # Hard fallback: no tool picked. Caller decides whether to fall back
    # to heuristic (preferred) or just dispatch general_chat. We return
    # general_chat as the safe default and let the Magento Router decide.
    if not picked_name:
        picked_name = "general_chat"
        picked_args = {"query": query}
        confidence = 0.0
        if not error_msg:
            error_msg = "no_tool_picked"

    # Token usage + cost.
    usage = {"input": 0, "output": 0, "total": 0}
    cost = 0.0
    if response is not None:
        try:
            # response.usage_metadata is a LangChain dict {input_tokens,
            # output_tokens, total_tokens}. Translate to the {input,
            # output, total} shape the rerank-service helpers expect.
            meta = getattr(response, "usage_metadata", None) or {}
            usage = {
                "input": int(meta.get("input_tokens", 0) or 0),
                "output": int(meta.get("output_tokens", 0) or 0),
                "total": int(meta.get("total_tokens", 0) or 0),
            }
            if not usage["total"]:
                usage["total"] = usage["input"] + usage["output"]
            pricing = MODEL_PRICING.get(model, {})
            cost = (
                usage["input"] * pricing.get("input", 0.0)
                + usage["output"] * pricing.get("output", 0.0)
            )
        except Exception as exc:
            logger.debug("[tool_call] usage extraction failed: %s", exc)

    # Persist for the admin dashboard cost view + per-tenant analytics.
    # Skipped silently on any DB hiccup (same posture as every other
    # track_usage call site — never DOS a request on a metrics write).
    try:
        track_usage(
            client_id=client_id,
            query_type="chat_tool_call",
            llm_provider=provider,
            llm_model=model,
            input_tokens=usage["input"],
            output_tokens=usage["output"],
            input_cost=usage["input"] * MODEL_PRICING.get(model, {}).get("input", 0.0),
            output_cost=usage["output"] * MODEL_PRICING.get(model, {}).get("output", 0.0),
            request_text_length=len(query),
            response_text_length=0,
        )
    except Exception as exc:
        logger.warning("[tool_call] usage tracking failed: %s", exc)

    # Single readable log line per call — pairs with the existing
    # `[rerank]` / chat-answer log lines so ops can grep one file to
    # reconstruct the full LLM cost of a turn.
    log_llm_interaction(
        provider=provider,
        model=model,
        purpose="chat_tool_call",
        prompt=query,
        response_text=f"tool={picked_name} args={picked_args}",
        input_tokens=usage["input"],
        output_tokens=usage["output"],
        cost=cost,
        client_id=client_id,
        duration_ms=duration_ms,
        error=error_msg,
        extra={
            "match_signals": match_signals,
            "history_count": len(conversation_history),
            "customer_context": customer_context,
            "tools_offered": len(ALL_TOOLS),
            "picked":        picked_name,
            "confidence":    confidence,
        },
    )

    return {
        "tool_name":   picked_name,
        "arguments":   picked_args,
        "confidence":  confidence,
        "usage":       {"input": usage["input"], "output": usage["output"], "cost": cost},
        "model":       model,
        "duration_ms": duration_ms,
        "error":       error_msg,
    }
