import json
import logging
import httpx
import tiktoken
from typing import List, Dict, Optional

from google import genai
from openai import OpenAI
import anthropic
from backend.app.services.token_usage_service import track_usage

# ---------------------------
# Logger Setup
# ---------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("llm_logger")


# ---------------------------
# HTTP Event Hooks (OpenAI & Anthropic)
# ---------------------------
def log_request(request: httpx.Request):
    body = request.content.decode("utf-8", errors="replace")
    headers = dict(request.headers)
    if "authorization" in headers:
        headers["authorization"] = "Bearer sk-***MASKED***"
    if "x-api-key" in headers:
        headers["x-api-key"] = "***MASKED***"

    try:
        body_formatted = json.dumps(json.loads(body), indent=2)
    except Exception:
        body_formatted = body

    logger.debug(
        f"\n{'='*60}\n"
        f"📤 REQUEST\n"
        f"  → {request.method} {request.url}\n"
        f"  Headers: {json.dumps(headers, indent=2)}\n"
        f"  Body:\n{body_formatted}\n"
        f"{'='*60}"
    )


def log_response(response: httpx.Response):
    response.read()  # Ensure body is fully loaded
    try:
        body_formatted = json.dumps(json.loads(response.text), indent=2)
    except Exception:
        body_formatted = response.text

    logger.debug(
        f"\n{'='*60}\n"
        f"📥 RESPONSE\n"
        f"  ← Status: {response.status_code}\n"
        f"  Body:\n{body_formatted}\n"
        f"{'='*60}"
    )


def make_http_client() -> httpx.Client:
    return httpx.Client(
        event_hooks={
            "request": [log_request],
            "response": [log_response],
        }
    )


# ---------------------------
# Gemini Logger (manual)
# ---------------------------
def log_gemini_request(model: str, prompt: str):
    logger.debug(
        f"\n{'='*60}\n"
        f"📤 GEMINI REQUEST\n"
        f"  Model: {model}\n"
        f"  Prompt:\n{prompt}\n"
        f"{'='*60}"
    )


def log_gemini_response(response):
    try:
        text = response.text
    except Exception:
        text = str(response)

    try:
        metadata = str(response.usage_metadata)
    except Exception:
        metadata = "N/A"

    logger.debug(
        f"\n{'='*60}\n"
        f"📥 GEMINI RESPONSE\n"
        f"  Text:\n{text}\n"
        f"  Usage Metadata: {metadata}\n"
        f"{'='*60}"
    )


# ---------------------------
# Helper: Extract JSON array
# ---------------------------
def extract_json_array(text: str):
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return None


# ---------------------------
# Helper: Estimate tokens (fallback)
# ---------------------------
def estimate_tokens(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    token_count = len(tokens)
    return token_count


# ---------------------------
# Helper: Get token usage
# ---------------------------
def get_token_usage(provider, response, prompt, response_text):
    try:
        if provider == "openai":
            usage = response.usage
            return {
                "input": usage.prompt_tokens,
                "output": usage.completion_tokens,
                "total": usage.total_tokens,
            }

        elif provider == "anthropic":
            usage = response.usage
            return {
                "input": usage.input_tokens,
                "output": usage.output_tokens,
                "total": usage.input_tokens + usage.output_tokens,
            }

        elif provider == "gemini":
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                metadata = response.usage_metadata
                input_tokens = getattr(metadata, "prompt_token_count", 0) or 0
                output_tokens = getattr(metadata, "candidates_token_count", 0) or 0
                total_tokens = getattr(metadata, "total_token_count", 0) or 0

                # candidates_token_count is None in some Gemini responses,
                # so derive output from total - input, fallback to estimating from text
                candidates = getattr(metadata, "candidates_token_count", None)
                if candidates is not None:
                    output_tokens = candidates
                elif total_tokens > input_tokens:
                    output_tokens = total_tokens - input_tokens  # derived
                elif output_tokens == 0 and response_text:
                    output_tokens = estimate_tokens(response_text)
                else:
                    output_tokens = estimate_tokens(response_text)  # last resort fallback

                return {
                    "input": input_tokens,
                    "output": output_tokens,
                    "total": input_tokens + output_tokens,  # recalculate proper total
                }
            else:
                return {
                    "input": estimate_tokens(prompt),
                    "output": estimate_tokens(response_text),
                    "total": estimate_tokens(prompt + response_text),
                }

    except Exception as e:
        logger.warning(f"⚠️ Token usage extraction failed: {e}")
        return {
            "input": estimate_tokens(prompt),
            "output": estimate_tokens(response_text),
            "total": estimate_tokens(prompt + response_text),
        }


# ---------------------------
# Optional: Cost calculation
# ---------------------------
MODEL_PRICING = {
    # ── Gemini ────────────────────────────────────────────────────────────────
    "gemini-3.1-pro-preview":  {"input": 0.000002,    "output": 0.000012},
    "gemini-2.5-pro":          {"input": 0.00000125,  "output": 0.00001},
    "gemini-2.5-flash":        {"input": 0.0000003,   "output": 0.0000025},
    "gemini-2.5-flash-lite":   {"input": 0.0000001,   "output": 0.0000004},
    "gemma-3-27b-it":          {"input": 0.00000008,  "output": 0.00000016},

    # ── OpenAI ────────────────────────────────────────────────────────────────
    "gpt-5.4":                 {"input": 0.0000025,   "output": 0.000015},
    "gpt-5.4-mini":            {"input": 0.00000075,  "output": 0.0000045},
    "gpt-5.4-nano":            {"input": 0.0000002,   "output": 0.00000125},
    "gpt-5.2":                 {"input": 0.00000175,  "output": 0.000014},

    # ── Anthropic ─────────────────────────────────────────────────────────────
    "claude-opus-4-6":         {"input": 0.000005,    "output": 0.000025},
    "claude-sonnet-4-6":       {"input": 0.000003,    "output": 0.000015},
    "claude-haiku-4-5-20251001": {"input": 0.000001,  "output": 0.000005},
    "claude-3-5-sonnet-20241022": {"input": 0.000003, "output": 0.000015},
}

def estimate_cost(model: str, usage: Dict) -> float:
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        return 0.0
    return (
        usage["input"] * pricing["input"] +
        usage["output"] * pricing["output"]
    )


# ---------------------------
# Main Function
# ---------------------------
def llm_rerank_content(
    query: str,
    content: List[Dict],
    limit: int = 10,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    client_id: str = "anonymous"
) -> List[Dict]:

    provider = llm_provider or "gemini"

    if provider == "gemini":
        model = llm_model or "gemini-1.5-flash"
    elif provider == "openai":
        model = llm_model or "gpt-4o-mini"
    elif provider == "anthropic":
        model = llm_model or "claude-3-5-haiku-20241022"
    else:
        model = "gemini-1.5-flash"

    api_key = llm_api_key

    if not content:
        return []

    if not api_key:
        logger.warning("⚠️ No API key provided. Returning top results without reranking.")
        return content[:limit]

    content_summaries = []
    content_map = {}

    # Process mixed content types (products, pages, posts)
    for item in content[:25]:
        content_type = item.get("content_type", "product")
        
        # Generate unique ID based on content type
        if content_type == "product":
            item_id = str(item.get("product_id") or item.get("id"))
        elif content_type == "page":
            item_id = f"page_{item.get('page_id')}"
        elif content_type == "post":
            item_id = f"post_{item.get('post_id')}"
        else:
            continue
        
        if not item_id:
            continue

        # Build summary based on content type
        if content_type == "product":
            summary = {
                "id": item_id,
                "type": "product",
                "name": item.get("name", ""),
                "category": item.get("categories", ""),
                "price": item.get("price", 0),
            }
        elif content_type == "page":
            summary = {
                "id": item_id,
                "type": "page",
                "title": item.get("title", ""),
                "excerpt": item.get("excerpt", "")[:200],  # Truncate excerpt
            }
        elif content_type == "post":
            summary = {
                "id": item_id,
                "type": "post",
                "title": item.get("title", ""),
                "excerpt": item.get("excerpt", "")[:200],
                "categories": item.get("categories", ""),
                "tags": item.get("tags", ""),
            }
        else:
            continue

        content_summaries.append(summary)
        content_map[item_id] = item

    prompt = f"""
    You are an expert content recommendation assistant for an e-commerce website.

    Customer query:
    {query}

    Available content (products, pages, blog posts):
    {json.dumps(content_summaries)}

    Task:
    - Select ONLY relevant content (products, pages, or posts)
    - For products: ignore wrong category/gender
    - For pages/posts: match the topic/theme of the query
    - Prefer exact matches over partial
    - Rank by relevance (best first)
    - Include a mix of content types if relevant

    Return ONLY JSON array of content IDs.
    Example: ["123", "page_456", "post_789"] or []
    """

    try:
        response_text = ""
        response = None

        # ---------------------------
        # GEMINI
        # ---------------------------
        if provider == "gemini":
            log_gemini_request(model, prompt)
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )
            log_gemini_response(response)
            response_text = response.text.strip()

        # ---------------------------
        # OPENAI
        # ---------------------------
        elif provider == "openai":
            client = OpenAI(
                api_key=api_key,
                http_client=make_http_client(),
            )
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            response_text = response.choices[0].message.content.strip()

        # ---------------------------
        # ANTHROPIC
        # ---------------------------
        elif provider == "anthropic":
            client = anthropic.Anthropic(
                api_key=api_key,
                http_client=make_http_client(),
            )
            response = client.messages.create(
                model=model,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = response.content[0].text.strip()

        else:
            return content[:limit]

        # ---------------------------
        # TOKEN USAGE & COST
        # ---------------------------
        usage = get_token_usage(provider, response, prompt, response_text)
        cost = estimate_cost(model, usage)

        logger.info(f"🔢 Token Usage: {usage}")
        logger.info(f"💰 Estimated Cost: ${round(cost, 8)}")
        
        # Track token usage
        try:
            track_usage(
                client_id=client_id,
                query_type="content_rerank",
                llm_provider=provider,
                llm_model=model,
                input_tokens=usage["input"],
                output_tokens=usage["output"],
                input_cost=usage["input"] * MODEL_PRICING.get(model, {}).get("input", 0),
                output_cost=usage["output"] * MODEL_PRICING.get(model, {}).get("output", 0),
                request_text_length=len(prompt),
                response_text_length=len(response_text)
            )
        except Exception as e:
            logger.warning(f"⚠️ Failed to track token usage: {e}")

        # ---------------------------
        # Parse & Return
        # ---------------------------
        json_text = extract_json_array(response_text)

        if json_text:
            relevant_ids = json.loads(json_text)
            relevant_content = []
            for item_id in relevant_ids:
                item_id_str = str(item_id)
                if item_id_str in content_map:
                    relevant_content.append(content_map[item_id_str])

            if relevant_content:
                return relevant_content[:limit]

        logger.warning("⚠️ No relevant content found after reranking.")
        return []

    except Exception as e:
        logger.error(f"❌ Error during LLM reranking: {str(e)}", exc_info=True)
        return content[:limit]


# ---------------------------
# Smart Trigger
# ---------------------------
def should_use_llm_reranking(query: str, content: List[Dict]) -> bool:
    simple_indicators = ["shirt", "pants", "dress", "shoes", "bag", "watch"]
    query_lower = query.lower()

    if any(indicator in query_lower for indicator in simple_indicators) and len(query.split()) <= 2:
        return False

    if len(content) > 5 or len(query.split()) > 3:
        return True

    complex_indicators = [
        "for", "with", "that", "which",
        "under", "over", "between",
        "size", "color", "material",
    ]

    if any(indicator in query_lower for indicator in complex_indicators):
        return True

    return False


# ---------------------------
# Backward Compatibility Wrapper
# ---------------------------
def llm_rerank_products(
    query: str,
    products: List[Dict],
    limit: int = 10,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    client_id: str = "anonymous"
) -> List[Dict]:
    """Backward compatibility wrapper for llm_rerank_content."""
    return llm_rerank_content(
        query=query,
        content=products,
        limit=limit,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        client_id=client_id
    )