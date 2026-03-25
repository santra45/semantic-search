import json
import time
import re
from typing import List, Dict, Optional

from google import genai
from openai import OpenAI
import anthropic

def extract_json_array(text: str):
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]
    return None

def llm_rerank_products(
    query: str,
    products: List[Dict],
    limit: int = 10,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_api_key: Optional[str] = None,
) -> List[Dict]:

    provider = llm_provider or "gemini"

    if provider == "gemini":
        model = llm_model or "gemini-1.5-flash"
    elif provider == "openai":
        model = llm_model or "gpt-4o-mini"
    elif provider == "anthropic":
        model = llm_model or "claude-3-5-haiku-20241022"
    else:
        model = llm_model or "gemini-1.5-flash"

    api_key = llm_api_key

    if not products:
        return []

    if not api_key:
        return products[:limit]

    product_summaries = []
    product_map = {}

    for product in products[:25]:
        p_id = str(product.get("product_id") or product.get("id"))
        if not p_id:
            continue

        summary = {
            "id": p_id,
            "name": product.get("name", ""),
            "category": product.get("categories", ""),
            "price": product.get("price", 0),
        }

        product_summaries.append(summary)
        product_map[p_id] = product

    prompt = f"""
    You are an expert e-commerce product recommendation assistant.

    Customer query:
    {query}

    Products:
    {json.dumps(product_summaries)}

    Task:
    - Select ONLY relevant products
    - Ignore wrong category/gender
    - Prefer exact matches over partial
    - Rank by relevance (best first)

    Return ONLY JSON array of product IDs.
    Example: ["123", "456"] or []
    """

    try:
        if provider == "gemini":
            gemini_client = genai.Client(api_key=api_key)
            response = gemini_client.models.generate_content(
                model=model,
                contents=prompt,
            )
            response_text = response.text.strip()

        elif provider == "openai":
            openai_client = OpenAI(api_key=api_key)
            response = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            response_text = response.choices[0].message.content.strip()

        elif provider == "anthropic":
            anthropic_client = anthropic.Anthropic(api_key=api_key)
            response = anthropic_client.messages.create(
                model=model,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = response.content[0].text.strip()

        else:
            return products[:limit]

        json_text = extract_json_array(response_text)

        if json_text:
            relevant_ids = json.loads(json_text)
            relevant_products = []
            for p_id in relevant_ids:
                p_id_str = str(p_id)
                if p_id_str in product_map:
                    relevant_products.append(product_map[p_id_str])

            if relevant_products:
                return relevant_products[:limit]

        return []

    except Exception:
        return products[:limit]


def should_use_llm_reranking(query: str, products: List[Dict]) -> bool:
    simple_indicators = ["shirt", "pants", "dress", "shoes", "bag", "watch"]
    query_lower = query.lower()

    if any(indicator in query_lower for indicator in simple_indicators) and len(query.split()) <= 2:
        return False

    if len(products) > 5 or len(query.split()) > 3:
        return True

    complex_indicators = ["for", "with", "that", "which", "under", "over", "between", "size", "color", "material"]

    if any(indicator in query_lower for indicator in complex_indicators):
        return True

    return False