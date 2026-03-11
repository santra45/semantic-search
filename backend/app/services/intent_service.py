from google import genai
from pydantic import BaseModel
from typing import Optional, List
from backend.app.config import GEMINI_API_KEY
import json
import re

client = genai.Client(api_key=GEMINI_API_KEY)

class SearchIntent(BaseModel):
    clean_query: str
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    only_in_stock: bool = False
    attributes: Optional[dict] = {}

def analyze_intent(user_query: str) -> SearchIntent:
    prompt = f"""
    You are a search intent analyzer for an e-commerce store. 
    Extract filters from the user query.
    
    Query: "{user_query}"
    
    Return ONLY a JSON object with:
    - "clean_query": The core product they want (e.g., "running shoes")
    - "min_price": number or null
    - "max_price": number or null
    - "only_in_stock": boolean (true if they say "available" or "in stock")
    - "attributes": dictionary of other traits (e.g., {{"color": "red"}})
    """
    
    # Using the Generate Content method to get structured data
    response = client.models.generate_content(
        model="gemma-3-27b-it", # Use the flash model for speed
        contents=prompt
    )

    text = response.text.strip()

    # Extract JSON from the response
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        json_text = json_match.group(0)
    else:
        json_text = text
    
    data = json.loads(json_text)
    return SearchIntent(**data)