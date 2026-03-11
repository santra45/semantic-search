import re
import asyncio
import aiohttp
from typing import List, Dict, Optional
import os
from backend.app.config import WORDPRESS_API_URL, FALLBACK_SCORE_THRESHOLD

# Common English stop words to filter out
STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
    'to', 'was', 'will', 'with', 'i', 'you', 'your', 'we', 'our',
    'they', 'them', 'their', 'this', 'these', 'those', 'or', 'but',
    'not', 'no', 'can', 'could', 'would', 'should', 'have', 'had',
    'what', 'when', 'where', 'why', 'how', 'who', 'which', 'if',
    'do', 'does', 'did', 'get', 'got', 'go', 'went', 'come', 'came'
}

def extract_keywords(query: str) -> List[str]:
    """
    Extract meaningful keywords from a search query.
    Removes stop words and punctuation, returns lowercase keywords.
    """
    # Convert to lowercase and remove punctuation
    cleaned = re.sub(r'[^\w\s]', ' ', query.lower())
    
    # Split into words and filter out stop words
    words = [word.strip() for word in cleaned.split() if word.strip() and word.strip() not in STOP_WORDS]
    
    # Remove duplicates while preserving order
    seen = set()
    keywords = []
    for word in words:
        if word not in seen and (len(word) > 1 or word.isdigit()):  # Allow single digits
            seen.add(word)
            keywords.append(word)
    
    return keywords

async def search_wordpress_fallback(
    client_id: str, 
    query: str, 
    license_key: str,
    api_url: Optional[str] = None,
    limit: int = 10
) -> List[Dict]:
    """
    Search WordPress using keyword-based fallback when semantic search scores are low.
    
    Args:
        client_id: Client identifier
        query: Original search query
        license_key: License key for authentication
        api_url: WordPress site URL (falls back to config)
        limit: Maximum number of results
        
    Returns:
        List of products in same format as Qdrant results
    """
    if not api_url:
        api_url = WORDPRESS_API_URL or os.getenv('WORDPRESS_API_URL', 'http://127.0.0.1/wordpress')
    
    keywords = extract_keywords(query)
    
    if not keywords:
        return []
    
    # Prepare WordPress API request
    endpoint = f"{api_url}/wp-json/ssw/v1/search-fallback"
    
    payload = {
        'license_key': license_key,
        'keywords': keywords,
        'query': query,  # Keep original query for reference
        'limit': limit
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.json()
                    return format_wordpress_results(data.get('results', []))
                else:
                    print(f"WordPress fallback API error: {response.status}")
                    return []
    except Exception as e:
        print(f"WordPress fallback search failed: {e}")
        return []

def format_wordpress_results(products: List[Dict]) -> List[Dict]:
    """
    Format WordPress product results to match Qdrant result format.
    """
    formatted_results = []
    
    for product in products:
        # Extract image URL
        image_url = ""
        if product.get('images'):
            images = product['images']
            if isinstance(images, list) and images:
                image_url = images[0].get('src', '')
            elif isinstance(images, str):
                image_url = images
        
        # Handle categories
        categories = ""
        if product.get('categories'):
            cats = product['categories']
            if isinstance(cats, list):
                categories = ", ".join([cat.get('name', str(cat)) for cat in cats])
            else:
                categories = str(cats)
        
        formatted = {
            "product_id": product.get('id', ''),
            "name": product.get('name', ''),
            "price": float(product.get('price', 0)),
            "permalink": product.get('permalink', ''),
            "image_url": image_url,
            "stock_status": product.get('stock_status', 'instock'),
            "categories": categories,
            "score": 0.0  # Fallback results don't have semantic scores
        }
        
        formatted_results.append(formatted)
    
    return formatted_results

def should_trigger_fallback(results: List[Dict], threshold: float = None) -> bool:
    """
    Determine if fallback search should be triggered based on semantic search scores.
    
    Args:
        results: Qdrant search results
        threshold: Score threshold (defaults to 0.58)
        
    Returns:
        True if fallback should be triggered
    """
    if threshold is None:
        threshold = float(FALLBACK_SCORE_THRESHOLD or os.getenv('FALLBACK_SCORE_THRESHOLD', 0.58))
    
    # Trigger fallback if no results or all scores are below threshold
    if not results:
        return True
    
    max_score = max(result.get('score', 0) for result in results)
    return max_score < threshold
