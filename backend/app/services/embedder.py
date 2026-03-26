from google import genai
from backend.app.config import GEMINI_API_KEY, EMBED_MODEL
# from sentence_transformers import SentenceTransformer
# import os

# local_model_path = "./local_models/bge-base-en-v1.5"
# model = SentenceTransformer(local_model_path)

def get_client(api_key: str = None):
    """Initialize Gemini client with provided API key or fallback to config."""
    key = api_key
    return genai.Client(api_key=key)

def embed_query(text: str, api_key: str = None) -> list[float]:
    """
    Embed a search query using the new SDK.
    """
    client = get_client(api_key)
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config={
            'task_type': 'RETRIEVAL_QUERY'
        }
    )
    # The new SDK returns an object, not a dictionary
    return result.embeddings[0].values


def embed_document(text: str, api_key: str = None) -> list[float]:
    """
    Embed a product document for indexing.
    """
    client = get_client(api_key)
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config={
            'task_type': 'RETRIEVAL_DOCUMENT'
        }
    )
    return result.embeddings[0].values

# def embed_query(text: str) -> list[float]:
#     """
#     Embed a search query locally (fast).
#     """
#     embedding = model.encode(text, normalize_embeddings=True)
#     return embedding.tolist()


# def embed_document(text: str) -> list[float]:
#     """
#     Embed a product document locally.
#     """
#     embedding = model.encode(text, normalize_embeddings=True)
#     return embedding.tolist()