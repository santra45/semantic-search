from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from backend.app.services.license_service import validate_license_key
from backend.app.services.llm_key_service import decrypt_key
from backend.app.services.database import get_db
import time

# Import LLM providers for testing
from google import genai
from openai import OpenAI
import anthropic

router = APIRouter()

def test_llm_api_key(provider: str, api_key: str, model: str = None) -> bool:
    """
    Test if LLM API key is working by making a simple request to the provider.
    """
    try:
        if provider == "openai":
            client = OpenAI(api_key=api_key)
            client.models.list()  # simplest validity check
            return True

        elif provider == "anthropic":
            client = anthropic.Anthropic(api_key=api_key)
            client.models.list()
            return True

        elif provider == "gemini":
            client = genai.Client(api_key=api_key)
            client.models.list()  # works in new SDK
            return True

        else:
            print(f"❌ Unsupported provider: {provider}")
            return False

    except Exception as e:
        print(f"❌ API key invalid for {provider}: {e}")
        return False

class TestConnectionRequest(BaseModel):
    license_key: str
    llm_api_key_encrypted: str = None
    llm_provider: str = None
    llm_model: str = None

class TestConnectionResponse(BaseModel):
    success: bool
    message: str
    api_version: str = "1.0.0"
    timestamp: str
    llm_configured: bool = False
    llm_working: bool = False

@router.post("/test-connection")
async def test_connection(
    req: TestConnectionRequest, 
    db: Session = Depends(get_db)
):
    """
    Test API connection without affecting search quotas or caches.
    Validates license key and optionally tests LLM API key functionality.
    """
    start_time = time.time()
    
    try:
        # Validate license key (this doesn't count towards usage)
        license_data = validate_license_key(req.license_key, db)
        
        # Test LLM API key decryption and functionality if provided
        llm_configured = False
        llm_working = False
        
        if req.llm_api_key_encrypted and req.llm_provider:
            try:
                decrypted_key = decrypt_key(req.llm_api_key_encrypted, req.license_key)
                llm_configured = bool(decrypted_key)
                
                # Test actual API functionality if decryption succeeded
                if llm_configured:
                    llm_working = test_llm_api_key(
                        req.llm_provider, 
                        decrypted_key, 
                        req.llm_model
                    )
                    
            except Exception as e:
                # Log error but don't fail the connection test
                print(f"❌ LLM API key test failed: {e}")
                llm_configured = False
                llm_working = False
        
        response_time = int((time.time() - start_time) * 1000)
        
        # Build status message
        status_parts = [f"Connection successful in {response_time}ms"]
        if llm_configured:
            if llm_working:
                status_parts.append("LLM API working")
            else:
                status_parts.append("LLM API configured but not working")
        else:
            status_parts.append("LLM not configured")
        
        return TestConnectionResponse(
            success=True,
            message=" — ".join(status_parts),
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            llm_configured=llm_configured,
            llm_working=llm_working
        )
        
    except ValueError as e:
        # License validation failed
        return TestConnectionResponse(
            success=False,
            message=f"License validation failed: {str(e)}",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            llm_configured=False,
            llm_working=False
        )
        
    except Exception as e:
        # Unexpected error
        return TestConnectionResponse(
            success=False,
            message=f"Connection test failed: {str(e)}",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            llm_configured=False,
            llm_working=False
        )

@router.get("/health")
async def health_check():
    """
    Simple health check endpoint that doesn't require authentication.
    """
    return {
        "status": "healthy",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "api_version": "1.0.0"
    }
