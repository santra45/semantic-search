from fastapi import APIRouter, HTTPException, Request, Form, Depends
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr, HttpUrl
from typing import Optional
import re
from urllib.parse import urlparse

from backend.app.services.database import get_db
from backend.app.services.license_service import create_client, generate_license_key

router = APIRouter()

# Template setup
templates = Jinja2Templates(directory="backend/app/templates")

# Pydantic models for validation
class SignupRequest(BaseModel):
    name: str
    email: EmailStr
    company_name: Optional[str] = None
    store_name: str
    store_url: HttpUrl
    woo_version: str
    estimated_products: str
    plan: str = "starter"

class SignupResponse(BaseModel):
    success: bool
    license_key: Optional[str] = None
    client_id: Optional[str] = None
    error: Optional[str] = None

def extract_domain(url: str) -> str:
    """Extract domain from URL for license validation."""
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except:
        raise ValueError("Invalid URL format")

@router.get("/onboarding", response_class=HTMLResponse)
async def onboarding_page(request: Request):
    """Serve the main onboarding page."""
    return templates.TemplateResponse("onboarding.html", {"request": request})

@router.post("/api/onboarding/signup")
async def signup_client(
    name: str = Form(...),
    email: str = Form(...),
    company_name: Optional[str] = Form(None),
    store_name: str = Form(...),
    store_url: str = Form(...),
    woo_version: str = Form(...),
    estimated_products: str = Form(...),
    plan: str = Form("starter"),
    db: Session = Depends(get_db)
):
    """Process client signup and generate license key."""
    try:
        # Validate and extract domain
        domain = extract_domain(store_url)
        
        # Validate plan
        valid_plans = ["starter", "growth", "pro"]
        if plan not in valid_plans:
            plan = "starter"  # Default to starter for now
        
        # Create client
        client = create_client(db, name, email, plan)
        client_id = client["id"]
        
        # Generate license key
        license_key = generate_license_key(
            db=db,
            client_id=client_id,
            allowed_domain=domain,
            plan=plan,
            valid_days=365  # 1 year validity
        )
        
        return SignupResponse(
            success=True,
            license_key=license_key,
            client_id=client_id
        )
        
    except ValueError as e:
        return SignupResponse(
            success=False,
            error=str(e)
        )
    except Exception as e:
        return SignupResponse(
            success=False,
            error="Registration failed. Please try again."
        )

@router.get("/api/onboarding/plans")
async def get_plans():
    """Get available plans and their features."""
    return {
        "plans": {
            "starter": {
                "name": "Starter",
                "price": "$0",
                "period": "month",
                "product_limit": 500,
                "search_limit": 10000,
                "features": [
                    "Semantic search",
                    "Basic analytics",
                    "Email support",
                    "Up to 500 products"
                ]
            },
            "growth": {
                "name": "Growth", 
                "price": "$29",
                "period": "month",
                "product_limit": 5000,
                "search_limit": 100000,
                "features": [
                    "Everything in Starter",
                    "Advanced analytics",
                    "Priority support",
                    "Up to 5,000 products",
                    "Custom branding"
                ]
            },
            "pro": {
                "name": "Pro",
                "price": "$99", 
                "period": "month",
                "product_limit": 25000,
                "search_limit": 500000,
                "features": [
                    "Everything in Growth",
                    "Dedicated support",
                    "Custom integrations",
                    "Up to 25,000 products",
                    "White-label options"
                ]
            }
        }
    }

@router.get("/api/onboarding/documentation")
async def get_documentation():
    """Get documentation content for the tabs."""
    return {
        "quick_start": {
            "title": "Quick Start Guide",
            "content": """
## 5-Minute Setup

### Step 1: Install Plugin
1. Download the plugin ZIP file
2. Go to WordPress Admin → Plugins → Add New
3. Click "Upload Plugin" and select the ZIP file
4. Click "Install Now" then "Activate"

### Step 2: Enter License Key
1. Go to WooCommerce → Settings → Semantic Search
2. Enter your license key from this page
3. Set your API URL (if different from default)
4. Click "Save Changes"

### Step 3: Sync Products
1. Click "Sync Products" button
2. Wait for indexing to complete
3. Your store now has semantic search!

### Step 4: Test Search
Go to your store's search bar and try searching for products using natural language.
            """
        },
        "api_reference": {
            "title": "API Reference",
            "content": """
## Search API

### Endpoint
```
POST /search
Authorization: Bearer <your_license_key>
Content-Type: application/json
```

### Request Body
```json
{
  "query": "red shoes for running",
  "limit": 10,
  "filters": {
    "category": "footwear",
    "price_range": [50, 200]
  }
}
```

### Response
```json
{
  "success": true,
  "results": [...],
  "total": 42,
  "search_time": 0.123
}
```

## Webhooks

### Product Sync
```
POST /webhook/product-created
POST /webhook/product-updated  
POST /webhook/product-deleted
```
            """
        },
        "troubleshooting": {
            "title": "Troubleshooting",
            "content": """
## Common Issues

### License Key Invalid
- Ensure you copied the entire key
- Check your domain matches the license
- Verify the key hasn't expired

### Search Not Working
- Check API URL is correct
- Verify products are synced
- Check error logs in WordPress

### Slow Performance
- Ensure Redis cache is running
- Check your server resources
- Consider upgrading your plan

### No Results Found
- Verify products are indexed
- Check search query format
- Review product attributes
            """
        },
        "advanced": {
            "title": "Advanced Configuration",
            "content": """
## Custom Search Results

### Filter Products by Category
```javascript
const filters = {
  category: "electronics",
  price_range: [100, 1000],
  in_stock: true
};
```

### Custom Result Ranking
```javascript
const boost_config = {
  "title": 2.0,
  "description": 1.5, 
  "attributes": 1.0
};
```

## Analytics Integration

### Custom Events
```javascript
// Track custom search events
analytics.track('semantic_search', {
  query: "winter boots",
  results_count: 23,
  conversion: true
});
```

## Performance Optimization

### Caching Strategy
- Enable Redis for best performance
- Cache popular search queries
- Pre-load seasonal products
            """
        }
    }

@router.get("/semantic-search-woo.zip")
async def download_plugin():
    """Serve the plugin ZIP file for download."""
    plugin_path = "backend/app/static/semantic-search-woo.zip"
    return FileResponse(
        path=plugin_path,
        filename="semantic-search-woo.zip",
        media_type="application/zip"
    )
