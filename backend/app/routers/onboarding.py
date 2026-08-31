"""
Onboarding: pick a platform and a product, get a license key scoped to it.

The flow changed shape with per-product licensing. It used to assume one
product (WooCommerce semantic search) and one key per customer. Now a customer
running three Magento modules needs three keys on the same store, so signup is
find-or-create on the client and idempotent on (client, domain, product)
rather than a one-shot registration that errors if the email is already known.
"""

import logging
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.app.services import catalog
from backend.app.services.database import get_db
from backend.app.services.license_service import (
    create_client,
    generate_license_key,
    get_client_by_email,
)

logger = logging.getLogger(__name__)

router = APIRouter()

templates = Jinja2Templates(directory="backend/app/templates")


class SignupResponse(BaseModel):
    success: bool
    license_key: Optional[str] = None
    client_id: Optional[str] = None
    product: Optional[dict] = None
    platform: Optional[dict] = None
    domain: Optional[str] = None
    plan: Optional[dict] = None
    install: Optional[dict] = None
    reissued: bool = False
    error: Optional[str] = None


def extract_domain(url: str) -> str:
    """Pull the bare host out of a store URL.

    The domain is what a license key is bound to, so this has to agree exactly
    with what DomainAuthorizer compares against at request time — host only,
    lowercased, no scheme, no port, no path, no leading www. Anything looser
    issues keys that fail on the first real request.
    """
    raw = (url or "").strip()
    if not raw:
        raise ValueError("Enter your store URL.")

    # urlparse puts a bare "shop.example.com" in .path, not .netloc, so give it
    # a scheme when the customer omitted one rather than reading the wrong field.
    if "://" not in raw:
        raw = "https://" + raw

    host = (urlparse(raw).netloc or "").lower().strip()
    if not host:
        raise ValueError("That store URL doesn't look right. Use the full address, like https://yourstore.com")

    host = host.split("@")[-1]      # strip any user:pass@
    host = host.split(":")[0]       # strip :8080
    if host.startswith("www."):
        host = host[4:]

    if "." not in host:
        # Allow bare hostnames for local development; reject the typo case
        # ("myyshop") that would otherwise mint an unusable key.
        if host not in {"localhost"}:
            raise ValueError(f"'{host}' isn't a full domain. Use the address shoppers visit, like https://yourstore.com")

    return host


def _install_instructions(product: dict, platform: dict, license_key: str, domain: str) -> dict:
    """Per-platform install steps, built from the catalogue entry.

    Magento and WooCommerce have nothing in common here — one is a Composer
    require plus CLI enable, the other is a ZIP upload in wp-admin. Returning
    the wrong one is the fastest route to a support ticket, so the steps are
    derived from the platform's install_style rather than templated once and
    hoped over.
    """
    if platform["install_style"] == "composer":
        return {
            "style": "composer",
            "heading": f"Install {product['name']} on Magento 2",
            "steps": [
                {
                    "label": "Require the module",
                    "code": f"composer require {product['artifact']}",
                },
                {
                    "label": "Enable it and run setup",
                    "code": (
                        f"bin/magento module:enable {product['module_name']}\n"
                        "bin/magento setup:upgrade\n"
                        "bin/magento cache:flush"
                    ),
                },
                {
                    "label": "Paste the key",
                    "detail": (
                        f"Stores → Configuration → Czargroup → {product['name']}. "
                        "Set the license key, save, then run the catalogue sync."
                    ),
                },
            ],
        }

    return {
        "style": "wp_zip",
        "heading": f"Install {product['name']} on WooCommerce",
        "steps": [
            {
                "label": "Upload the plugin",
                "detail": (
                    f"Plugins → Add New → Upload Plugin, choose {product['artifact']}, "
                    "then Install Now and Activate."
                ),
            },
            {
                "label": "Paste the key",
                "detail": (
                    f"WooCommerce → Settings → {product['name']}. "
                    "Set the license key and save."
                ),
            },
            {
                "label": "Sync your catalogue",
                "detail": "Run the first sync from the same settings screen. It runs in the background.",
            },
        ],
    }


@router.get("/onboarding", response_class=HTMLResponse)
async def onboarding_page(request: Request):
    """Serve the onboarding page."""
    return templates.TemplateResponse(request, "onboarding.html")


@router.get("/api/onboarding/catalog")
async def get_catalog():
    """Platforms, their products, and the plan tiers.

    The page renders its pickers entirely from this, so adding a product to
    catalog.PRODUCTS puts it on the site with no template change.
    """
    return catalog.public_catalog()


@router.post("/api/onboarding/signup")
async def signup_client(
    name: str = Form(...),
    email: str = Form(...),
    company_name: Optional[str] = Form(None),
    store_name: str = Form(...),
    store_url: str = Form(...),
    product_code: str = Form(...),
    platform: Optional[str] = Form(None),
    platform_version: Optional[str] = Form(None),
    estimated_products: Optional[str] = Form(None),
    plan: str = Form(catalog.DEFAULT_MODULE_PLAN),
    db: Session = Depends(get_db),
):
    """Issue a license key scoped to one product on one store.

    Idempotent on (client, domain, product): asking twice returns the key you
    already have rather than minting a second one. Customers do re-submit —
    they lose the tab, or they come back for a different product and fill the
    same form again — and silently accumulating duplicate active keys for one
    install would make per-product usage ambiguous later.
    """
    try:
        # Product first: everything downstream (platform, install steps, the
        # key's scope) derives from it, and a bad code should fail before we
        # create a client record.
        resolved_platform = catalog.validate_selection(product_code, platform)
        product = catalog.get_product(product_code)
        platform_meta = catalog.get_platform(resolved_platform)

        domain = extract_domain(store_url)

        if not catalog.is_valid_module_plan(plan):
            plan = catalog.DEFAULT_MODULE_PLAN

        # Find-or-create. A customer buying their second product is the normal
        # case now, not an error — the old code raised "email already exists"
        # here and made multi-product impossible through the UI.
        existing = get_client_by_email(db, email)
        if existing:
            if not existing["is_active"]:
                return SignupResponse(
                    success=False,
                    error="That account is inactive. Get in touch and we'll sort it out.",
                )
            client_id = existing["id"]
        else:
            client_id = create_client(db, name, email, plan)["id"]

        # Already licensed for this product on this store? Hand back the same
        # key. Note this reads the row rather than trusting the JWT, so a key
        # deactivated in the database isn't resurrected by a re-submit.
        prior = db.execute(text("""
            SELECT license_key
            FROM license_keys
            WHERE client_id      = :client_id
              AND allowed_domain = :domain
              AND product_code   = :product_code
              AND is_active      = 1
            ORDER BY created_at DESC
            LIMIT 1
        """), {
            "client_id": client_id,
            "domain": domain,
            "product_code": product_code,
        }).fetchone()

        if prior:
            license_key = prior.license_key
            reissued = True
        else:
            license_key = generate_license_key(
                db=db,
                client_id=client_id,
                allowed_domain=domain,
                plan=plan,
                valid_days=365,
                product_code=product_code,
            )
            reissued = False

        logger.info(
            "onboarding issued key client=%s domain=%s product=%s platform=%s plan=%s reissued=%s",
            client_id, domain, product_code, resolved_platform, plan, reissued,
        )

        return SignupResponse(
            success=True,
            license_key=license_key,
            client_id=client_id,
            product={
                "code": product["code"],
                "name": product["name"],
                "artifact": product["artifact"],
                "module_name": product["module_name"],
            },
            platform={
                "code": platform_meta["code"],
                "name": platform_meta["name"],
                "accent": platform_meta["accent"],
            },
            domain=domain,
            plan=catalog.MODULE_PLANS[plan],
            install=_install_instructions(product, platform_meta, license_key, domain),
            reissued=reissued,
        )

    except ValueError as e:
        # Raised by validate_selection and extract_domain, both of which write
        # their messages for a customer to read.
        return SignupResponse(success=False, error=str(e))
    except Exception:
        logger.exception("onboarding signup failed for %s / %s", email, product_code)
        return SignupResponse(
            success=False,
            error="We couldn't issue the key. Try again, and if it keeps failing let us know.",
        )


@router.get("/api/onboarding/plans")
async def get_plans():
    """Plan tiers. Kept at its original path — the WooCommerce plugin's
    settings screen fetches this to show the customer what they're on."""
    return {"plans": {code: plan for code, plan in catalog.MODULE_PLANS.items()}}


@router.get("/{artifact}.zip")
async def download_plugin(artifact: str):
    """Serve a WordPress plugin ZIP.

    Only artifacts named in the catalogue are servable — the path segment
    reaches the filesystem, so an allowlist of known plugin slugs is what keeps
    it from being a traversal primitive.
    """
    filename = f"{artifact}.zip"

    known = {
        p["artifact"] for p in catalog.PRODUCTS.values()
        if p["artifact"].endswith(".zip")
    }
    if filename not in known:
        raise HTTPException(status_code=404, detail="No such download")

    # The build artifacts aren't checked into the repo, so on a fresh checkout
    # this is a 404 rather than a 500 from FileResponse hitting a missing path.
    path = Path("backend/app/static") / filename
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"{filename} isn't available for download yet. Ask us for a copy.",
        )

    return FileResponse(
        path=str(path),
        filename=filename,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "no-cache",
        },
    )
