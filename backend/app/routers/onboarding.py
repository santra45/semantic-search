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
from sqlalchemy.orm import Session

from backend.app.services import (
    auth_cache,
    catalog,
    licensing_service,
    tenancy_service,
)
from backend.app.services.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter()

templates = Jinja2Templates(directory="backend/app/templates")


class SignupResponse(BaseModel):
    success: bool
    # None on the repeat path: only the hash is stored, so a key already issued
    # cannot be shown again. key_prefix is what identifies it instead.
    license_key: Optional[str] = None
    key_prefix: Optional[str] = None
    client_id: Optional[str] = None
    site_id: Optional[str] = None
    subscription_id: Optional[str] = None
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


def _evict(*results: dict) -> None:
    """Drop the cached authorisation contexts the mutators just invalidated.

    Every function in tenancy_service and licensing_service returns
    "key_hashes" — the keys whose cached context its write made stale — so one
    helper covers all of them and a caller never has to know which mutator owes
    an eviction. Almost always empty on the signup path (a brand-new client,
    site or subscription holds no licences yet); it is called anyway because
    the one case where it is NOT empty is a repeat signup that refreshed a
    site's metadata, and that is exactly the case nobody remembers to handle.
    """
    hashes = [h for result in results for h in (result.get("key_hashes") or [])]
    if hashes:
        auth_cache.invalidate_many(hashes)


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

        # Find-or-create down the whole chain: client, then the store, then the
        # subscription for this product on that store. A customer buying their
        # second module is the ordinary case, so every step is idempotent and
        # none of them overwrite what a previous purchase established.
        #
        # Domain is validated here first so a typo fails before any row is
        # written, but find_or_create_site normalises it again and site["domain"]
        # is the canonical value — a licence is bound to what it stored, not to
        # what this function computed.
        extract_domain(store_url)

        client = tenancy_service.find_or_create_client(db, name, email, company_name)
        site = tenancy_service.find_or_create_site(
            db,
            client_id=client["id"],
            domain=store_url,
            platform=resolved_platform,
            store_name=store_name,
            platform_version=platform_version,
            index_plan=catalog.index_plan_for_catalogue_size(estimated_products),
        )

        # create_subscription is find-or-create and deliberately ignores the
        # plan it is passed, so the plan is applied in a second step below.
        # Opening it here on its own defaults keeps the (plan, status) pair
        # coherent: 'starter' with the default status='trial' is a pair the
        # service layer refuses outright, which is what made every submission
        # through this form fail.
        subscription = licensing_service.create_subscription(
            db,
            site_id=site["id"],
            product_code=product_code,
        )

        # Apply the purchased plan ONLY on a subscription this request opened.
        # set_subscription_plan is an upsert and re-running onboarding is
        # ordinary — a merchant reinstalls, or comes back for a second module
        # and fills the same form again with the pricing page's default radio
        # checked. Applying it unconditionally would drop a Pro customer to
        # Starter on their own re-submit, which is the exact downgrade
        # create_subscription's find-or-create exists to prevent; doing it here
        # instead would just move the bug one line down.
        if subscription.get("created") and plan != catalog.TRIAL_MODULE_PLAN:
            subscription = licensing_service.set_subscription_plan(
                db,
                subscription_id=subscription["id"],
                plan=plan,
            )

        domain = site["domain"]
        client_id = client["id"]

        # Already holding a live key for this subscription? There is nothing to
        # hand back but the prefix — only the SHA-256 hash is stored, so the
        # plaintext is genuinely unrecoverable. Minting a second key on a
        # re-submit would leave two valid credentials for one install and make
        # revocation ambiguous, so the repeat path issues nothing.
        live = next(
            (lic for lic in licensing_service.list_licences(db, subscription["id"])
             if lic["is_active"]),
            None,
        )

        if live:
            license_key = None
            key_prefix = live["key_prefix"]
            reissued = True
        else:
            # environment comes off the site, not off the form. It decides
            # whose inference spend the usage rows are stamped with, and a
            # merchant must not be able to choose that for themselves.
            licence = licensing_service.issue_licence(
                db,
                subscription_id=subscription["id"],
                environment=site["environment"],
                valid_days=365,
            )
            license_key = licence["key"]
            key_prefix = licence["key_prefix"]
            reissued = False
            _evict(licence)

        # The mutators all report which cached authorisation contexts they
        # invalidated; evicting them is the caller's job and one helper covers
        # every shape.
        _evict(client, site, subscription)

        logger.info(
            "onboarding issued key client=%s site=%s domain=%s product=%s "
            "platform=%s plan=%s prefix=%s reissued=%s",
            client_id, site["id"], domain, product_code, resolved_platform,
            plan, key_prefix, reissued,
        )

        return SignupResponse(
            success=True,
            license_key=license_key,
            key_prefix=key_prefix,
            client_id=client_id,
            site_id=site["id"],
            subscription_id=subscription["id"],
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
            plan=catalog.MODULE_PLANS[subscription["plan"]],
            install=_install_instructions(product, platform_meta, license_key or "", domain),
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
