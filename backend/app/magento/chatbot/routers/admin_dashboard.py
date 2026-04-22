"""
Deprecated — chat history is now stored and rendered by the Magento module.

This file is kept only so stale imports don't break. All admin dashboard + chat
viewing endpoints were removed. The Magento module reads its own DB via
ResourceModel Collections; nothing on this backend has the data any more.
"""

from fastapi import APIRouter

router = APIRouter()
