"""
Central logging configuration for the semantic-search backend.

Two named loggers, two rotating files:

    api  → logs/api.log   Every inbound FastAPI request + its response.
    llm  → logs/llm.log   Every outbound LLM call (embeddings, reranks, chat).

Each file rotates at 10 MB with 5 backups kept (≈ 60 MB total per logger
worst-case). Log-level is INFO by default, DEBUG when the LOG_DEBUG env var is
set to "1"/"true".

Nothing propagates up to the root logger so these two files never pollute
the Uvicorn console.
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Logs live outside the backend/ package so they survive `pip install -e` reloads.
LOG_DIR = Path(
    os.getenv("CZAR_LOG_DIR", Path(__file__).resolve().parents[3] / "logs")
)
LOG_DIR.mkdir(parents=True, exist_ok=True)


def _level() -> int:
    return logging.DEBUG if os.getenv("LOG_DEBUG", "").lower() in ("1", "true", "yes") else logging.INFO


def _make_logger(name: str, filename: str) -> logging.Logger:
    logger = logging.getLogger(name)
    # Idempotent — don't double-register handlers if this module is re-imported.
    if getattr(logger, "_czar_configured", False):
        return logger

    logger.setLevel(_level())
    logger.propagate = False

    file_handler = RotatingFileHandler(
        LOG_DIR / filename,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    # Bare %(message)s — the formatting is done inside the log statements
    # themselves, with ASCII separators + icons so a tail -f stays readable.
    file_handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)

    logger._czar_configured = True  # type: ignore[attr-defined]
    return logger


# Exported singletons
api_logger = _make_logger("czar.api", "api.log")
llm_logger = _make_logger("czar.llm", "llm.log")
