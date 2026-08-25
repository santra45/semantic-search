"""
Add `product_code` and `platform` to license_keys.

Until now a license key was scoped to (client, domain) and implied every
product. That made per-product accounting impossible: three of the Magento
modules (AIChatbot, AIProductQA, AISearch) all call the same backend
endpoints, so the route a request arrived on cannot tell you which product
the customer is running. The product identity has to travel on the key.

Both columns are NULLABLE on purpose. Keys already deployed in customer
stores carry no product claim and cannot be invalidated, so NULL means
"legacy key, all products" and those keys keep working untouched. New keys
issued through onboarding carry a real product code. The legacy count
dropping to zero is the migration's own progress bar — see report below.

Safe to run repeatedly: every step checks information_schema first and skips
work already done. Nothing here drops, rewrites, or reorders existing data.
"""

import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text

from backend.app.services.database import engine

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

TABLE = "license_keys"

# (column, DDL) pairs. Kept as data rather than one ALTER statement so a
# partially-applied migration — someone ran an earlier version, or the first
# ALTER succeeded and the second failed — resumes cleanly instead of erroring
# on the column that already exists.
COLUMNS = [
    (
        "product_code",
        # 64 chars is generous for codes like 'magento_product_qa'; sized so a
        # future product with a longer name doesn't need a second migration.
        "ADD COLUMN product_code VARCHAR(64) NULL "
        "COMMENT 'catalog.PRODUCTS key; NULL = legacy all-products key'",
    ),
    (
        "platform",
        "ADD COLUMN platform VARCHAR(32) NULL "
        "COMMENT 'catalog.PLATFORMS key; derived from product_code'",
    ),
]

# Per-product lookups are the access pattern this table is about to acquire:
# "which key authorises AIProductQA on this store". Without the index that
# becomes a scan on every license validation.
INDEXES = [
    ("idx_client_product", "ADD INDEX idx_client_product (client_id, product_code)"),
]


def _column_exists(conn, column: str) -> bool:
    return conn.execute(text("""
        SELECT COUNT(*) FROM information_schema.columns
        WHERE table_schema = DATABASE()
          AND table_name   = :table
          AND column_name  = :column
    """), {"table": TABLE, "column": column}).scalar() > 0


def _index_exists(conn, index: str) -> bool:
    return conn.execute(text("""
        SELECT COUNT(*) FROM information_schema.statistics
        WHERE table_schema = DATABASE()
          AND table_name   = :table
          AND index_name   = :index
    """), {"table": TABLE, "index": index}).scalar() > 0


def migrate() -> None:
    applied, skipped = [], []

    with engine.connect() as conn:
        exists = conn.execute(text("""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = DATABASE() AND table_name = :table
        """), {"table": TABLE}).scalar()
        if not exists:
            raise RuntimeError(
                f"Table `{TABLE}` not found in the connected database. "
                f"Check DATABASE_URL points at the right schema."
            )

        for column, ddl in COLUMNS:
            if _column_exists(conn, column):
                skipped.append(f"column {column}")
                continue
            conn.execute(text(f"ALTER TABLE {TABLE} {ddl}"))
            applied.append(f"column {column}")

        for index, ddl in INDEXES:
            if _index_exists(conn, index):
                skipped.append(f"index {index}")
                continue
            conn.execute(text(f"ALTER TABLE {TABLE} {ddl}"))
            applied.append(f"index {index}")

        conn.commit()

        logger.info("")
        for item in applied:
            logger.info("  added    %s", item)
        for item in skipped:
            logger.info("  present  %s (skipped)", item)

        # Migration progress: how many keys still have no product claim. These
        # are the ones sitting in customer stores from before this change.
        rows = conn.execute(text(f"""
            SELECT COALESCE(product_code, '(legacy)') AS product_code,
                   COALESCE(platform, '(legacy)')     AS platform,
                   COUNT(*)                           AS key_count
            FROM {TABLE}
            GROUP BY product_code, platform
            ORDER BY key_count DESC
        """)).mappings().all()

        logger.info("")
        logger.info("  license keys by product")
        for row in rows:
            logger.info(
                "    %-20s %-14s %d",
                row["product_code"], row["platform"], row["key_count"],
            )

        legacy = sum(r["key_count"] for r in rows if r["product_code"] == "(legacy)")
        logger.info("")
        if legacy:
            logger.info(
                "  %d key(s) still unscoped. They keep working as all-products "
                "keys; reissue through onboarding to scope them.", legacy,
            )
        else:
            logger.info("  every key is scoped to a product.")
        logger.info("")


if __name__ == "__main__":
    migrate()
