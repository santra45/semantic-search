#!/usr/bin/env python3
"""
One-time backfill for token_usage_tracking rows whose cost columns are zero.

Bug being repaired
------------------
The /retrieve/answer endpoint was inserting `chat_answer` rows without
passing input_cost / output_cost to the tracker. Token counts were correct,
but cost columns defaulted to 0.0 — so the admin dashboard's per-message
billing summed to zero for chat_answer no matter how many chats happened.

This script recomputes cost from input_tokens / output_tokens / llm_model
using the same MODEL_PRICING table the tracker would have used at write
time, then UPDATEs the affected rows in place.

Idempotent — only touches rows where total_cost is still ~0 AND there were
non-zero tokens. Re-running is safe; the second run finds nothing to do.

Defaults to chat_answer rows only (that's the known bug). Other query types
already wrote correct cost values from day one.

Usage
-----
    # Always preview first.
    python scripts/backfill_chat_answer_cost.py --dry-run

    # Actually update.
    python scripts/backfill_chat_answer_cost.py --yes

    # Scope to one tenant.
    python scripts/backfill_chat_answer_cost.py --client-id CLIENT_XYZ --yes

    # Belt-and-braces: backfill any query type whose rows have zeroed cost
    # but non-zero tokens (e.g. if you suspect another path had the same bug).
    python scripts/backfill_chat_answer_cost.py --query-type chat_intent --dry-run

Run from the semantic-search project root so the `backend.app` imports resolve.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

try:
    from sqlalchemy import text
    from backend.app.services.database import get_db
    from backend.app.services.llm_rerank_service import MODEL_PRICING
except ImportError as exc:
    sys.stderr.write(
        f"ERROR: import failed ({exc}).\n"
        "       Run this script from the semantic-search/ project root, with the\n"
        "       backend's Python environment active (the same one that runs the\n"
        "       FastAPI app). The script imports backend.app.services.database\n"
        "       to reuse the configured DB connection.\n"
    )
    sys.exit(1)


# Below this magnitude we treat the stored cost as "zero" — handles both
# DECIMAL exact-zero and any FLOAT residue from earlier writes.
ZERO_COST_EPSILON = 0.0000001


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill cost columns on token_usage_tracking rows that "
                    "have non-zero tokens but zeroed cost.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--query-type",
        default="chat_answer",
        help="Restrict to this query_type (default: chat_answer — the only "
             "type known to be affected by the bug).",
    )
    parser.add_argument(
        "--client-id",
        help="Optional: limit to one tenant's rows.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change, don't write.",
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip the y/N confirmation prompt.",
    )
    args = parser.parse_args()

    db = next(get_db())

    # 1) Find candidate rows. Only rows that BOTH have zeroed cost AND have
    # non-zero tokens — anything else is either correctly costed already or
    # genuinely had zero usage (no LLM call billed).
    where = (
        "query_type = :qt "
        "AND total_cost < :eps "
        "AND (input_tokens > 0 OR output_tokens > 0)"
    )
    params = {"qt": args.query_type, "eps": ZERO_COST_EPSILON}
    if args.client_id:
        where += " AND client_id = :cid"
        params["cid"] = args.client_id

    rows = db.execute(text(f"""
        SELECT request_id, llm_model, input_tokens, output_tokens
        FROM token_usage_tracking
        WHERE {where}
    """), params).fetchall()

    if not rows:
        print(f"No {args.query_type} rows with zeroed cost + non-zero tokens found. Nothing to do.")
        return

    # 2) Show breakdown by model so the user can spot any model that lacks
    # pricing data (we'd skip those — can't fabricate a cost).
    model_counts: dict[str, int] = {}
    for r in rows:
        model_counts[r.llm_model] = model_counts.get(r.llm_model, 0) + 1

    print(f"Found {len(rows)} {args.query_type} row(s) candidate for backfill.\n")
    print(f"{'MODEL':<40} {'ROWS':>6}  PRICING")
    print("-" * 70)
    for model, count in sorted(model_counts.items(), key=lambda x: -x[1]):
        marker = "OK" if model in MODEL_PRICING else "MISSING (will skip)"
        print(f"{model:<40} {count:>6}  {marker}")

    # 3) Compute the actual updates. Skip rows whose model isn't in the
    # pricing table — we don't have data to fabricate a real cost, and zero
    # is at least an honest "we don't know".
    updates: list[dict] = []
    skipped_no_pricing = 0
    skipped_zero_cost = 0
    for r in rows:
        pricing = MODEL_PRICING.get(r.llm_model)
        if not pricing:
            skipped_no_pricing += 1
            continue
        in_tokens = int(r.input_tokens or 0)
        out_tokens = int(r.output_tokens or 0)
        input_cost = in_tokens * pricing.get("input", 0.0)
        output_cost = out_tokens * pricing.get("output", 0.0)
        total_cost = input_cost + output_cost
        if total_cost <= 0:
            # Pricing exists but is zero for both classes (shouldn't happen
            # for any real entry but guard anyway).
            skipped_zero_cost += 1
            continue
        updates.append({
            "request_id": r.request_id,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
        })

    if not updates:
        print(f"\nNothing to update — {skipped_no_pricing} skipped (no pricing), "
              f"{skipped_zero_cost} skipped (computed cost is zero).")
        return

    # 4) Show a sample so the user can sanity-check the math.
    print(f"\nSample updates (first 3 of {len(updates)}):")
    for u in updates[:3]:
        print(f"  request_id={u['request_id']}")
        print(f"    input_cost:  ${u['input_cost']:.10f}")
        print(f"    output_cost: ${u['output_cost']:.10f}")
        print(f"    total_cost:  ${u['total_cost']:.10f}")

    grand_total = sum(u["total_cost"] for u in updates)
    print(f"\nSummary:")
    print(f"  Will update:           {len(updates)} row(s)")
    print(f"  Skipped (no pricing):  {skipped_no_pricing}")
    print(f"  Skipped (zero cost):   {skipped_zero_cost}")
    print(f"  Total cost to backfill: ${grand_total:.6f}")

    if args.dry_run:
        print("\n(dry run — pass --yes to actually update)")
        return

    if not args.yes:
        resp = input(f"\nUpdate {len(updates)} row(s)? [y/N] ").strip().lower()
        if resp != "y":
            print("Aborted.")
            return

    # 5) Apply updates. Single transaction; commit at the end. Acceptable
    # because chat_answer row counts are typically small (one row per
    # RAG-summarised chat turn, so probably hundreds-to-low-thousands per
    # store). If your store has many more, consider chunking the commits.
    update_sql = text("""
        UPDATE token_usage_tracking
        SET input_cost = :input_cost,
            output_cost = :output_cost,
            total_cost = :total_cost
        WHERE request_id = :request_id
    """)

    written = 0
    try:
        for u in updates:
            db.execute(update_sql, u)
            written += 1
        db.commit()
    except Exception as exc:
        db.rollback()
        sys.stderr.write(f"\nERROR after writing {written} row(s): {exc}\n")
        sys.stderr.write("Transaction rolled back. Re-run --dry-run to see remaining work.\n")
        sys.exit(1)

    print(f"\nUpdated {written} row(s). Total cost backfilled: ${grand_total:.6f}")


if __name__ == "__main__":
    main()
