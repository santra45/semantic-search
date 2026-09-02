"""Seed sites.indexed_items from Qdrant, and grandfather anyone it puts over.

WHY THIS HAS TO RUN BEFORE ENFORCEMENT MOVES

sites.indexed_items ships as 0 on every row. The collections are not empty —
they hold the catalogues stores have already synced. Point enforcement at the
column before this script has run and every store is handed a fresh full
allowance on top of what it already holds, which is the opposite of a ceiling.

WHY IT GRANDFATHERS

The counter now counts every logical entity: products, CMS pages and blocks,
categories, FAQs, WordPress pages and posts. The sync routers previously
counted products alone and said so — "non-product content is free". So a store
sitting comfortably under its limit on the old arithmetic can be over it on the
new one through no action of its own, purely because the definition widened
underneath them.

Refusing their next sync would be charging them retroactively for content they
were told was free. Instead, any site whose true count exceeds its ceiling is
raised to the smallest rung that fits, and every such move is printed as a
BILLING REVIEW line naming the site, the old rung, the new rung and the count
that forced it. That list is the paper trail: somebody has to decide whether
those stores get invoiced at the new rung or keep it as a goodwill grant, and
this script does not make that decision — it only makes sure nobody's store
breaks while it is being made.

A site whose count exceeds even the largest rung is reported and NOT moved.
There is no plan to raise it to, and silently leaving it over its ceiling is
the honest state — the counter tells the truth, and what to do about it is a
conversation, not a default.

USAGE

    python -m scripts.backfill_indexed_items            # dry run, changes nothing
    python -m scripts.backfill_indexed_items --apply    # writes

Idempotent: running it twice is a no-op the second time, because it SETS the
count rather than adjusting it, and a site already on a sufficient rung is left
alone.
"""

from __future__ import annotations

import argparse
import sys

from sqlalchemy import text

from backend.app.services import catalog
from backend.app.services.database import SessionLocal
from backend.app.services.qdrant_service import count_indexed_entities


def smallest_rung_for(count: int) -> str | None:
    """The cheapest INDEX_PLANS rung whose ceiling holds *count*, or None.

    Walks INDEX_PLAN_ORDER rather than sorting by limit, because the order is
    the product's own statement of which plan is 'next' and a rung priced
    higher for reasons other than capacity would sort wrong.
    """
    for code in catalog.INDEX_PLAN_ORDER:
        if count <= catalog.INDEX_PLANS[code]["catalogue_limit"]:
            return code
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write the changes; without it the script only reports",
    )
    args = parser.parse_args()

    db = SessionLocal()
    try:
        sites = db.execute(text("""
            SELECT s.id, s.domain, s.client_id, s.index_plan,
                   s.catalogue_limit, s.indexed_items, c.name AS client_name
            FROM sites s
            JOIN clients c ON c.id = s.client_id
            ORDER BY c.name, s.domain
        """)).fetchall()

        if not sites:
            print("No sites. Nothing to do.")
            return 0

        print(f"{'DRY RUN — no writes' if not args.apply else 'APPLYING'}\n")
        print(f"{'store':<40} {'plan':<8} {'was':>8} {'true':>8} {'limit':>9}  note")
        print("-" * 92)

        upgrades: list[tuple] = []
        over_largest: list[tuple] = []
        touched_site_ids: list[str] = []
        changed = 0

        for site in sites:
            # Counted from Qdrant, which is the only thing that knows what is
            # actually in the collection. A store whose collection was never
            # created counts 0, which is correct and not an error.
            true_count = count_indexed_entities(str(site.client_id), str(site.domain))
            limit = int(site.catalogue_limit)
            note = ""

            new_plan = None
            if true_count > limit:
                fits = smallest_rung_for(true_count)
                if fits is None:
                    over_largest.append((site.client_name, site.domain, true_count))
                    note = "OVER LARGEST RUNG — not moved"
                elif fits != site.index_plan:
                    new_plan = fits
                    upgrades.append(
                        (site.client_name, site.domain, site.index_plan, fits, true_count)
                    )
                    note = f"grandfathered {site.index_plan} -> {fits}"

            if true_count != int(site.indexed_items) or new_plan:
                changed += 1
                touched_site_ids.append(str(site.id))

            print(
                f"{str(site.domain)[:39]:<40} {str(site.index_plan):<8} "
                f"{int(site.indexed_items):>8,} {true_count:>8,} {limit:>9,}  {note}"
            )

            if args.apply:
                if new_plan:
                    # Ceiling and rung move together — set_index_plan is not
                    # used here because it evicts the auth cache per call and
                    # this loop already knows it is the only writer.
                    db.execute(text("""
                        UPDATE sites
                        SET index_plan = :plan, catalogue_limit = :limit
                        WHERE id = :id
                    """), {
                        "plan": new_plan,
                        "limit": catalog.INDEX_PLANS[new_plan]["catalogue_limit"],
                        "id": site.id,
                    })
                db.execute(text("""
                    UPDATE sites SET indexed_items = :n WHERE id = :id
                """), {"n": true_count, "id": site.id})

        if args.apply:
            db.commit()
            # Every cached authorisation context carries indexed_items and
            # catalogue_limit, so a run that changed either has to evict the
            # keys for that site or the old numbers stay live for the TTL.
            # Per site rather than wholesale: auth_cache has no flush-all call
            # and should not grow one — it would empty the cache for every
            # tenant to fix the handful this run touched.
            if touched_site_ids:
                from backend.app.services import auth_cache
                evicted = 0
                for site_id in touched_site_ids:
                    try:
                        evicted += auth_cache.invalidate_for_site(db, site_id)
                    except Exception as exc:
                        print(f"  could not evict cache for site {site_id}: {exc}")
                print("")
                print(f"auth cache: {evicted} key(s) evicted across "
                      f"{len(touched_site_ids)} site(s)")

        print("-" * 92)
        print(f"{len(sites)} sites, {changed} would change" if not args.apply
              else f"{len(sites)} sites, {changed} changed")

        if upgrades:
            print(f"\nBILLING REVIEW — {len(upgrades)} site(s) grandfathered onto a higher rung.")
            print("These stores were within their limit on the old products-only count and")
            print("are over it on the new all-entities count. Decide whether to invoice at")
            print("the new rung or record it as a grant; nothing here does that for you.")
            for name, domain, old, new, count in upgrades:
                print(f"  {name} / {domain}: {old} -> {new} ({count:,} entities)")

        if over_largest:
            print(f"\nNO RUNG FITS — {len(over_largest)} site(s) exceed the largest plan.")
            print("Left on their current plan and over their ceiling. Needs a human.")
            for name, domain, count in over_largest:
                print(f"  {name} / {domain}: {count:,} entities")

        if not args.apply:
            print("\nRe-run with --apply to write.")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
