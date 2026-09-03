import { useQuery } from "@tanstack/react-query";
import { api } from "../api";

/**
 * The two plan ladders, fetched rather than hardcoded.
 *
 * catalog.py is the authority for what a plan IS — codes, limits, ordering —
 * and hardcoding a copy here would mean a plan added on the server never
 * appears in the console, or worse, one removed there stays selectable until
 * somebody picks it and gets a 422.
 *
 * THEY ARE TWO LADDERS AND THEY ARE NOT INTERCHANGEABLE:
 *
 *   index_plans   catalogue SIZE, bought once per SITE, because a store's
 *                 modules share one Qdrant collection
 *   module_plans  request QUOTA, bought per MODULE
 *
 * Neither can be derived from the other, which is why the console edits them
 * from two different places and never offers a single "plan" control.
 */

export interface PlanRung {
  code: string;
  name?: string;
  price?: string;
  period?: string;
  features?: string[];
  /** INDEX_PLANS only — catalogue items, bought once per site. */
  catalogue_limit?: number;
  /** MODULE_PLANS only — requests per month, bought per module. */
  request_limit?: number;
  /** False on `trial`: a rung the customer cannot choose. */
  selectable?: boolean;
  /** Counts added by the API, not by catalog.py. */
  sites?: number;
  subscriptions?: number;
  [k: string]: unknown;
}

export interface Plans {
  index_plans: PlanRung[];
  module_plans: PlanRung[];
  revenue_computable: boolean;
}

export function usePlans() {
  return useQuery<Plans>({
    queryKey: ["plans"],
    queryFn: () => api.get<Plans>("/api/admin/plans"),
    // Reference data. It changes when somebody edits catalog.py and deploys,
    // which is not something worth re-checking on every screen mount.
    staleTime: 30 * 60_000,
  });
}

/** `Growth · 100,000 req/mo · $29` for a dropdown.
 *
 *  The limit fields are named catalogue_limit and request_limit, NOT `products`
 *  and `requests` — an earlier version guessed those and silently rendered
 *  name-and-price only, which made two rungs of the same price
 *  indistinguishable in the picker. Falls back to the bare code rather than
 *  inventing a label, so an unfamiliar rung stays selectable. */
export function rungLabel(rung: PlanRung): string {
  const bits: string[] = [rung.name ? String(rung.name) : rung.code];
  if (typeof rung.request_limit === "number")
    bits.push(`${rung.request_limit.toLocaleString()} req/mo`);
  if (typeof rung.catalogue_limit === "number")
    bits.push(`${rung.catalogue_limit.toLocaleString()} items`);
  if (rung.price) bits.push(String(rung.price));
  return bits.join(" · ");
}
