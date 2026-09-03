/**
 * Formatters. The important one is that `null` renders as an em dash.
 *
 * ────────────────────────────────────────────────────────────────────────────
 * ABSENT IS NOT ZERO, AND THIS FILE IS WHERE THAT IS ENFORCED.
 *
 * The read API distinguishes them deliberately: a tenant still on a v1 JWT
 * resolves no v2 context, so `usage_service.record()` writes nothing and there
 * is no row to count — its request total comes back as `null`, not `0`. A
 * subscription with a v2 licence that nobody has used comes back as `0`.
 *
 * "Used the product, spent nothing" and "not measured at all" are different
 * facts, and only one of them is true of each. Rendering both as `0` is how a
 * console spends a migration window quietly reporting that the platform costs
 * nothing — which is the failure the backend's `i_or_none` was written to
 * prevent, and which this file is the other half of.
 *
 * So: never `?? 0` on the way to a screen. Use these.
 * ────────────────────────────────────────────────────────────────────────────
 */

export const ABSENT = "—";

/** Integer, thousands-separated. `null`/`undefined` → em dash. */
export function num(value: number | null | undefined): string {
  if (value === null || value === undefined) return ABSENT;
  return value.toLocaleString();
}

/**
 * Cost in USD. Small numbers matter here — a single answer can cost
 * $0.0008 — so this does not round to cents. A dashboard that shows $0.00 for
 * real spend is telling the same lie as one that shows 0 for unmeasured.
 */
export function cost(value: number | null | undefined): string {
  if (value === null || value === undefined) return ABSENT;
  if (value === 0) return "$0";
  if (value < 0.01) return `$${value.toFixed(6)}`;
  return `$${value.toFixed(2)}`;
}

export function pct(value: number | null | undefined): string {
  if (value === null || value === undefined) return ABSENT;
  return `${value.toFixed(1)}%`;
}

/** ISO timestamp → local short form. */
export function when(value: string | null | undefined): string {
  if (!value) return ABSENT;
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return String(value);
  return d.toLocaleString(undefined, {
    year: "numeric", month: "short", day: "2-digit",
    hour: "2-digit", minute: "2-digit",
  });
}

export function day(value: string | null | undefined): string {
  if (!value) return ABSENT;
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return String(value);
  return d.toLocaleDateString(undefined, { month: "short", day: "2-digit" });
}

/** "in 12 days" / "3 days ago" / em dash. Used for expiry columns. */
export function relative(value: string | null | undefined): string {
  if (!value) return ABSENT;
  const d = new Date(value).getTime();
  if (Number.isNaN(d)) return String(value);
  const days = Math.round((d - Date.now()) / 86_400_000);
  if (days === 0) return "today";
  if (days > 0) return days === 1 ? "tomorrow" : `in ${days} days`;
  return days === -1 ? "yesterday" : `${Math.abs(days)} days ago`;
}

/**
 * Product codes are permanent by contract and written into billing history, so
 * they are never renamed — but `magento_product_qa` is not what an operator
 * calls it. Falls through to the raw code for anything unmapped rather than
 * inventing a label, so a product added to the catalogue shows up recognisably
 * instead of blank.
 */
const PRODUCT_LABELS: Record<string, string> = {
  magento_search: "Magento · Search",
  magento_chatbot: "Magento · Chatbot",
  magento_product_qa: "Magento · Product Q&A",
  woo_search: "Woo · Search",
  woo_product_qa: "Woo · Product Q&A",
};

export function productLabel(code: string | null | undefined): string {
  if (!code) return ABSENT;
  return PRODUCT_LABELS[code] ?? code;
}
