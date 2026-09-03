import type { ReactNode } from "react";
import { ABSENT } from "../lib/format";

/**
 * The small shared pieces. Each one exists because it encodes a fact from the
 * API contract that would otherwise be re-decided, differently, on every screen.
 */

/**
 * Status as a COLOUR AND A WORD, never colour alone — a red dot and a green dot
 * are the same dot to a red-green colourblind operator, and this console's whole
 * job is telling live tenants from dead ones.
 */
export function Pill({ tone, children }: { tone: Tone; children: ReactNode }) {
  return <span className={`pill pill-${tone}`}>{children}</span>;
}

export type Tone = "ok" | "warn" | "bad" | "muted" | "info";

export function StatusPill({ status }: { status: string | null | undefined }) {
  if (!status) return <span className="muted">{ABSENT}</span>;
  const tone: Tone =
    status === "active" ? "ok"
    : status === "trial" ? "info"
    : status === "suspended" || status === "cancelled" ? "bad"
    : "muted";
  return <Pill tone={tone}>{status}</Pill>;
}

/**
 * The dev badge. Shown on every row that has an environment, and it is the
 * reason the environment filter defaults to `all` rather than hiding dev: an
 * operator should see demo stores in the list AND be unable to mistake one for
 * a customer.
 */
export function EnvBadge({ environment }: { environment: string | null | undefined }) {
  if (!environment) return null;
  const isProd = environment === "production";
  return (
    <Pill tone={isProd ? "ok" : "muted"}>{isProd ? "prod" : "dev"}</Pill>
  );
}

/**
 * Where a figure came from. The read API attaches `usage_source` to every
 * response that quotes usage, because during the v1→v2 window a small number
 * and an absent one are the only distinction worth making — the archive stops
 * at 2026-08-27 and the live ledger starts 2026-09-01.
 */
export interface UsageSource {
  v1_last_row?: string | null;
  v2_last_row?: string | null;
  v2_started?: boolean;
  [k: string]: unknown;
}

export function ProvenanceNote({ source }: { source: UsageSource | null | undefined }) {
  if (!source) return null;
  const v2Empty = source.v2_last_row === null || source.v2_last_row === undefined;
  if (!v2Empty) return null;
  return (
    <div className="note">
      The live usage ledger has no rows yet, so every figure here is
      pre-migration history. A zero means <strong>not measured</strong>, not
      <em> no spend</em>.
    </div>
  );
}

/**
 * Request limits are observational until AICHATBOT_QUOTA_ENFORCEMENT is armed,
 * and it is not set in production. Showing a limit without saying so lets an
 * operator "cap" a tenant and watch them sail past it.
 */
export function QuotaNote({ enforced }: { enforced: boolean | undefined }) {
  if (enforced !== false) return null;
  return <span className="note-inline">limits not enforced</span>;
}

export function Card({ title, children, right }: { title: string; children: ReactNode; right?: ReactNode }) {
  return (
    <section className="panel">
      <header className="panel-head">
        <h2>{title}</h2>
        {right}
      </header>
      <div className="panel-body">{children}</div>
    </section>
  );
}

export function Kpi({
  label, value, hint,
}: { label: string; value: ReactNode; hint?: ReactNode }) {
  return (
    <div className="kpi">
      <div className="kpi-label">{label}</div>
      <div className="kpi-value">{value}</div>
      {hint && <div className="kpi-hint">{hint}</div>}
    </div>
  );
}

export function Loading({ what = "…" }: { what?: string }) {
  return <p className="muted">Loading {what}</p>;
}

/**
 * A failed panel says so. It does NOT render an empty table — the backend's
 * reads fail soft and return [], and a component that cannot tell an empty
 * result from a failed one is how a broken query becomes a confident "no
 * tenants".
 */
export function ErrorNote({ error }: { error: unknown }) {
  const msg = error instanceof Error ? error.message : String(error);
  return <div className="err">{msg}</div>;
}

export function Empty({ children }: { children: ReactNode }) {
  return <p className="muted empty">{children}</p>;
}
