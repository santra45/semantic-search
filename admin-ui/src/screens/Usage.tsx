import { useState } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { api } from "../api";
import { useFilters } from "../lib/filters";
import { ABSENT, cost, num, productLabel } from "../lib/format";
import { ErrorNote, Loading } from "../components/Bits";

/**
 * Where the money goes.
 *
 * ────────────────────────────────────────────────────────────────────────────
 * REGROUP, DO NOT JUST FILTER.
 *
 * The dimension switcher is the primary control and the composition bar is the
 * same total re-partitioned each time — turning one quantity to a different
 * angle rather than loading a different page. "Most of the spend is one model"
 * is a shape, and a table of numbers is a poor way to show a shape.
 *
 * TWO DENOMINATORS, AND ONLY ONE OF THEM DIVIDES.
 *
 * usage_events writes ONE billable row per customer action plus several
 * non-billable ones for embeddings, reranks and tool calls, so requests and
 * ledger rows differ by roughly 4x and BOTH are correct. Cost per REQUEST is a
 * real unit economic; cost per ROW is meaningless. The screen shows both counts
 * side by side, and divides only by requests — and refuses to divide at all
 * where requests is zero, which happens for genuine reasons: a catalogue sync
 * is pure non-billable work that still costs money.
 *
 * Grouping by `kind` or `call_type` is how the 4x becomes legible instead of
 * being a footnote. That is worth making easy to reach rather than hiding.
 * ────────────────────────────────────────────────────────────────────────────
 */

const DIMENSIONS = [
  { key: "product", label: "Product" },
  { key: "model", label: "Model" },
  { key: "call_type", label: "Call type" },
  { key: "kind", label: "Kind" },
  { key: "client", label: "Customer" },
  { key: "site", label: "Store" },
  { key: "environment", label: "Environment" },
  { key: "key_owner", label: "Funded by" },
  { key: "day", label: "Day" },
] as const;

interface Row {
  bucket: string | null;
  requests: number;
  cost: number;
  tokens: number;
  input_tokens: number;
  output_tokens: number;
  ledger_rows: number;
}

interface UsageData {
  group_by: string;
  window_days: number;
  dimensions: string[];
  rows: Row[];
  totals: {
    requests: number | null;
    cost: number | null;
    tokens: number | null;
    ledger_rows: number | null;
  };
}

export function Usage() {
  const { days, environment, envParam } = useFilters();
  const [groupBy, setGroupBy] = useState("product");

  const { data, isLoading, error, isFetching } = useQuery<UsageData>({
    queryKey: ["usage", groupBy, days, environment],
    queryFn: () =>
      api.get<UsageData>(`/api/admin/usage?group_by=${groupBy}&days=${days}${envParam()}`),
    placeholderData: keepPreviousData,
  });

  const totalCost = data?.totals.cost ?? 0;
  const ranked = (data?.rows ?? []).slice().sort((a, b) => b.cost - a.cost);

  return (
    <div className="us">
      <div className="us-dims" role="group" aria-label="Group by">
        {DIMENSIONS.map((d) => (
          <button
            key={d.key}
            className={groupBy === d.key ? "us-dim is-on" : "us-dim"}
            onClick={() => setGroupBy(d.key)}
            aria-pressed={groupBy === d.key}
          >
            {d.label}
          </button>
        ))}
      </div>

      {error ? <ErrorNote error={error} />
      : isLoading ? <Loading what="usage" />
      : !data ? null
      : (
        <div className={isFetching ? "us-body is-stale" : "us-body"}>
          <Composition rows={ranked} total={totalCost} groupBy={groupBy} days={days} />
          <Table rows={ranked} total={totalCost} totals={data.totals} groupBy={groupBy} />
        </div>
      )}
    </div>
  );
}

/** Rank-ordered opacity rather than a categorical palette.
 *
 *  --signal means MEASURED everywhere else in this console, and handing out six
 *  hues here would dilute that. A single-hue ramp also encodes what the bar is
 *  actually sorted by — magnitude — instead of decorating categories that
 *  change every time the dimension does. */
function shade(rank: number, count: number): number {
  if (count <= 1) return 0.85;
  return 0.85 - (rank / Math.max(count - 1, 1)) * 0.6;
}

function Composition({
  rows, total, groupBy, days,
}: { rows: Row[]; total: number; groupBy: string; days: number }) {
  if (!rows.length || total <= 0) {
    return (
      <section className="us-comp is-empty">
        <span className="eyebrow">Spend by {groupBy} · {days} days</span>
        <p className="us-empty">
          No cost recorded in this window. That is not the same as no activity —
          six of eight stores present v1 keys and produce no ledger rows at all.
        </p>
      </section>
    );
  }

  return (
    <section className="us-comp">
      <div className="us-comp-head">
        <span className="eyebrow">Spend by {groupBy} · {days} days</span>
        <span className="us-comp-total">{cost(total)}</span>
      </div>

      <div className="us-bar" role="img"
           aria-label={`Spend split by ${groupBy}: ` +
             rows.slice(0, 5).map((r) => `${label(r.bucket, groupBy)} ${Math.round((r.cost / total) * 100)}%`).join(", ")}>
        {rows.map((r, i) => {
          const share = (r.cost / total) * 100;
          if (share <= 0) return null;
          return (
            <span
              key={String(r.bucket)}
              className="us-seg"
              style={{ width: `${share}%`, opacity: shade(i, rows.length) }}
              title={`${label(r.bucket, groupBy)} — ${cost(r.cost)} (${share.toFixed(1)}%)`}
            />
          );
        })}
      </div>

      <ul className="us-legend">
        {rows.slice(0, 6).map((r, i) => (
          <li key={String(r.bucket)}>
            <span className="us-swatch" style={{ opacity: shade(i, rows.length) }} />
            {label(r.bucket, groupBy)}
            <em>{((r.cost / total) * 100).toFixed(1)}%</em>
          </li>
        ))}
        {rows.length > 6 && <li className="us-legend-more">+{rows.length - 6} more below</li>}
      </ul>
    </section>
  );
}

function Table({
  rows, total, totals, groupBy,
}: { rows: Row[]; total: number; totals: UsageData["totals"]; groupBy: string }) {
  return (
    <section className="us-table">
      <div className="us-head">
        <span className="eyebrow">{groupBy}</span>
        <span className="eyebrow us-r">Cost</span>
        <span className="eyebrow us-r">Share</span>
        <span className="eyebrow us-r">Requests</span>
        <span className="eyebrow us-r">Ledger rows</span>
        <span className="eyebrow us-r">Tokens</span>
        <span className="eyebrow us-r">Per request</span>
      </div>

      {rows.map((r) => {
        const share = total > 0 ? (r.cost / total) * 100 : 0;
        return (
          <div className="us-row" key={String(r.bucket)}>
            {/* The share bar sits behind the label rather than in its own
                column: it is context for the row, not a value to compare
                against its neighbours cell by cell. */}
            <span className="us-name">
              <span className="us-rowbar" style={{ width: `${share}%` }} aria-hidden="true" />
              <span className="us-name-text">{label(r.bucket, groupBy)}</span>
            </span>
            <span className="us-r us-num">{cost(r.cost)}</span>
            <span className="us-r us-num us-dim-text">{share.toFixed(1)}%</span>
            <span className="us-r us-num">{num(r.requests)}</span>
            {/* Shown beside requests on purpose. They differ by ~4x for correct
                reasons and the larger is otherwise mistaken for the smaller. */}
            <span className="us-r us-num us-dim-text">{num(r.ledger_rows)}</span>
            <span className="us-r us-num us-dim-text">{num(r.tokens)}</span>
            <span className="us-r us-num">{perRequest(r)}</span>
          </div>
        );
      })}

      <div className="us-total">
        <span>Total</span>
        <span className="us-r us-num">{cost(totals.cost)}</span>
        <span className="us-r" />
        <span className="us-r us-num">{num(totals.requests)}</span>
        <span className="us-r us-num us-dim-text">{num(totals.ledger_rows)}</span>
        <span className="us-r us-num us-dim-text">{num(totals.tokens)}</span>
        <span className="us-r us-num">
          {totals.requests && totals.cost
            ? cost(totals.cost / totals.requests)
            : ABSENT}
        </span>
      </div>
    </section>
  );
}

/**
 * Cost per BILLABLE request, or an em dash.
 *
 * Never divides by ledger rows, and never renders a number when requests is
 * zero — which is a real state, not an error: a catalogue sync is entirely
 * non-billable work that still costs money, so its cost is real and its
 * per-request figure does not exist. Showing 0, or infinity, would both be
 * inventions.
 */
function perRequest(r: Row): string {
  if (!r.requests) return ABSENT;
  return cost(r.cost / r.requests);
}

function label(bucket: string | null, groupBy: string): string {
  if (bucket === null || bucket === "") return "(unattributed)";
  if (groupBy === "product") return productLabel(bucket);
  if (groupBy === "key_owner") return bucket === "czargroup" ? "Czargroup" : "Merchant";
  return bucket;
}
