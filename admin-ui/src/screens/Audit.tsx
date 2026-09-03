import { useEffect, useState } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api } from "../api";
import { useFilters } from "../lib/filters";
import { ABSENT, when } from "../lib/format";
import { ErrorNote, Loading } from "../components/Bits";

/**
 * Who changed what.
 *
 * ────────────────────────────────────────────────────────────────────────────
 * AN ENTRY IS A DIFF, NOT A ROW.
 *
 * The usual audit viewer is a table of action names, opaque ids and a JSON blob
 * nobody reads. But before_json and after_json are COMPLETE row snapshots — the
 * backend stores whole rows precisely so a reader years later does not need the
 * code to know what a row looked like — which means the screen can compute the
 * field-level change and show that instead. "status: trial → suspended" is the
 * fact; the JSON is the packaging.
 *
 * THE OTHER HALF IS `evicted`, AND THIS IS THE ONLY SCREEN THAT SHOWS IT.
 *
 * auth_cache holds a resolved licence context for 300 seconds and eviction is
 * the caller's job, so a write that returns 200 and forgets nothing looks
 * completely successful while changing nothing for five minutes. The column
 * exists to make that visible after the fact, and burying it in a table cell
 * would waste it — so an action that SHOULD have evicted and did not is called
 * out in the entry itself.
 *
 *   null  the action had no cached state behind it — correct, not a problem
 *   0     it had one and forgot nothing — the bug, flagged
 *   -1    the eviction itself failed; the write IS live and the cache is not
 *   n     n contexts forgotten
 * ────────────────────────────────────────────────────────────────────────────
 */

interface Entry {
  id: number;
  admin_user_id: string | null;
  actor_email: string;
  is_break_glass: boolean;
  action: string;
  target_type: string;
  target_id: string;
  before: unknown;
  after: unknown;
  reason: string | null;
  evicted: number | null;
  ip: string | null;
  created_at: string | null;
}

interface AuditData {
  total: number;
  limit: number;
  offset: number;
  window_days: number;
  entries: Entry[];
}

/** The backend's own action vocabulary. Fixed, so a select beats a text box. */
const ACTIONS = [
  "client.disable", "client.enable",
  "site.disable", "site.enable", "site.index_plan", "site.environment",
  "subscription.create", "subscription.pause", "subscription.resume",
  "subscription.plan", "subscription.term",
  "licence.issue", "licence.rotate", "licence.revoke",
  "product.withdraw", "product.restore",
];

/** Actions where evicting nothing is genuinely fine — they touch no cached
 *  context. Anything else reporting 0 is worth a second look. */
const NO_CACHE_ACTIONS = new Set([
  "product.withdraw", "product.restore", "subscription.create",
]);

export function Audit() {
  const { days } = useFilters();
  const [actor, setActor] = useState("");
  const [debouncedActor, setDebouncedActor] = useState("");
  const [action, setAction] = useState("");
  const [offset, setOffset] = useState(0);
  const limit = 50;

  useEffect(() => {
    const t = setTimeout(() => { setDebouncedActor(actor); setOffset(0); }, 250);
    return () => clearTimeout(t);
  }, [actor]);

  const qs =
    `?days=${days}&limit=${limit}&offset=${offset}` +
    (debouncedActor ? `&actor_email=${encodeURIComponent(debouncedActor)}` : "") +
    (action ? `&action=${action}` : "");

  const { data, isLoading, error, isFetching } = useQuery<AuditData>({
    queryKey: ["audit", days, debouncedActor, action, offset],
    queryFn: () => api.get<AuditData>(`/api/admin/audit${qs}`),
    placeholderData: keepPreviousData,
  });

  return (
    <div className="au">
      <div className="au-controls">
        <input
          className="au-search"
          type="search"
          placeholder="Filter by who"
          value={actor}
          onChange={(e) => setActor(e.target.value)}
          aria-label="Filter by actor"
        />
        <select value={action} onChange={(e) => { setAction(e.target.value); setOffset(0); }}
                aria-label="Filter by action">
          <option value="">All actions</option>
          {ACTIONS.map((a) => <option key={a} value={a}>{a}</option>)}
        </select>
      </div>

      {error ? <ErrorNote error={error} />
      : isLoading ? <Loading what="the audit log" />
      : !data ? null
      : data.entries.length === 0 ? (
        <p className="au-none">
          Nothing recorded in this window. The log is append-only and written in
          the same transaction as each change, so an empty result means nothing
          was changed — not that something failed to record.
        </p>
      ) : (
        <>
          <ol className="au-feed">
            {data.entries.map((e) => <EntryCard key={e.id} e={e} stale={isFetching} />)}
          </ol>
          <div className="au-foot">
            <span>{offset + 1}–{Math.min(offset + limit, data.total)} of {data.total}</span>
            <div className="au-pager">
              <button className="linkbtn" disabled={offset === 0}
                      onClick={() => setOffset(Math.max(0, offset - limit))}>Previous</button>
              <button className="linkbtn" disabled={offset + limit >= data.total}
                      onClick={() => setOffset(offset + limit)}>Next</button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

/** MySQL JSON arrives as an object through some drivers and a string through
 *  others. Parsed defensively rather than assuming, because guessing wrong
 *  turns every diff into "[object Object]". */
function asObject(v: unknown): Record<string, unknown> | null {
  if (v === null || v === undefined) return null;
  if (typeof v === "string") {
    try { const p = JSON.parse(v); return typeof p === "object" && p ? p as Record<string, unknown> : null; }
    catch { return null; }
  }
  if (typeof v === "object") return v as Record<string, unknown>;
  return null;
}

function fmt(v: unknown): string {
  if (v === null || v === undefined || v === "") return ABSENT;
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

function EntryCard({ e, stale }: { e: Entry; stale: boolean }) {
  const before = asObject(e.before);
  const after = asObject(e.after);

  // Only the fields that actually moved. A snapshot has a dozen columns and
  // eleven of them are noise on any given change.
  const changes: { field: string; from: unknown; to: unknown }[] = [];
  if (before && after) {
    for (const k of new Set([...Object.keys(before), ...Object.keys(after)])) {
      // Identity columns never "change" in a useful sense and would appear on
      // every entry as from===to noise if compared.
      if (k === "id") continue;
      if (JSON.stringify(before[k]) !== JSON.stringify(after[k])) {
        changes.push({ field: k, from: before[k], to: after[k] });
      }
    }
  }

  const evictionProblem =
    e.evicted === 0 && !NO_CACHE_ACTIONS.has(e.action)
      ? "evicted nothing — if a key was live, this took up to 5 minutes to bite"
      : e.evicted === -1
      ? "the eviction itself failed — the change is live, the cache is not"
      : null;

  const [group, verb] = e.action.includes(".") ? e.action.split(".") : [e.action, ""];

  return (
    <li className={stale ? "au-entry is-stale" : "au-entry"}>
      <div className="au-line">
        <span className="au-action">
          <span className="au-group">{group}</span>
          {verb && <span className="au-verb">{verb}</span>}
        </span>
        <span className="au-time">{when(e.created_at)}</span>
      </div>

      <div className="au-who">
        {e.is_break_glass ? (
          // Unattributable owner-level access. Called out rather than rendered
          // as just another email, because "somebody with the operator key" is
          // a much weaker answer to "who did this" than a named account.
          <span className="pill pill-warn">break-glass</span>
        ) : (
          <span className="au-actor">{e.actor_email}</span>
        )}
        <TargetLink type={e.target_type} id={e.target_id} />
        {e.ip && <span className="au-ip">{e.ip}</span>}
        <span className={evictionProblem ? "au-evict is-bad" : "au-evict"}>
          {e.evicted === null ? "no cached state"
            : e.evicted === -1 ? "eviction failed"
            : `evicted ${e.evicted}`}
        </span>
      </div>

      {evictionProblem && <p className="au-warn">{evictionProblem}</p>}

      {changes.length > 0 && (
        <dl className="au-diff">
          {changes.map((c) => (
            <div key={c.field}>
              <dt>{c.field}</dt>
              <dd>
                <span className="au-from">{fmt(c.from)}</span>
                <span className="au-arrow" aria-label="changed to">→</span>
                <span className="au-to">{fmt(c.to)}</span>
              </dd>
            </div>
          ))}
        </dl>
      )}

      {/* A create has no before, so there is nothing to diff — say that rather
          than rendering an empty block that looks like a failure. */}
      {changes.length === 0 && !before && after && (
        <p className="au-created">Created. No previous state to compare.</p>
      )}

      {e.reason && <blockquote className="au-reason">{e.reason}</blockquote>}
    </li>
  );
}

/** Deep-link where the console has a screen for it; plain text where it does
 *  not, rather than a link that goes nowhere. */
function TargetLink({ type, id }: { type: string; id: string }) {
  const short = id.length > 12 ? `${id.slice(0, 8)}…` : id;
  const to =
    type === "client" ? `/tenants/${id}`
    : type === "licence" ? `/licences/${id}`
    : null;
  return to ? (
    <Link to={to} className="au-target">{type} {short}</Link>
  ) : (
    <span className="au-target is-plain">{type} {short}</span>
  );
}
