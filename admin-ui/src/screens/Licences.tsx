import { useEffect, useState } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api } from "../api";
import { useActions } from "../lib/actions";
import { useCan } from "../lib/session";
import { useFilters } from "../lib/filters";
import { ABSENT, productLabel, relative, when } from "../lib/format";
import { ErrorNote, Loading } from "../components/Bits";
import { openRevoke } from "./tenantActions";

/**
 * Every key in the estate.
 *
 * ────────────────────────────────────────────────────────────────────────────
 * THIS SCREEN HANDLES CREDENTIALS, AND THE LAYOUT SAYS SO.
 *
 * licences.licence_key holds plaintext since 2026-09-03 — that is what makes
 * "resend my key" answerable without rotating a working install, and it is also
 * what makes this table a secrets table.
 *
 * So the list NEVER carries a usable key. Rows show the prefix; the full value
 * comes from a separate owner-only endpoint, one key at a time, and every
 * reveal is logged at WARNING server-side. Putting the plaintext in the list
 * payload would spray every credential in the estate into browser memory and
 * proxy logs on the first page load — the same mistake as logging it, at
 * greater scale.
 *
 * `has_plaintext = false` is its own state, not an error: those keys were
 * minted before the column existed and can NEVER be shown, only rotated. The
 * row says so rather than offering a reveal button that would fail.
 * ────────────────────────────────────────────────────────────────────────────
 */

interface LicenceRow {
  id: string;
  key_prefix: string | null;
  has_plaintext: boolean;
  is_active: boolean;
  subscription_id: string;
  subscription_status: string;
  product_code: string;
  site_id: string;
  domain: string;
  environment: string;
  client_id: string;
  client_name: string;
  issued_at: string | null;
  expires_at: string | null;
  revoked_at: string | null;
  last_seen: string | null;
}

interface LicencesData {
  total: number;
  limit: number;
  offset: number;
  licences: LicenceRow[];
}

const STATUSES = [
  { value: "", label: "All keys" },
  { value: "active", label: "Active" },
  { value: "expired", label: "Expired" },
  { value: "revoked", label: "Revoked" },
];

export function Licences() {
  const { environment, envParam } = useFilters();
  const isOwner = useCan("owner");
  const act = useActions([["licences"], ["tenants"], ["overview"]]);

  const [search, setSearch] = useState("");
  const [debounced, setDebounced] = useState("");
  const [status, setStatus] = useState("active");
  const [expiring, setExpiring] = useState("");
  const [offset, setOffset] = useState(0);
  const limit = 25;

  useEffect(() => {
    const t = setTimeout(() => { setDebounced(search); setOffset(0); }, 250);
    return () => clearTimeout(t);
  }, [search]);

  const qs =
    `?limit=${limit}&offset=${offset}` + envParam() +
    (status ? `&status=${status}` : "") +
    (expiring ? `&expiring_days=${expiring}` : "") +
    (debounced ? `&search=${encodeURIComponent(debounced)}` : "");

  const { data, isLoading, error, isFetching } = useQuery<LicencesData>({
    queryKey: ["licences", environment, status, expiring, debounced, offset],
    queryFn: () => api.get<LicencesData>(`/api/admin/licences${qs}`),
    placeholderData: keepPreviousData,
  });

  return (
    <div className="lc">
      {act.sheetNode}

      <div className="lc-controls">
        <input
          className="lc-search"
          type="search"
          placeholder="Search domain or customer"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          aria-label="Search licences"
        />
        <select value={status} onChange={(e) => { setStatus(e.target.value); setOffset(0); }}
                aria-label="Filter by status">
          {STATUSES.map((s) => <option key={s.value} value={s.value}>{s.label}</option>)}
        </select>
        <select value={expiring} onChange={(e) => { setExpiring(e.target.value); setOffset(0); }}
                aria-label="Filter by expiry">
          <option value="">Any expiry</option>
          <option value="7">Expiring in 7 days</option>
          <option value="30">Expiring in 30 days</option>
          <option value="90">Expiring in 90 days</option>
        </select>
      </div>

      {/* Said once, at the top, rather than on every row. */}
      <p className="lc-note">
        Keys are shown by prefix. The full value is revealed one at a time, to
        owners, and every reveal is logged.
      </p>

      {error ? <ErrorNote error={error} />
      : isLoading ? <Loading what="licences" />
      : !data ? null
      : data.licences.length === 0 ? (
        <p className="lc-none">
          No key matches these filters. Widen the status or clear the search.
        </p>
      ) : (
        <>
          <div className={isFetching ? "lc-table is-stale" : "lc-table"}>
            {data.licences.map((l) => (
              <LicenceCard key={l.id} l={l} isOwner={isOwner} act={act} />
            ))}
          </div>
          <div className="lc-foot">
            <span>{offset + 1}–{Math.min(offset + limit, data.total)} of {data.total}</span>
            <div className="lc-pager">
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

function LicenceCard({
  l, isOwner, act,
}: { l: LicenceRow; isOwner: boolean; act: ReturnType<typeof useActions> }) {
  const expired = !!l.expires_at && new Date(l.expires_at) <= new Date();
  const dead = !l.is_active || expired;

  const state = !l.is_active ? { tone: "bad", word: "revoked" }
    : expired ? { tone: "bad", word: "expired" }
    : { tone: "ok", word: "active" };

  return (
    <article className={dead ? "lc-row is-dead" : "lc-row"}>
      <span className={`lc-dot is-${state.tone}`} aria-hidden="true" />

      <span className="lc-key">
        <Link to={`/licences/${l.id}`} className="lc-prefix">
          {l.key_prefix ?? ABSENT}
        </Link>
        <span className="lc-where">
          {productLabel(l.product_code)} · {l.domain}
        </span>
      </span>

      <span className="lc-owner">
        <Link to={`/tenants/${l.client_id}`}>{l.client_name}</Link>
        {l.environment === "production"
          ? <>{" "}<span className="pill pill-ok">prod</span></>
          : <>{" "}<span className="pill pill-muted">dev</span></>}
      </span>

      <span className="lc-dates">
        <span className={`pill pill-${state.tone}`}>{state.word}</span>
        <span className="lc-when">
          {l.revoked_at ? `revoked ${when(l.revoked_at)}`
            : l.expires_at ? `expires ${relative(l.expires_at)}`
            : "no expiry"}
        </span>
      </span>

      <span className="lc-seen">
        {/* Attributed via the subscription, because usage_events carries no
            licence_id — so this survives a rotation and is labelled as the
            subscription's activity rather than this key's. */}
        {l.last_seen ? `subscription last used ${relative(l.last_seen)}` : "never used"}
      </span>

      <span className="lc-actions">
        {isOwner && <RevealButton licence={l} />}
        {isOwner && l.is_active && (
          <button className="td-btn is-danger" onClick={() => openRevoke(act, l)}>
            Revoke
          </button>
        )}
      </span>
    </article>
  );
}

/**
 * One key, on demand.
 *
 * Deliberately NOT prefetched with the row. The endpoint logs every call at
 * WARNING, so fetching all twenty-five on page load would fill the log with
 * reveals nobody asked for and make the real ones impossible to find.
 */
function RevealButton({ licence }: { licence: LicenceRow }) {
  const [shown, setShown] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [failed, setFailed] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  if (!licence.has_plaintext) {
    return (
      <span className="lc-nokey" title="Minted before plaintext was stored — SHA-256 is one-way">
        cannot be shown
      </span>
    );
  }

  if (shown) {
    return (
      <span className="lc-shown">
        <code>{shown}</code>
        <button
          className="td-btn"
          onClick={() => {
            navigator.clipboard?.writeText(shown).then(
              () => { setCopied(true); setTimeout(() => setCopied(false), 1500); },
              () => setCopied(false),
            );
          }}
        >
          {copied ? "Copied" : "Copy"}
        </button>
        <button className="td-btn" onClick={() => setShown(null)}>Hide</button>
      </span>
    );
  }

  return (
    <>
      <button
        className="td-btn"
        disabled={busy}
        onClick={async () => {
          setBusy(true);
          setFailed(null);
          try {
            const r = await api.get<{ key: string }>(`/api/admin/licences/${licence.id}/key`);
            setShown(r.key);
          } catch (e) {
            setFailed(e instanceof Error ? e.message : String(e));
          } finally {
            setBusy(false);
          }
        }}
      >
        {busy ? "…" : "Show key"}
      </button>
      {failed && <span className="lc-failed">{failed}</span>}
    </>
  );
}
