import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api } from "../api";
import { useFilters } from "../lib/filters";
import { ABSENT, cost, day, num, pct, productLabel, relative } from "../lib/format";
import { EnvBadge, ErrorNote, Loading } from "../components/Bits";

/**
 * The operator's first screen.
 *
 * ────────────────────────────────────────────────────────────────────────────
 * THE THESIS: COVERAGE BEFORE FIGURES.
 *
 * A conventional KPI dashboard would be dishonest here. Six of eight stores
 * present v1 JWTs, resolve no v2 context, and produce no ledger rows at all —
 * their spend is not zero, it is UNOBSERVED. Rendering a total over an estate
 * that is only partly instrumented, without saying so, is how a console gets
 * believed and is wrong.
 *
 * So the estate strip comes first and everything else is subordinate to it: an
 * operator learns how much of the system they can see before they read a single
 * number. Absence is drawn as TEXTURE — a dark hatched segment — rather than as
 * a colour or a zero, because it is a different kind of fact from a measurement.
 *
 * The page is deliberately dim, with luminance only where there is signal. An
 * instrument panel, not a marketing dashboard.
 * ────────────────────────────────────────────────────────────────────────────
 */

interface CoverageSite {
  site_id: string;
  domain: string;
  environment: string;
  event_rows: number;
  live_licences: number;
  reporting: boolean;
}

interface OverviewData {
  window_days: number;
  estate: { clients: number | null; sites: number | null; subscriptions: number | null; licences: number | null };
  coverage: { sites: CoverageSite[]; sites_total: number; sites_reporting: number };
  totals: { requests: number | null; cost: number | null; tokens: number | null; ledger_rows: number | null };
  series: { day: string; requests: number; cost: number; tokens: number }[];
  by_product: { product_code: string; requests: number; cost: number; tokens: number }[];
  by_key_owner: { key_owner: string; cost: number; rows: number }[];
  top_tenants: { client_id: string; name: string; requests: number; cost: number }[];
  licences_expiring_30d: {
    licence_id: string; key_prefix: string | null; product_code: string;
    domain: string; environment: string; client_name: string; expires_at: string | null;
  }[];
  sites_near_catalogue_ceiling: {
    site_id: string; domain: string; environment: string; index_plan: string;
    indexed_items: number; catalogue_limit: number; pct: number;
  }[];
  quota_enforced: boolean;
}

export function Overview() {
  const { days, envParam, environment } = useFilters();
  const qs = `?days=${days}${envParam()}`;

  const { data, isLoading, error } = useQuery<OverviewData>({
    queryKey: ["overview", days, environment],
    queryFn: () => api.get<OverviewData>(`/api/admin/overview${qs}`),
  });

  if (isLoading) return <Loading what="overview" />;
  if (error) return <ErrorNote error={error} />;
  if (!data) return null;

  return (
    <div className="ov">
      <EstateStrip coverage={data.coverage} />
      <Totals data={data} />
      <div className="ov-split">
        <CostTrend series={data.series} days={data.window_days} />
        <ByProduct rows={data.by_product} coverage={data.coverage} />
      </div>
      <Attention data={data} />
    </div>
  );
}

/* ── Signature: the estate strip ─────────────────────────────────────────── */

function EstateStrip({ coverage }: { coverage: OverviewData["coverage"] }) {
  const { sites, sites_total, sites_reporting } = coverage;
  const dark = sites_total - sites_reporting;

  return (
    <section className="ov-strip">
      <div className="ov-strip-head">
        <span className="eyebrow">Telemetry coverage</span>
        <span className="ov-strip-count">
          <strong>{sites_reporting}</strong>
          <span className="ov-strip-of">of {sites_total}</span>
          <span className="ov-strip-word">sites reporting</span>
        </span>
      </div>

      <div className="ov-bars" role="img"
           aria-label={`${sites_reporting} of ${sites_total} sites reporting telemetry`}>
        {sites.map((s) => (
          <div
            key={s.site_id}
            className={s.reporting ? "ov-bar is-lit" : "ov-bar is-dark"}
            // title carries the detail so the strip itself stays wordless —
            // it is a glance instrument, and labelling eight segments inline
            // would turn it back into a table.
            title={
              s.reporting
                ? `${s.domain} — ${num(s.event_rows)} ledger rows`
                : `${s.domain} — no telemetry` +
                  (s.live_licences === 0
                    ? " (no active licence)"
                    : " (licensed but silent)")
            }
          >
            <span className="ov-bar-label">{s.domain.replace(/^www\./, "")}</span>
          </div>
        ))}
      </div>

      {dark > 0 && (
        <p className="ov-strip-note">
          {dark} {dark === 1 ? "site emits" : "sites emit"} no telemetry in this
          window. Their spend is <strong>unobserved</strong>, not zero — every
          figure below covers the {sites_reporting} lit above.
        </p>
      )}
    </section>
  );
}

/* ── Totals ──────────────────────────────────────────────────────────────── */

function Totals({ data }: { data: OverviewData }) {
  const { totals, estate, by_key_owner, quota_enforced } = data;
  const funded = by_key_owner.find((k) => k.key_owner === "client");

  return (
    <section className="ov-totals">
      <Figure
        label="Billable requests"
        value={num(totals.requests)}
        // Rows and requests differ by ~4x and both are correct: one customer
        // action writes one billable row plus several for embeddings, reranks
        // and tool calls. Stating both stops the larger number being mistaken
        // for the smaller one.
        foot={`${num(totals.ledger_rows)} ledger rows total`}
      />
      <Figure
        label="Spend"
        value={cost(totals.cost)}
        foot={
          funded
            ? `${cost(funded.cost)} funded by merchants`
            : "all funded by Czargroup"
        }
      />
      <Figure label="Tokens" value={num(totals.tokens)} />
      <Figure
        label="Estate"
        value={num(estate.licences)}
        foot={`live licences · ${num(estate.subscriptions)} subscriptions · ${num(estate.clients)} clients`}
      />
      {!quota_enforced && (
        <p className="ov-quota">
          Request limits are recorded but <strong>not enforced</strong> —
          <code>AICHATBOT_QUOTA_ENFORCEMENT</code> is unset, so a tenant over
          plan keeps working.
        </p>
      )}
    </section>
  );
}

function Figure({ label, value, foot }: { label: string; value: string; foot?: string }) {
  const absent = value === ABSENT;
  return (
    <div className={absent ? "ov-fig is-absent" : "ov-fig"}>
      <div className="eyebrow">{label}</div>
      <div className="ov-fig-value">{value}</div>
      {foot && <div className="ov-fig-foot">{foot}</div>}
    </div>
  );
}

/* ── Cost trend, hand-rolled ─────────────────────────────────────────────── */

function CostTrend({ series, days }: { series: OverviewData["series"]; days: number }) {
  if (!series.length) {
    return (
      <Panel title={`Spend · ${days} days`}>
        <p className="ov-empty">
          No ledger rows in this window. Nothing has been measured yet — that is
          not the same as nothing having happened.
        </p>
      </Panel>
    );
  }

  const W = 640, H = 150, PAD = 4;
  const max = Math.max(...series.map((p) => p.cost), 0.000001);
  const step = series.length > 1 ? (W - PAD * 2) / (series.length - 1) : 0;
  const pt = (i: number, v: number) => [
    PAD + i * step,
    H - PAD - (v / max) * (H - PAD * 2),
  ];

  const line = series.map((p, i) => pt(i, p.cost).join(",")).join(" ");
  const area = `${PAD},${H - PAD} ${line} ${PAD + (series.length - 1) * step},${H - PAD}`;
  const peak = series.reduce((a, b) => (b.cost > a.cost ? b : a), series[0]);

  return (
    <Panel
      title={`Spend · ${days} days`}
      right={<span className="ov-peak">peak {cost(peak.cost)} · {day(peak.day)}</span>}
    >
      <svg className="ov-chart" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none"
           role="img" aria-label={`Daily spend over ${days} days, peaking at ${cost(peak.cost)}`}>
        <defs>
          <linearGradient id="ovFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--signal)" stopOpacity="0.28" />
            <stop offset="100%" stopColor="var(--signal)" stopOpacity="0" />
          </linearGradient>
        </defs>
        <polygon points={area} fill="url(#ovFill)" />
        <polyline points={line} fill="none" stroke="var(--signal)" strokeWidth="1.5"
                  vectorEffect="non-scaling-stroke" />
        {series.map((p, i) => {
          const [x, y] = pt(i, p.cost);
          return <circle key={p.day} cx={x} cy={y} r="2.5" fill="var(--signal)"
                         vectorEffect="non-scaling-stroke">
            <title>{`${day(p.day)} — ${cost(p.cost)}, ${num(p.requests)} requests`}</title>
          </circle>;
        })}
      </svg>
      <div className="ov-axis">
        <span>{day(series[0].day)}</span>
        <span>{day(series[series.length - 1].day)}</span>
      </div>
    </Panel>
  );
}

/* ── Per product ─────────────────────────────────────────────────────────── */

function ByProduct({
  rows, coverage,
}: { rows: OverviewData["by_product"]; coverage: OverviewData["coverage"] }) {
  const max = Math.max(...rows.map((r) => r.cost), 0.000001);
  const dark = coverage.sites_total - coverage.sites_reporting;

  return (
    <Panel title="By product">
      {rows.length === 0 ? (
        <p className="ov-empty">No attributed spend in this window.</p>
      ) : (
        <ul className="ov-prod">
          {rows.map((r) => (
            <li key={r.product_code}>
              <div className="ov-prod-top">
                <span className="ov-prod-name">{productLabel(r.product_code)}</span>
                <span className="ov-prod-cost">{cost(r.cost)}</span>
              </div>
              <div className="ov-prod-track">
                <div className="ov-prod-fill" style={{ width: `${(r.cost / max) * 100}%` }} />
              </div>
              <div className="ov-prod-foot">
                {num(r.requests)} requests · {num(r.tokens)} tokens
              </div>
            </li>
          ))}
        </ul>
      )}
      {dark > 0 && (
        <p className="ov-prod-note">
          Attribution comes off the licence, so the {dark} unlit{" "}
          {dark === 1 ? "site is" : "sites are"} absent from this breakdown
          entirely — not shown as zero.
        </p>
      )}
    </Panel>
  );
}

/* ── Attention ───────────────────────────────────────────────────────────── */

function Attention({ data }: { data: OverviewData }) {
  const { licences_expiring_30d: expiring, sites_near_catalogue_ceiling: ceiling } = data;

  // An ops screen with nothing wrong should say so, not render empty panels.
  if (!expiring.length && !ceiling.length) {
    return (
      <section className="ov-clear">
        <span className="ov-clear-mark" aria-hidden="true" />
        No licences expiring in 30 days, no site near its catalogue ceiling.
      </section>
    );
  }

  return (
    <div className="ov-split">
      {ceiling.length > 0 && (
        <Panel title="Catalogue ceiling">
          {/* The only limit actually enforced today — sites.indexed_items
              against catalogue_limit blocks writes for real, unlike request
              quota. Worth ranking above expiry for that reason. */}
          <ul className="ov-list">
            {ceiling.map((s) => (
              <li key={s.site_id}>
                <Link to={`/sites/${s.site_id}`} className="ov-list-main">{s.domain}</Link>
                <EnvBadge environment={s.environment} />
                <span className="ov-list-meta">
                  {num(s.indexed_items)} / {num(s.catalogue_limit)}
                </span>
                <span className={s.pct >= 95 ? "ov-pct is-fault" : "ov-pct is-ember"}>
                  {pct(s.pct)}
                </span>
              </li>
            ))}
          </ul>
        </Panel>
      )}

      {expiring.length > 0 && (
        <Panel title="Licences expiring">
          <ul className="ov-list">
            {expiring.map((l) => (
              <li key={l.licence_id}>
                <Link to={`/licences/${l.licence_id}`} className="ov-list-main mono">
                  {l.key_prefix ?? ABSENT}
                </Link>
                <EnvBadge environment={l.environment} />
                <span className="ov-list-meta">{productLabel(l.product_code)} · {l.domain}</span>
                <span className="ov-pct is-ember">{relative(l.expires_at)}</span>
              </li>
            ))}
          </ul>
        </Panel>
      )}
    </div>
  );
}

function Panel({
  title, children, right,
}: { title: string; children: React.ReactNode; right?: React.ReactNode }) {
  return (
    <section className="ov-panel">
      <header>
        <span className="eyebrow">{title}</span>
        {right}
      </header>
      <div className="ov-panel-body">{children}</div>
    </section>
  );
}
