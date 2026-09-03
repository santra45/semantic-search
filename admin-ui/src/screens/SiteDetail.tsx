import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import { api } from "../api";
import { useFilters } from "../lib/filters";
import { ABSENT, num, pct, productLabel, when } from "../lib/format";
import { ErrorNote, Loading } from "../components/Bits";

/**
 * One store install.
 *
 * ────────────────────────────────────────────────────────────────────────────
 * TWO KINDS OF LIMIT, DRAWN DIFFERENTLY.
 *
 * This is the only screen that shows both, and they are not the same kind of
 * fact:
 *
 *   CATALOGUE CEILING  sites.indexed_items against catalogue_limit. Enforced —
 *                      writes past it are refused. A real gauge.
 *   REQUEST QUOTA      usage_counters against subscriptions.request_limit.
 *                      NOT enforced: AICHATBOT_QUOTA_ENFORCEMENT is unset, so a
 *                      tenant at 300% keeps working.
 *
 * Rendering four identical bars would imply four identical ceilings, and an
 * operator would reasonably conclude they had capped something. The enforced
 * one is solid; the observational ones are dashed and say so. Same data shape,
 * different epistemic status, and the drawing carries the difference.
 *
 * `requests_this_period` is null when no counter row exists for the period —
 * nothing billable recorded yet, which is not zero requests.
 * ────────────────────────────────────────────────────────────────────────────
 */

interface Sub {
  id: string;
  product_code: string;
  status: string;
  plan: string;
  request_limit: number;
  disabled_reason: string | null;
  expires_at: string | null;
  active_licences: number;
  requests_this_period: number | null;
}

interface SiteDetailData {
  site: {
    id: string; domain: string; platform: string; platform_version: string | null;
    store_name: string | null; collection_name: string | null; environment: string;
    index_plan: string; catalogue_limit: number; indexed_items: number;
    catalogue_pct: number | null; is_active: boolean; created_at: string | null;
    client_id: string; client_name: string;
  };
  subscriptions: Sub[];
  window_days: number;
  quota_enforced: boolean;
}

export function SiteDetail() {
  const { siteId } = useParams();
  const { days } = useFilters();

  const { data, isLoading, error } = useQuery<SiteDetailData>({
    queryKey: ["site", siteId, days],
    queryFn: () => api.get<SiteDetailData>(`/api/admin/sites/${siteId}?days=${days}`),
  });

  if (isLoading) return <Loading what="site" />;
  if (error) return <ErrorNote error={error} />;
  if (!data) return null;

  const { site, subscriptions, quota_enforced } = data;
  const limit = site.catalogue_limit;
  const used = site.indexed_items;
  const pctUsed = limit > 0 ? (used / limit) * 100 : null;
  const tone = pctUsed === null ? "" : pctUsed >= 95 ? " is-fault" : pctUsed >= 80 ? " is-ember" : "";

  return (
    <div className="sd">
      <nav className="sd-crumb">
        <Link to="/tenants">Tenants</Link>
        <span aria-hidden="true">/</span>
        <Link to={`/tenants/${site.client_id}`}>{site.client_name}</Link>
        <span aria-hidden="true">/</span>
        <span>{site.domain}</span>
      </nav>

      <header className={site.is_active ? "sd-head" : "sd-head is-off"}>
        <div>
          <h1>
            {site.domain}
            {site.environment === "production"
              ? <>{" "}<span className="pill pill-ok">prod</span></>
              : <>{" "}<span className="pill pill-muted">dev</span></>}
            {!site.is_active && <>{" "}<span className="pill pill-bad">suspended</span></>}
          </h1>
          <p className="sd-meta">
            {site.platform}
            {site.platform_version && ` ${site.platform_version}`}
            {site.store_name && ` · ${site.store_name}`}
            {" · registered "}{when(site.created_at)}
          </p>
          {site.collection_name && (
            <p className="sd-collection">
              {/* Named because it is the one identifier that maps this store to
                  what is actually in Qdrant — the first thing anyone needs when
                  a sync goes wrong. */}
              <span className="eyebrow">Qdrant collection</span>
              <code>{site.collection_name}</code>
            </p>
          )}
        </div>
      </header>

      {/* ── The enforced ceiling. Solid, and the only one. ── */}
      <section className="sd-gauge-card">
        <div className="sd-gauge-head">
          <div>
            <span className="eyebrow">Catalogue ceiling · {site.index_plan}</span>
            <p className="sd-gauge-note">
              Enforced. Writes past this are refused, and a downgrade below what
              is already indexed is rejected — a store over its own ceiling has
              no clean way out.
            </p>
          </div>
          <span className="sd-gauge-num">
            {num(used)}<span> / {limit > 0 ? num(limit) : "unlimited"}</span>
          </span>
        </div>
        <div className="sd-track">
          <div className={`sd-fill${tone}`}
               style={{ width: pctUsed === null ? "0%" : `${Math.min(pctUsed, 100)}%` }} />
        </div>
        <div className="sd-gauge-foot">
          {pctUsed === null ? "no ceiling on this plan" : `${pct(pctUsed)} used`}
          {" · counts every logical entity — products, CMS pages, categories, FAQs"}
        </div>
      </section>

      <section className="sd-subs-card">
        <header className="sd-subs-head">
          <span className="eyebrow">Modules on this store</span>
          {!quota_enforced && (
            <span className="pill pill-warn">request limits not enforced</span>
          )}
        </header>

        {subscriptions.length === 0 ? (
          <p className="sd-empty">
            No modules on this store. It is registered and licensable, but
            nothing has been sold to it.
          </p>
        ) : (
          <ul className="sd-subs">
            {subscriptions.map((s) => (
              <SubRow key={s.id} s={s} enforced={quota_enforced} clientId={site.client_id} />
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}

function SubRow({
  s, enforced, clientId,
}: { s: Sub; enforced: boolean; clientId: string }) {
  const used = s.requests_this_period;
  const limit = s.request_limit;
  const burn = used !== null && limit > 0 ? (used / limit) * 100 : null;
  const live = s.status === "active" || s.status === "trial";

  return (
    <li className={live ? "sd-sub" : "sd-sub is-off"}>
      <div className="sd-sub-top">
        <span className="sd-sub-name">{productLabel(s.product_code)}</span>
        <span className={live ? "pill pill-ok" : "pill pill-bad"}>{s.status}</span>
        <span className="sd-sub-plan">{s.plan}</span>
        <span className="sd-sub-keys">
          {s.active_licences === 0
            ? "no active key"
            : `${num(s.active_licences)} active ${s.active_licences === 1 ? "key" : "keys"}`}
        </span>
      </div>

      {/* Dashed, and labelled "observational". This is the same shape as the
          catalogue gauge above and deliberately does not look like it. */}
      <div className="sd-quota">
        <div className="sd-quota-top">
          <span>
            {used === null ? ABSENT : num(used)}
            <em> / {num(limit)} this period</em>
          </span>
          <span className="sd-quota-tag">
            {enforced ? "enforced" : "observational"}
          </span>
        </div>
        <div className={enforced ? "sd-track" : "sd-track is-ghost"}>
          <div className={enforced ? "sd-fill" : "sd-fill is-ghost"}
               style={{ width: burn === null ? "0%" : `${Math.min(burn, 100)}%` }} />
        </div>
        {used === null && (
          <p className="sd-quota-none">
            No counter row for this period — nothing billable recorded yet,
            which is not the same as zero requests.
          </p>
        )}
      </div>

      {s.disabled_reason && (
        <p className="sd-sub-reason">
          <span className="eyebrow">Shown to the merchant</span>
          {s.disabled_reason}
        </p>
      )}

      <p className="sd-sub-foot">
        {s.expires_at ? `expires ${when(s.expires_at)}` : "no expiry"}
        {" · "}
        <Link to={`/tenants/${clientId}`}>manage on the customer page</Link>
      </p>
    </li>
  );
}
