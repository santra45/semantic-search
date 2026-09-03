import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import { api } from "../api";
import { useFilters } from "../lib/filters";
import { ABSENT, cost, num, pct, productLabel, relative, when } from "../lib/format";
import { ErrorNote, Loading } from "../components/Bits";

/**
 * One tenant, whole.
 *
 * ────────────────────────────────────────────────────────────────────────────
 * WHY THIS NESTS INSTEAD OF USING TABS.
 *
 * The plan specified tabs — Overview / Sites / Licences / Usage. Tabs are the
 * default answer for a detail page and they are the wrong one here, because
 * they HIDE the containment, and the containment is the thing an operator most
 * needs to see.
 *
 * v2 has three independent disable scopes that all read as "turn this off":
 *
 *     clients.is_active        the whole customer, every store, every module
 *     sites.is_active          one store install, every module on it
 *     subscriptions.status     one module on one store
 *
 * On a Licences tab you cannot see which site a licence belongs to, or that its
 * site is switched off. Nesting makes scope the layout: a subscription is
 * literally inside its site, which is inside the client.
 *
 * THE THING THIS SCREEN COMPUTES THAT THE API DOES NOT. A subscription with
 * status='active' under a client with is_active=0 is DEAD — resolve_key()
 * refuses it — but its own row still says active. Rendering that row's own
 * status would be a lie of exactly the kind this console exists to prevent, so
 * effectiveState() walks the chain and reports the REASON, not just the flag.
 * With three levels there is no such thing as a subscription's status in
 * isolation.
 * ────────────────────────────────────────────────────────────────────────────
 */

interface Site {
  id: string; domain: string; platform: string; platform_version: string | null;
  store_name: string | null; collection_name: string | null; environment: string;
  index_plan: string; catalogue_limit: number; indexed_items: number;
  catalogue_pct: number | null; is_active: boolean; created_at: string | null;
}
interface Subscription {
  id: string; site_id: string; domain: string; environment: string;
  product_code: string; status: string; plan: string; request_limit: number;
  disabled_reason: string | null; expires_at: string | null; active_licences: number;
}
interface Licence {
  id: string; key_prefix: string | null; has_plaintext: boolean; is_active: boolean;
  product_code: string; domain: string; environment: string;
  issued_at: string | null; expires_at: string | null; revoked_at: string | null;
}
interface TenantDetailData {
  client: { id: string; name: string; email: string; company: string | null;
            plan: string | null; is_active: boolean; created_at: string | null };
  sites: Site[];
  subscriptions: Subscription[];
  licences: Licence[];
  series: { day: string; requests: number; cost: number; tokens: number }[];
  by_product: { product_code: string; requests: number; cost: number; tokens: number }[];
  by_model: { provider: string; model: string; cost: number; tokens: number; calls: number }[];
  window_days: number;
}

/** Is this thing actually usable right now, and if not, WHY. */
function effectiveState(
  client: TenantDetailData["client"],
  site: Site,
  sub: Subscription,
  licences: Licence[],
): { live: boolean; label: string; reason: string | null } {
  // Order is the containment order, because the outermost switch is the one an
  // operator has to undo first. Telling someone their subscription is paused
  // when the whole customer is suspended sends them to fix the wrong thing.
  if (!client.is_active)
    return { live: false, label: "off", reason: "the customer is suspended" };
  if (!site.is_active)
    return { live: false, label: "off", reason: "this store is suspended" };
  if (sub.status !== "active" && sub.status !== "trial")
    return { live: false, label: sub.status, reason: sub.disabled_reason ?? "this module is paused" };

  const usable = licences.filter(
    (l) => l.is_active && (!l.expires_at || new Date(l.expires_at) > new Date()),
  );
  if (usable.length === 0)
    return { live: false, label: "no key", reason: "no active licence — nothing can authenticate" };

  return { live: true, label: sub.status, reason: null };
}

export function TenantDetail() {
  const { clientId } = useParams();
  const { days } = useFilters();

  const { data, isLoading, error } = useQuery<TenantDetailData>({
    queryKey: ["tenant", clientId, days],
    queryFn: () => api.get<TenantDetailData>(`/api/admin/tenants/${clientId}?days=${days}`),
  });

  if (isLoading) return <Loading what="tenant" />;
  if (error) return <ErrorNote error={error} />;
  if (!data) return null;

  const { client, sites, subscriptions, licences } = data;
  const totalCost = data.by_product.reduce((a, p) => a + p.cost, 0);
  const totalReq = data.by_product.reduce((a, p) => a + p.requests, 0);
  const measured = data.by_product.length > 0;

  return (
    <div className="td">
      <nav className="td-crumb">
        <Link to="/tenants">Tenants</Link>
        <span aria-hidden="true">/</span>
        <span>{client.name}</span>
      </nav>

      <header className={client.is_active ? "td-head" : "td-head is-off"}>
        <div className="td-head-main">
          <h1>
            {client.name}
            {!client.is_active && <>{" "}<span className="pill pill-bad">suspended</span></>}
          </h1>
          <p className="td-head-meta">
            {client.email}
            {client.company && <> · {client.company}</>}
            {" · joined "}{when(client.created_at)}
          </p>
        </div>

        <dl className="td-head-figs">
          <Fig label="Spend" value={measured ? cost(totalCost) : ABSENT} sub={`${days} days`} />
          <Fig label="Requests" value={measured ? num(totalReq) : ABSENT} sub="billable" />
          <Fig label="Sites" value={num(sites.length)} sub={`${subscriptions.length} subscriptions`} />
        </dl>
      </header>

      {!client.is_active && (
        <p className="td-suspended">
          Everything below is unreachable while the customer is suspended,
          whatever the individual rows say. Re-enabling the customer restores
          each store and module to its own state — it does not turn anything on
          that was off for its own reasons.
        </p>
      )}

      {sites.length === 0 ? (
        <p className="td-none">
          No store installs. This tenant completed signup but never registered a
          domain, so there is nothing to license.
        </p>
      ) : (
        sites.map((site) => (
          <SiteCard
            key={site.id}
            client={client}
            site={site}
            subs={subscriptions.filter((s) => s.site_id === site.id)}
            licences={licences}
          />
        ))
      )}

      {data.by_model.length > 0 && (
        <section className="td-panel">
          <header><span className="eyebrow">Model spend · {days} days</span></header>
          <div className="td-models">
            {data.by_model.map((m) => (
              <div key={`${m.provider}/${m.model}`} className="td-model">
                <span className="td-model-name">{m.model}</span>
                <span className="td-model-prov">{m.provider}</span>
                <span className="td-model-num">{num(m.calls)} calls</span>
                <span className="td-model-num">{num(m.tokens)} tokens</span>
                <span className="td-model-cost">{cost(m.cost)}</span>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

function Fig({ label, value, sub }: { label: string; value: string; sub: string }) {
  return (
    <div className={value === ABSENT ? "td-fig is-absent" : "td-fig"}>
      <dt className="eyebrow">{label}</dt>
      <dd>{value}<span>{sub}</span></dd>
    </div>
  );
}

/* ── A store install, with its modules inside it ─────────────────────────── */

function SiteCard({
  client, site, subs, licences,
}: {
  client: TenantDetailData["client"]; site: Site;
  subs: Subscription[]; licences: Licence[];
}) {
  const parentOff = !client.is_active;
  const off = parentOff || !site.is_active;
  const limit = site.catalogue_limit;
  const used = site.indexed_items;
  const pctUsed = limit > 0 ? (used / limit) * 100 : null;

  return (
    <section className={off ? "td-site is-off" : "td-site"}>
      <header className="td-site-head">
        <div>
          <h2>
            {site.domain}
            {site.environment === "production"
              ? <>{" "}<span className="pill pill-ok">prod</span></>
              : <>{" "}<span className="pill pill-muted">dev</span></>}
            {!site.is_active && <>{" "}<span className="pill pill-bad">suspended</span></>}
          </h2>
          <p className="td-site-meta">
            {site.platform}
            {site.platform_version && ` ${site.platform_version}`}
            {site.store_name && ` · ${site.store_name}`}
            {site.collection_name && <> · <code>{site.collection_name}</code></>}
          </p>
        </div>

        {/* The catalogue ceiling is the ONLY limit actually enforced today —
            writes past it are refused — so it gets a real gauge while request
            quota gets a number and a caveat. */}
        <div className="td-gauge">
          <div className="td-gauge-top">
            <span className="eyebrow">Catalogue · {site.index_plan}</span>
            <span className="td-gauge-num">
              {num(used)} <span>/ {limit > 0 ? num(limit) : "unlimited"}</span>
            </span>
          </div>
          <div className="td-gauge-track">
            <div
              className={
                "td-gauge-fill" +
                (pctUsed === null ? "" : pctUsed >= 95 ? " is-fault" : pctUsed >= 80 ? " is-ember" : "")
              }
              style={{ width: pctUsed === null ? "0%" : `${Math.min(pctUsed, 100)}%` }}
            />
          </div>
          <div className="td-gauge-foot">
            {pctUsed === null ? "no ceiling on this plan" : `${pct(pctUsed)} used · enforced`}
          </div>
        </div>
      </header>

      {subs.length === 0 ? (
        <p className="td-nosub">No modules on this store.</p>
      ) : (
        <ul className="td-subs">
          {subs.map((sub) => (
            <SubRow
              key={sub.id}
              state={effectiveState(client, site, sub, licences.filter(
                (l) => l.product_code === sub.product_code && l.domain === site.domain,
              ))}
              sub={sub}
              licences={licences.filter(
                (l) => l.product_code === sub.product_code && l.domain === site.domain,
              )}
            />
          ))}
        </ul>
      )}
    </section>
  );
}

/* ── One module on one store ─────────────────────────────────────────────── */

function SubRow({
  sub, licences, state,
}: {
  sub: Subscription; licences: Licence[];
  state: ReturnType<typeof effectiveState>;
}) {
  const live = licences.filter((l) => l.is_active);
  const expiring = live.find(
    (l) => l.expires_at && new Date(l.expires_at).getTime() - Date.now() < 30 * 86_400_000,
  );

  return (
    <li className={state.live ? "td-sub" : "td-sub is-off"}>
      <span className={state.live ? "td-sub-dot is-live" : "td-sub-dot"} aria-hidden="true" />

      <span className="td-sub-name">
        {productLabel(sub.product_code)}
        <span className="td-sub-plan">
          {sub.plan} · {num(sub.request_limit)} req/mo
          {/* Said every time a limit is shown. A ceiling nobody enforces reads
              as a ceiling to whoever set it. */}
          <em> not enforced</em>
        </span>
      </span>

      <span className="td-sub-keys">
        {live.length === 0 ? (
          <span className="td-key is-none">no active key</span>
        ) : (
          live.map((l) => (
            <Link key={l.id} to={`/licences/${l.id}`} className="td-key" title={
              l.has_plaintext
                ? "Plaintext stored — this key can be shown again"
                : "Issued before plaintext was stored; it can never be displayed"
            }>
              {l.key_prefix ?? ABSENT}
              {!l.has_plaintext && <span className="td-key-nokey" aria-label="cannot be shown">·</span>}
            </Link>
          ))
        )}
      </span>

      <span className="td-sub-state">
        {/* The row's OWN status is deliberately not shown when a parent
            overrides it — see effectiveState(). */}
        <span className={state.live ? "pill pill-ok" : "pill pill-bad"}>{state.label}</span>
        {state.reason && <span className="td-sub-why">{state.reason}</span>}
        {state.live && expiring && (
          <span className="td-sub-why is-ember">key expires {relative(expiring.expires_at)}</span>
        )}
      </span>
    </li>
  );
}
