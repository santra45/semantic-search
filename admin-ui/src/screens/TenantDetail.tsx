import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import { api } from "../api";
import { useActions } from "../lib/actions";
import type { Client, Licence, Site, Subscription, TenantDetailData } from "../lib/tenantTypes";
import type { PlanRung } from "../lib/plans";
import {
  openClientDisable, openClientEnable, openIndexPlanEdit, openIssue, openPause,
  openPlanEdit, openPromote, openResume, openSiteDisable, openSiteEnable,
  openTermEdit, type Act,
} from "./tenantActions";
import { usePlans } from "../lib/plans";
import { useCan } from "../lib/session";
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


/** Is this thing actually usable right now, and if not, WHY. */
function effectiveState(
  client: Client,
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
  const isOperator = useCan("operator");
  const isOwner = useCan("owner");
  const act = useActions([["tenant", clientId, days], ["tenants"], ["overview"]]);
  // Fetched once and shared by every plan dropdown on the page, so the two
  // ladders always offer what catalog.py actually defines.
  const plans = usePlans();

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
      {act.sheetNode}
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
          {isOperator && (
            <div className="td-fig td-actions">
              <dt className="eyebrow">Customer</dt>
              <dd>
                {client.is_active ? (
                  <button
                    className="td-btn is-danger"
                    onClick={() => openClientDisable(act, client)}
                  >
                    Suspend customer
                  </button>
                ) : (
                  <button className="td-btn" onClick={() => openClientEnable(act, client)}>
                    Restore customer
                  </button>
                )}
              </dd>
            </div>
          )}
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
            act={act}
            isOperator={isOperator}
            isOwner={isOwner}
            indexRungs={plans.data?.index_plans ?? []}
            moduleRungs={plans.data?.module_plans ?? []}
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
  client, site, subs, licences, act, isOperator, isOwner, indexRungs, moduleRungs,
}: {
  client: Client; site: Site;
  subs: Subscription[]; licences: Licence[];
  act: Act; isOperator: boolean; isOwner: boolean;
  indexRungs: PlanRung[]; moduleRungs: PlanRung[];
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

        {isOperator && (
          <div className="td-site-actions">
            <button className="td-btn" disabled={!indexRungs.length}
                    onClick={() => openIndexPlanEdit(act, site, indexRungs)}>
              Catalogue plan
            </button>
            {site.is_active ? (
              <button className="td-btn is-danger"
                      disabled={parentOff}
                      title={parentOff ? "The customer is already suspended" : undefined}
                      onClick={() => openSiteDisable(act, site)}>
                Suspend store
              </button>
            ) : (
              <button className="td-btn" onClick={() => openSiteEnable(act, site)}>
                Restore store
              </button>
            )}
            {isOwner && site.environment !== "production" && (
              <button className="td-btn" onClick={() => openPromote(act, site)}>
                Promote to production
              </button>
            )}
          </div>
        )}
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
              act={act}
              moduleRungs={moduleRungs}
              // A pause button under a suspended customer would suggest the
              // module is what is holding things up. Disabled, with the reason
              // in the tooltip, rather than hidden — hiding it would make the
              // control look like it does not exist at this level at all.
              parentOff={off}
              isOperator={isOperator}
              isOwner={isOwner}
            />
          ))}
        </ul>
      )}
    </section>
  );
}

/* ── One module on one store ─────────────────────────────────────────────── */

function SubRow({
  sub, licences, state, act, parentOff, isOperator, isOwner, moduleRungs,
}: {
  sub: Subscription; licences: Licence[];
  state: ReturnType<typeof effectiveState>;
  act: Act; parentOff: boolean; isOperator: boolean; isOwner: boolean;
  moduleRungs: PlanRung[];
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

      {(isOperator || isOwner) && (
        <span className="td-sub-actions">
          {isOperator && (
            <>
              <button className="td-btn" disabled={!moduleRungs.length}
                      onClick={() => openPlanEdit(act, sub, moduleRungs)}>
                Plan
              </button>
              <button className="td-btn" onClick={() => openTermEdit(act, sub)}>
                Extend
              </button>
            </>
          )}
          {isOperator && (sub.status === "active" || sub.status === "trial" ? (
            <button className="td-btn is-danger" disabled={parentOff}
                    title={parentOff ? "A parent scope is already suspended" : undefined}
                    onClick={() => openPause(act, sub)}>
              Pause
            </button>
          ) : (
            <button className="td-btn" disabled={parentOff}
                    onClick={() => openResume(act, sub)}>
              Resume
            </button>
          ))}
          {isOwner && (
            <button className="td-btn" onClick={() => openIssue(act, sub, live.length > 0)}>
              {live.length > 0 ? "Rotate key" : "Issue key"}
            </button>
          )}
        </span>
      )}
    </li>
  );
}
