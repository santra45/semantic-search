import { num } from "../lib/format";
import { usePlans, type PlanRung } from "../lib/plans";
import { ErrorNote, Loading } from "../components/Bits";

/**
 * The two plan ladders.
 *
 * ────────────────────────────────────────────────────────────────────────────
 * TWO LADDERS, NOT ONE PRICING TABLE.
 *
 * They are independent and neither can be derived from the other:
 *
 *   INDEX_PLANS   catalogue SIZE, bought ONCE PER STORE, because every module
 *                 on a store shares one Qdrant collection
 *   MODULE_PLANS  request QUOTA, bought PER MODULE
 *
 * A store on three modules holds ONE index plan and THREE module plans. Merging
 * them into a single pricing table — the obvious layout — would imply a
 * customer picks one rung, and a console that implied that would eventually let
 * somebody edit the wrong one. So they are rendered as two ladders, each
 * stating its own scope, side by side but never interleaved.
 *
 * Deriving a store's ceiling from the highest of its module plans breaks on
 * cancellation: drop the module carrying the biggest plan and the derived
 * ceiling falls below what is already indexed, with no clean resolution. That
 * is why the split exists at all.
 * ────────────────────────────────────────────────────────────────────────────
 */

export function Plans() {
  const { data, isLoading, error } = usePlans();

  if (isLoading) return <Loading what="plans" />;
  if (error) return <ErrorNote error={error} />;
  if (!data) return null;

  return (
    <div className="pl">
      <p className="pl-intro">
        Two independent ladders. A store buys <strong>one</strong> catalogue plan
        however many modules it runs, and <strong>one request plan per module</strong>.
        Neither can be worked out from the other, which is why they are edited
        from different screens — catalogue size on the store, request quota on
        the subscription.
      </p>

      <div className="pl-split">
        <Ladder
          title="Catalogue size"
          scope="bought once per store"
          rungs={data.index_plans}
          countKey="sites"
          countWord="stores"
          limitOf={(r) => (typeof r.catalogue_limit === "number"
            ? `${r.catalogue_limit.toLocaleString()} items` : null)}
          enforced
        />
        <Ladder
          title="Request quota"
          scope="bought per module"
          rungs={data.module_plans}
          countKey="subscriptions"
          countWord="subscriptions"
          limitOf={(r) => (typeof r.request_limit === "number"
            ? `${r.request_limit.toLocaleString()} req/mo` : null)}
          enforced={false}
        />
      </div>

      {!data.revenue_computable && (
        <p className="pl-revenue">
          Prices are display strings in <code>catalog.py</code>, not amounts, so
          nothing here can total revenue. A revenue view needs a real{" "}
          <code>price_cents</code> field first — deliberately not faked from
          these.
        </p>
      )}
    </div>
  );
}

function Ladder({
  title, scope, rungs, countKey, countWord, limitOf, enforced,
}: {
  title: string;
  scope: string;
  rungs: PlanRung[];
  countKey: "sites" | "subscriptions";
  countWord: string;
  limitOf: (r: PlanRung) => string | null;
  enforced: boolean;
}) {
  const total = rungs.reduce((a, r) => a + (Number(r[countKey]) || 0), 0);

  return (
    <section className="pl-ladder">
      <header className="pl-ladder-head">
        <div>
          <h2>{title}</h2>
          <p className="pl-scope">{scope}</p>
        </div>
        {/* Said on the ladder, once, rather than on every rung. The catalogue
            ceiling refuses writes; the request limit does not, and an operator
            setting one should know which they are setting. */}
        <span className={enforced ? "pill pill-ok" : "pill pill-warn"}>
          {enforced ? "enforced" : "not enforced"}
        </span>
      </header>

      <ol className="pl-rungs">
        {rungs.map((r) => {
          const count = Number(r[countKey]) || 0;
          const share = total > 0 ? (count / total) * 100 : 0;
          const limit = limitOf(r);
          return (
            <li key={r.code} className={count > 0 ? "pl-rung is-used" : "pl-rung"}>
              <span className="pl-bar" style={{ width: `${share}%` }} aria-hidden="true" />
              <div className="pl-rung-body">
                <div className="pl-rung-top">
                  <span className="pl-name">{r.name ?? r.code}</span>
                  {/* A rung the customer cannot pick — `trial` is assigned, not
                      chosen — is worth marking so nobody wonders why it is
                      missing from the pricing page. */}
                  {r.selectable === false && (
                    <span className="pill pill-muted">not selectable</span>
                  )}
                  <span className="pl-price">
                    {r.price}
                    {r.period && <em> {r.period}</em>}
                  </span>
                </div>
                <div className="pl-rung-foot">
                  {limit && <span className="pl-limit">{limit}</span>}
                  <span className="pl-count">
                    {count > 0 ? `${num(count)} ${countWord}` : `no ${countWord}`}
                  </span>
                </div>
                {Array.isArray(r.features) && r.features.length > 0 && (
                  <ul className="pl-features">
                    {r.features.map((f) => <li key={f}>{f}</li>)}
                  </ul>
                )}
              </div>
            </li>
          );
        })}
      </ol>
    </section>
  );
}
