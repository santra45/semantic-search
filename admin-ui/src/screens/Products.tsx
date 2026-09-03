import { useQuery } from "@tanstack/react-query";
import { api } from "../api";
import { useActions } from "../lib/actions";
import { useCan } from "../lib/session";
import { useFilters } from "../lib/filters";
import { cost, num } from "../lib/format";
import { ErrorNote, Loading } from "../components/Bits";

/**
 * The five sellable modules.
 *
 * ────────────────────────────────────────────────────────────────────────────
 * `is_sellable` MEANS "CANNOT BE BOUGHT", NOT "CANNOT BE USED", AND THIS SCREEN
 * IS WHERE THAT GETS MISREAD.
 *
 * schema_v2 states the contract: withdrawn products go false, EXISTING
 * SUBSCRIPTIONS KEEP RESOLVING, onboarding stops offering it. The request
 * chokepoint deliberately never reads the column — wiring it there would break
 * every live subscription on the product the moment it was withdrawn.
 *
 * Somebody will eventually open this page during an incident expecting a kill
 * switch. So each card carries TWO separate facts side by side — whether the
 * product is on sale, and how many subscriptions are still running — and a
 * withdrawn product says out loud that its existing installs are untouched.
 * One number cannot be mistaken for the other when both are on screen.
 * ────────────────────────────────────────────────────────────────────────────
 */

interface Product {
  code: string;
  name: string;
  platform: string;
  key_segment: string;
  is_sellable: boolean;
  subscriptions: number;
  active_subscriptions: number;
  sites: number;
  clients: number;
  requests: number | null;
  cost: number | null;
  tokens: number | null;
}

interface ProductsData {
  window_days: number;
  environment: string | null;
  products: Product[];
}

export function Products() {
  const { days, environment, envParam } = useFilters();
  const isOwner = useCan("owner");
  const act = useActions([["products"], ["plans"], ["overview"]]);

  const { data, isLoading, error } = useQuery<ProductsData>({
    queryKey: ["products", days, environment],
    queryFn: () => api.get<ProductsData>(`/api/admin/products?days=${days}${envParam()}`),
  });

  if (isLoading) return <Loading what="products" />;
  if (error) return <ErrorNote error={error} />;
  if (!data) return null;

  const maxCost = Math.max(...data.products.map((p) => p.cost ?? 0), 0.000001);
  const withdrawn = data.products.filter((p) => !p.is_sellable).length;

  return (
    <div className="pr">
      {act.sheetNode}

      {withdrawn > 0 && (
        <p className="pr-withdrawn-note">
          {withdrawn} {withdrawn === 1 ? "product is" : "products are"} withdrawn
          from sale. Their existing subscriptions still resolve and still work —
          withdrawing stops onboarding offering the module, it does not stop
          anyone using it.
        </p>
      )}

      <div className="pr-grid">
        {data.products.map((p) => (
          <Card key={p.code} p={p} maxCost={maxCost} isOwner={isOwner} act={act} days={days} />
        ))}
      </div>
    </div>
  );
}

function Card({
  p, maxCost, isOwner, act, days,
}: {
  p: Product; maxCost: number; isOwner: boolean;
  act: ReturnType<typeof useActions>; days: number;
}) {
  return (
    <article className={p.is_sellable ? "pr-card" : "pr-card is-withdrawn"}>
      <header className="pr-head">
        <div>
          <h2>{p.name}</h2>
          <p className="pr-meta">
            <code>{p.code}</code>
            <span className="pr-seg" title="The label inside a licence key — not a credential">
              {p.key_segment}
            </span>
          </p>
        </div>
        <span className="pill pill-muted">{p.platform}</span>
      </header>

      {/* The two facts, deliberately adjacent. "On sale" and "running" are
          independent, and this pairing is the whole point of the card. */}
      <div className="pr-states">
        <div className="pr-state">
          <span className="eyebrow">On sale</span>
          <strong className={p.is_sellable ? "is-yes" : "is-no"}>
            {p.is_sellable ? "yes" : "withdrawn"}
          </strong>
          <span className="pr-state-hint">offered by onboarding</span>
        </div>
        <div className="pr-state">
          <span className="eyebrow">Running</span>
          <strong className={p.active_subscriptions > 0 ? "is-yes" : ""}>
            {num(p.active_subscriptions)}
          </strong>
          <span className="pr-state-hint">
            {p.is_sellable
              ? "live subscriptions"
              : "live subscriptions · unaffected by withdrawal"}
          </span>
        </div>
      </div>

      <dl className="pr-stats">
        <div><dt>Customers</dt><dd>{num(p.clients)}</dd></div>
        <div><dt>Stores</dt><dd>{num(p.sites)}</dd></div>
        <div><dt>Subscriptions</dt><dd>{num(p.subscriptions)}</dd></div>
      </dl>

      <div className="pr-spend">
        <div className="pr-spend-top">
          <span className="eyebrow">Spend · {days} days</span>
          <span className="pr-cost">{cost(p.cost)}</span>
        </div>
        <div className="pr-track">
          <div className="pr-fill" style={{ width: `${((p.cost ?? 0) / maxCost) * 100}%` }} />
        </div>
        <div className="pr-spend-foot">
          {num(p.requests)} requests · {num(p.tokens)} tokens
        </div>
      </div>

      {isOwner && (
        <div className="pr-actions">
          {p.is_sellable ? (
            <button className="td-btn is-danger" onClick={() => openWithdraw(act, p)}>
              Withdraw from sale
            </button>
          ) : (
            <button className="td-btn" onClick={() => openRestore(act, p)}>
              Put back on sale
            </button>
          )}
        </div>
      )}
    </article>
  );
}

function openWithdraw(act: ReturnType<typeof useActions>, p: Product) {
  act.open(
    {
      title: `Withdraw ${p.name} from sale`,
      tone: "danger",
      verb: "Withdraw",
      reasonRequired: true,
      // The affected count has to be echoed back, so an operator cannot fire
      // this without having read what it touches.
      confirmText: String(p.subscriptions),
      confirmLabel: "Type the number of existing subscriptions",
      body: (
        <>
          <strong>This is not a kill switch.</strong> Onboarding stops offering{" "}
          {p.name}, and that is all. Its {num(p.subscriptions)} existing{" "}
          {p.subscriptions === 1 ? "subscription keeps" : "subscriptions keep"}{" "}
          resolving and working exactly as now — the request chokepoint does not
          read this flag, deliberately, because reading it would break every
          live install the moment you clicked.
          <br />
          <br />
          To actually stop traffic, pause the subscriptions individually.
        </>
      ),
    },
    async (reason) => {
      await api.post(`/api/admin/products/${p.code}/withdraw`, {
        reason,
        confirm_affected: p.subscriptions,
      });
      return {
        message: `${p.name} withdrawn from sale`,
        detail: `${num(p.subscriptions)} existing subscriptions are unaffected.`,
        undo: {
          label: "Put back",
          run: () => api.post(`/api/admin/products/${p.code}/restore`, {}),
        },
      };
    },
  );
}

function openRestore(act: ReturnType<typeof useActions>, p: Product) {
  act.open(
    {
      title: `Put ${p.name} back on sale`,
      verb: "Put back on sale",
      body: <>Onboarding starts offering {p.name} again. Nothing else changes.</>,
    },
    async (reason) => {
      await api.post(`/api/admin/products/${p.code}/restore`, { reason });
      return { message: `${p.name} is on sale again` };
    },
  );
}
