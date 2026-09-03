import { useEffect, useState } from "react";
import { useQuery, keepPreviousData } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api } from "../api";
import { useFilters } from "../lib/filters";
import { cost, num, when } from "../lib/format";
import { ErrorNote, Loading } from "../components/Bits";

/**
 * The tenant list.
 *
 * Extends the overview's grammar rather than inventing a second one. There, a
 * horizontal strip showed telemetry coverage ACROSS sites; here a leading rail
 * on each row shows the same thing DOWN clients — lit where the tenant produces
 * ledger rows, hatched where it does not. One idea, two axes, so an operator
 * learns the vocabulary once.
 *
 * The five-slot product ledger is the other half. Every row reserves the same
 * five positions in the same order, filled or empty, so "who is on Product
 * Q&A?" is a vertical scan instead of reading eight comma-separated lists.
 */

const PRODUCT_SLOTS = [
  { code: "magento_search", short: "msrch", name: "Magento · Search" },
  { code: "magento_chatbot", short: "mchat", name: "Magento · Chatbot" },
  { code: "magento_product_qa", short: "mpqa", name: "Magento · Product Q&A" },
  { code: "woo_search", short: "wsrch", name: "Woo · Search" },
  { code: "woo_product_qa", short: "wpqa", name: "Woo · Product Q&A" },
] as const;

interface Tenant {
  client_id: string;
  name: string;
  email: string;
  is_active: boolean;
  created_at: string | null;
  site_count: number;
  subscription_count: number;
  environments: string[];
  products: string[];
  requests: number | null;
  cost: number | null;
}

interface TenantsData {
  total: number;
  limit: number;
  offset: number;
  window_days: number;
  tenants: Tenant[];
}

const SORTS = [
  { key: "cost", label: "Spend" },
  { key: "requests", label: "Requests" },
  { key: "name", label: "Name" },
  { key: "sites", label: "Sites" },
  { key: "created", label: "Newest" },
] as const;

export function Tenants() {
  const { days, environment, envParam } = useFilters();
  const [search, setSearch] = useState("");
  const [debounced, setDebounced] = useState("");
  const [product, setProduct] = useState("");
  const [status, setStatus] = useState("");
  const [sort, setSort] = useState<string>("cost");
  const [offset, setOffset] = useState(0);
  const limit = 25;

  // Debounced, so typing a domain fires one request rather than one per
  // keystroke against a database that is also serving storefronts.
  useEffect(() => {
    const t = setTimeout(() => {
      setDebounced(search);
      setOffset(0);
    }, 250);
    return () => clearTimeout(t);
  }, [search]);

  const qs =
    `?days=${days}&limit=${limit}&offset=${offset}&sort=${sort}` +
    envParam() +
    (debounced ? `&search=${encodeURIComponent(debounced)}` : "") +
    (product ? `&product=${product}` : "") +
    (status ? `&status=${status}` : "");

  const { data, isLoading, error, isFetching } = useQuery<TenantsData>({
    queryKey: ["tenants", days, environment, debounced, product, status, sort, offset],
    queryFn: () => api.get<TenantsData>(`/api/admin/tenants${qs}`),
    // Keeps the previous page on screen while the next loads, so filtering does
    // not blank the table and bounce the scroll position on every keystroke.
    placeholderData: keepPreviousData,
  });

  return (
    <div className="tn">
      <div className="tn-controls">
        <input
          className="tn-search"
          type="search"
          placeholder="Search name, email or domain"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          aria-label="Search tenants"
        />

        <select value={product} onChange={(e) => { setProduct(e.target.value); setOffset(0); }}
                aria-label="Filter by product">
          <option value="">All products</option>
          {PRODUCT_SLOTS.map((p) => (
            <option key={p.code} value={p.code}>{p.name}</option>
          ))}
        </select>

        <select value={status} onChange={(e) => { setStatus(e.target.value); setOffset(0); }}
                aria-label="Filter by status">
          <option value="">Active and disabled</option>
          <option value="active">Active only</option>
          <option value="inactive">Disabled only</option>
        </select>

        <div className="tn-sorts" role="group" aria-label="Sort by">
          {SORTS.map((s) => (
            <button
              key={s.key}
              className={sort === s.key ? "tn-sort is-on" : "tn-sort"}
              onClick={() => { setSort(s.key); setOffset(0); }}
            >
              {s.label}
            </button>
          ))}
        </div>
      </div>

      {error ? (
        <ErrorNote error={error} />
      ) : isLoading ? (
        <Loading what="tenants" />
      ) : !data ? null : data.tenants.length === 0 ? (
        <p className="tn-none">
          No tenant matches these filters. Clear the search or widen the
          environment to see the rest of the estate.
        </p>
      ) : (
        <>
          <div className={isFetching ? "tn-table is-stale" : "tn-table"}>
            <div className="tn-head" aria-hidden="true">
              <span />
              <span className="eyebrow">Tenant</span>
              <span className="eyebrow">Products</span>
              <span className="eyebrow tn-r">Sites</span>
              <span className="eyebrow tn-r">Requests</span>
              <span className="eyebrow tn-r">Spend</span>
            </div>

            {data.tenants.map((t) => <Row key={t.client_id} t={t} />)}
          </div>

          <div className="tn-foot">
            <span>
              {offset + 1}–{Math.min(offset + limit, data.total)} of {data.total}
            </span>
            <div className="tn-pager">
              <button className="linkbtn" disabled={offset === 0}
                      onClick={() => setOffset(Math.max(0, offset - limit))}>
                Previous
              </button>
              <button className="linkbtn" disabled={offset + limit >= data.total}
                      onClick={() => setOffset(offset + limit)}>
                Next
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function Row({ t }: { t: Tenant }) {
  // null means the tenant produces no ledger rows at all — a v1 key resolves no
  // v2 context, so nothing is recorded. Distinct from 0, which is a real
  // measurement of no traffic, and the rail is what carries that distinction
  // down the column where a per-cell em dash alone would be missed.
  const measured = t.requests !== null;

  return (
    <Link
      to={`/tenants/${t.client_id}`}
      className={
        "tn-row" + (measured ? " is-measured" : " is-dark") + (t.is_active ? "" : " is-off")
      }
    >
      <span
        className="tn-rail"
        aria-hidden="true"
        title={measured ? "Producing telemetry" : "No telemetry — v1 key or no traffic"}
      />

      <span className="tn-who">
        <span className="tn-name">
          {t.name}
          {/* Disabled tenants stay VISIBLE and greyed rather than being filtered
              out — a tenant that vanishes from the list is one nobody
              remembers to turn back on. */}
          {!t.is_active && <span className="pill pill-bad">disabled</span>}
          {t.environments.length === 1 && t.environments[0] === "development" && (
            <span className="pill pill-muted">dev</span>
          )}
        </span>
        <span className="tn-email">{t.email}</span>
      </span>

      <span className="tn-slots">
        {PRODUCT_SLOTS.map((p) => {
          const has = t.products.includes(p.code);
          return (
            <span
              key={p.code}
              className={has ? "tn-slot is-on" : "tn-slot"}
              title={has ? p.name : `${p.name} — not subscribed`}
            >
              {p.short}
            </span>
          );
        })}
      </span>

      {/* Explicit per-cell classes rather than nth-of-type: that pseudo-class
          counts ELEMENTS, not classes, so a rule meant to hide "the first .tn-r"
          silently matches nothing when the first span is the rail. data-label
          is what the stacked mobile layout prints in front of each number. */}
      <span className="tn-r tn-num tn-sites" data-label="Sites">{num(t.site_count)}</span>
      <span className={"tn-r tn-num tn-req" + (measured ? "" : " is-absent")}
            data-label="Requests">
        {num(t.requests)}
      </span>
      <span className={"tn-r tn-num tn-spend" + (measured ? "" : " is-absent")}
            data-label="Spend">
        {cost(t.cost)}
      </span>

      <span className="tn-created">joined {when(t.created_at)}</span>
    </Link>
  );
}
