import { NavLink, Outlet } from "react-router-dom";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "../api";
import { useMe } from "../lib/session";
import { useFilters, type EnvFilter } from "../lib/filters";
import { can } from "../api";

const NAV: { to: string; label: string; min?: "viewer" | "operator" | "owner" }[] = [
  { to: "/", label: "Overview" },
  { to: "/tenants", label: "Tenants" },
  { to: "/licences", label: "Licences" },
  { to: "/products", label: "Products" },
  { to: "/plans", label: "Plans" },
  { to: "/usage", label: "Usage" },
  { to: "/audit", label: "Audit" },
  { to: "/system", label: "System" },
];

export function Layout() {
  const me = useMe();
  const qc = useQueryClient();
  const { environment, setEnvironment, days, setDays } = useFilters();

  const logout = useMutation({
    mutationFn: () => api.post("/api/admin/auth/logout"),
    // clear(), not invalidate. After logout every cached answer belongs to a
    // session that no longer exists; invalidating would refetch them all and
    // flash a tenant list on the way to the login screen.
    onSuccess: () => qc.clear(),
  });

  return (
    <div className="app">
      {me.is_break_glass && (
        // Loud on purpose. Break-glass carries owner rights and its audit rows
        // are attributed to "break-glass" rather than a person, so anyone using
        // it is unattributable — and it is meant to be removed now that real
        // accounts exist.
        <div className="banner">
          Signed in with the break-glass operator key — actions are logged
          without a named actor. Remove <code>AICHATBOT_OPERATOR_KEY</code> from
          production.
        </div>
      )}

      <header className="topbar">
        <strong className="brand">Czargroup Admin</strong>

        <nav className="nav">
          {NAV.filter((n) => !n.min || can(me.role, n.min)).map((n) => (
            <NavLink
              key={n.to}
              to={n.to}
              end={n.to === "/"}
              className={({ isActive }) => (isActive ? "navlink active" : "navlink")}
            >
              {n.label}
            </NavLink>
          ))}
        </nav>

        <div className="spacer" />

        {/* Global, and applied to every list and KPI. See lib/filters.tsx for
            why this is a control rather than a column. */}
        <label className="inline-field">
          <span>Env</span>
          <select
            value={environment}
            onChange={(e) => setEnvironment(e.target.value as EnvFilter)}
          >
            <option value="all">All</option>
            <option value="development">Development</option>
            <option value="production">Production</option>
          </select>
        </label>

        <label className="inline-field">
          <span>Window</span>
          <select value={days} onChange={(e) => setDays(Number(e.target.value))}>
            <option value={7}>7 days</option>
            <option value={30}>30 days</option>
            <option value={90}>90 days</option>
            <option value={365}>1 year</option>
          </select>
        </label>

        <span className="pill pill-muted" title={`Role: ${me.role}`}>{me.role}</span>
        <span className="muted who">{me.email}</span>
        <button className="linkbtn" onClick={() => logout.mutate()} disabled={logout.isPending}>
          Sign out
        </button>
      </header>

      <main className="content">
        <Outlet />
      </main>
    </div>
  );
}
