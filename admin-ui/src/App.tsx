import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { Login } from "./screens/Login";
import { Layout } from "./components/Layout";
import { MeProvider, useSession, Unauthenticated } from "./lib/session";
import { FiltersProvider } from "./lib/filters";
import { ToastProvider } from "./lib/toast";
import { Loading } from "./components/Bits";
import { Placeholder } from "./screens/Placeholder";
import { Overview } from "./screens/Overview";
import { Tenants } from "./screens/Tenants";
import { TenantDetail } from "./screens/TenantDetail";
import { Licences } from "./screens/Licences";

/**
 * The shell. Auth gate first, then providers, then routes.
 *
 * The gate is deliberately outside the router: an unauthenticated user has no
 * business having routes resolved for them, and rendering the login screen from
 * inside a route means every screen needs its own "am I signed in" branch.
 */
export function App() {
  const session = useSession();

  if (session.isLoading) {
    return <div className="center"><Loading what="console" /></div>;
  }

  // 401 is the expected answer for a signed-out visitor, not an error state.
  if (session.error instanceof Unauthenticated || !session.data) {
    return <Login />;
  }

  return (
    <MeProvider me={session.data}>
      <FiltersProvider>
        <ToastProvider>
          {/* basename, because the console is served from /admin. Without it
              every route resolves one segment too high and the router renders
              nothing while the URL bar looks correct. */}
          <BrowserRouter basename="/admin">
            <Routes>
              <Route element={<Layout />}>
                <Route index element={<Overview />} />
                <Route path="tenants" element={<Tenants />} />
                <Route path="tenants/:clientId" element={<TenantDetail />} />
                <Route path="sites/:siteId" element={<Placeholder title="Site detail" />} />
                <Route path="licences" element={<Licences />} />
                <Route path="licences/:licenceId" element={<Placeholder title="Licence detail" />} />
                <Route path="products" element={<Placeholder title="Products" />} />
                <Route path="plans" element={<Placeholder title="Plans" />} />
                <Route path="usage" element={<Placeholder title="Usage explorer" />} />
                <Route path="audit" element={<Placeholder title="Audit log" />} />
                <Route path="system" element={<Placeholder title="System health" />} />
                {/* The server hands index.html to ANY /admin/* path — it cannot
                    tell a typo from a route it does not know — so the SPA owns
                    not-found. Redirecting rather than 404-ing, since the only
                    way to get here is a stale link. */}
                <Route path="*" element={<Navigate to="/" replace />} />
              </Route>
            </Routes>
          </BrowserRouter>
        </ToastProvider>
      </FiltersProvider>
    </MeProvider>
  );
}
