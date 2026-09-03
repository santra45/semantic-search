import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, Unauthenticated, type Me } from "./api";

/**
 * Slice 1: prove the whole pipeline end to end — Vite build -> FastAPI static
 * mount -> authenticated round-trip -> session cookie -> logout.
 *
 * The routed screens land on top of this shell; what is here is the part that
 * has to be right before any of them are worth writing.
 */
export function App() {
  // /auth/me is the single source of truth for "am I signed in". There is no
  // client-side session state to drift from it, because the cookie is httpOnly
  // and this is the only way to ask.
  const me = useQuery<Me>({
    queryKey: ["me"],
    queryFn: () => api.get<Me>("/api/admin/auth/me"),
    retry: false,
  });

  if (me.isLoading) {
    return (
      <div className="center">
        <p className="muted">Loading…</p>
      </div>
    );
  }

  if (me.error instanceof Unauthenticated || !me.data) {
    return <Login />;
  }

  return <Shell me={me.data} />;
}

function Login() {
  const qc = useQueryClient();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const login = useMutation({
    mutationFn: () => api.post("/api/admin/auth/login", { email, password }),
    // Invalidate rather than write the result in: the response body and
    // /auth/me could drift, and there should be exactly one shape of "who am I"
    // in this app.
    onSuccess: () => qc.invalidateQueries({ queryKey: ["me"] }),
  });

  return (
    <div className="center">
      <form
        className="card"
        onSubmit={(e) => {
          e.preventDefault();
          login.mutate();
        }}
      >
        <h1>Czargroup Admin</h1>
        <p className="muted">Operator console</p>

        <label htmlFor="email">Email</label>
        <input
          id="email"
          type="email"
          autoComplete="username"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
        />

        <label htmlFor="password">Password</label>
        <input
          id="password"
          type="password"
          autoComplete="current-password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />

        <button type="submit" disabled={login.isPending}>
          {login.isPending ? "Signing in…" : "Sign in"}
        </button>

        {login.error && (
          // Rendered verbatim from the server. It says the same thing for an
          // unknown address and a wrong password, on purpose — improving the
          // message here would reintroduce the account-enumeration oracle the
          // backend closes with a matched response time.
          <div className="err">{(login.error as Error).message}</div>
        )}
      </form>
    </div>
  );
}

function Shell({ me }: { me: Me }) {
  const qc = useQueryClient();
  const logout = useMutation({
    mutationFn: () => api.post("/api/admin/auth/logout"),
    // clear(), not invalidate: after logout every cached answer belongs to a
    // session that no longer exists, and a stale tenant list flashing before
    // the login screen is both wrong and a small disclosure.
    onSuccess: () => qc.clear(),
  });

  return (
    <>
      {me.is_break_glass && (
        // Loud on purpose. Break-glass has owner rights and its audit rows are
        // attributed to "break-glass" rather than a person, so anyone using it
        // should be aware they are unattributable — and reminded it is meant to
        // be removed now that real accounts exist.
        <div className="banner">
          Signed in with the break-glass operator key. Actions are logged
          without a named actor. Remove <code>AICHATBOT_OPERATOR_KEY</code> from
          production now that real accounts exist.
        </div>
      )}

      <div className="topbar">
        <strong>Czargroup Admin</strong>
        <span className="pill">{me.role}</span>
        <div className="spacer" />
        <span className="muted">{me.email}</span>
        <button
          className="linkbtn"
          onClick={() => logout.mutate()}
          disabled={logout.isPending}
        >
          Sign out
        </button>
      </div>

      <main>
        <p className="muted">
          Signed in as {me.name}. Screens land here next — overview, tenants,
          licences, usage, audit.
        </p>
      </main>
    </>
  );
}
