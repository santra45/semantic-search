import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "../api";

export function Login() {
  const qc = useQueryClient();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const login = useMutation({
    mutationFn: () => api.post("/api/admin/auth/login", { email, password }),
    // Invalidate rather than writing the response in: /auth/me is the single
    // source of truth for who is signed in, and two shapes of that answer is
    // how they drift.
    onSuccess: () => qc.invalidateQueries({ queryKey: ["me"] }),
  });

  const err = login.error as (Error & { status?: number }) | null;

  return (
    <div className="center">
      <form className="card" onSubmit={(e) => { e.preventDefault(); login.mutate(); }}>
        <h1>Czargroup Admin</h1>
        <p className="muted">Operator console</p>

        <label htmlFor="email">Email</label>
        <input id="email" type="email" autoComplete="username" value={email}
               onChange={(e) => setEmail(e.target.value)} required autoFocus />

        <label htmlFor="password">Password</label>
        <input id="password" type="password" autoComplete="current-password" value={password}
               onChange={(e) => setPassword(e.target.value)} required />

        <button type="submit" disabled={login.isPending}>
          {login.isPending ? "Signing in…" : "Sign in"}
        </button>

        {err && (
          // Rendered verbatim. The server says the same thing for an unknown
          // address and a wrong password on purpose, and pairs it with a matched
          // response time — "improving" the message here would reopen the
          // account-enumeration oracle both halves exist to close.
          <div className="err">
            {err.message}
            {err.status === 429 && (
              <div className="toast-detail">
                Five failed attempts locks sign-in for 15 minutes.
              </div>
            )}
          </div>
        )}
      </form>
    </div>
  );
}
