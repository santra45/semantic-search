import { createContext, useContext, type ReactNode } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api, can, Unauthenticated, type Me, type Role } from "../api";

/**
 * Who is signed in.
 *
 * There is no client-side session state and there must not be. The cookie is
 * httpOnly, so `/api/admin/auth/me` is the ONLY way to ask, and making it the
 * single source of truth means the UI cannot drift into believing someone is
 * signed in after the server has stopped agreeing.
 */

const MeContext = createContext<Me | null>(null);

export function MeProvider({ me, children }: { me: Me; children: ReactNode }) {
  return <MeContext.Provider value={me}>{children}</MeContext.Provider>;
}

export function useMe(): Me {
  const me = useContext(MeContext);
  // Loud rather than optional-chained everywhere: every screen renders inside
  // the provider by construction, so a null here is a routing bug and should
  // say so once instead of producing blank fields in twelve components.
  if (!me) throw new Error("useMe() outside <MeProvider>. Screens render inside the shell.");
  return me;
}

export function useSession() {
  const qc = useQueryClient();
  return useQuery<Me>({
    queryKey: ["me"],
    queryFn: () => api.get<Me>("/api/admin/auth/me"),
    retry: false,
    // Long stale time: the role does not change mid-session, and re-asking on
    // every screen mount adds a round-trip to every navigation.
    staleTime: 5 * 60_000,
    // Kept so callers can force a refresh after a password change.
    meta: { qc },
  });
}

/**
 * Hide what the operator cannot use.
 *
 * COSMETIC ONLY, and worth being explicit about: the server enforces every role
 * on every endpoint (`require_operator` / `require_owner`), and it would still
 * refuse if this component rendered the button anyway. Hiding is about not
 * offering someone an action that will fail — not about preventing it.
 */
export function IfRole({
  min,
  children,
  fallback = null,
}: {
  min: Role;
  children: ReactNode;
  fallback?: ReactNode;
}) {
  const me = useMe();
  return <>{can(me.role, min) ? children : fallback}</>;
}

export function useCan(min: Role): boolean {
  return can(useMe().role, min);
}

export { Unauthenticated };
