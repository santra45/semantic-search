/**
 * The one place that talks to /api/admin.
 *
 * WHY THERE IS NO TOKEN HANDLING HERE. The session cookie is httpOnly, so this
 * code cannot read it and does not need to: the browser attaches it to every
 * same-origin request on its own. Anything in this file that reached for
 * localStorage or an Authorization header would be reintroducing a credential
 * the backend deliberately keeps out of JavaScript's reach — which is what
 * makes an XSS on this origin unable to walk away with an operator session.
 */

export class ApiError extends Error {
  status: number;
  body: unknown;
  constructor(status: number, message: string, body: unknown) {
    super(message);
    this.status = status;
    this.body = body;
  }
}

/** Thrown on 401 so the app can show the login screen instead of an error. */
export class Unauthenticated extends ApiError {}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    // Belt and braces. Same-origin requests send cookies by default, but an
    // explicit value means a future move to a different host fails loudly
    // rather than silently dropping the session on every request.
    credentials: "same-origin",
  });

  // 204 and empty bodies are real answers, not parse failures.
  const text = await res.text();
  let body: unknown = null;
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      body = text;
    }
  }

  if (!res.ok) {
    const detail =
      (body && typeof body === "object" && "detail" in body
        ? String((body as { detail: unknown }).detail)
        : null) ?? `Request failed (${res.status})`;

    // 401 means "not signed in", 403 means "signed in, not allowed". Conflating
    // them sends someone who lacks a role back to a login screen they are
    // already past, which reads as a broken console rather than a permission.
    if (res.status === 401) throw new Unauthenticated(401, detail, body);
    throw new ApiError(res.status, detail, body);
  }

  return body as T;
}

export const api = {
  get: <T>(path: string) => request<T>(path),
  post: <T>(path: string, data?: unknown) =>
    request<T>(path, { method: "POST", body: JSON.stringify(data ?? {}) }),
  patch: <T>(path: string, data?: unknown) =>
    request<T>(path, { method: "PATCH", body: JSON.stringify(data ?? {}) }),
};

// ── Types mirroring what the API actually returns ───────────────────────────
//
// `| null` on the numbers is not defensive typing, it is the contract. The read
// API distinguishes "measured as zero" from "not measured at all" — a tenant
// still on a v1 JWT produces no ledger rows, so its request count is null — and
// TypeScript is what stops a component rendering that as 0.

export type Role = "viewer" | "operator" | "owner";

export interface Me {
  id: string | null;
  email: string;
  name: string;
  role: Role;
  is_break_glass: boolean;
}

export const ROLE_ORDER: Role[] = ["viewer", "operator", "owner"];

/** Server enforces this too. Here it only decides what to render. */
export function can(role: Role | undefined, min: Role): boolean {
  if (!role) return false;
  return ROLE_ORDER.indexOf(role) >= ROLE_ORDER.indexOf(min);
}
