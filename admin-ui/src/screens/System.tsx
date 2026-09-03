import { useQuery } from "@tanstack/react-query";
import { api } from "../api";
import { when } from "../lib/format";
import { ErrorNote, Loading } from "../components/Bits";

/**
 * System health.
 *
 * ────────────────────────────────────────────────────────────────────────────
 * NOT JUST "IS IT UP" — "WHAT IS OUTSTANDING".
 *
 * A dependency list of three green ticks is a screen nobody reads twice. The
 * useful half of this endpoint is the CONFIGURATION, and two of those values
 * are findings rather than statuses:
 *
 *   operator_key_configured  While AICHATBOT_OPERATOR_KEY is set, anyone
 *                            holding it has owner rights and their actions log
 *                            as "break-glass" with no named actor. It exists to
 *                            bootstrap the first owner. A real owner account now
 *                            exists, so it is outstanding work, not a feature.
 *
 *   quota_enforcement        Unset, which is WHY every request limit in this
 *                            console says "not enforced". Surfacing it here is
 *                            what connects that caveat to its cause.
 *
 * So findings come first and dependencies come second. A green tick is not
 * news; an unremoved break-glass key is.
 * ────────────────────────────────────────────────────────────────────────────
 */

interface HealthData {
  checks: Record<string, { ok: boolean; error?: string; collections?: number }>;
  schema: Record<string, boolean>;
  migrations: { version: string; filename: string; applied_at: string | null }[];
  flags: { quota_enforcement: string | null; operator_key_configured: boolean };
}

export function System() {
  const { data, isLoading, error } = useQuery<HealthData>({
    queryKey: ["health"],
    queryFn: () => api.get<HealthData>("/api/admin/health"),
    // Short, because this is the screen somebody opens WHILE something is
    // breaking, and a cached answer from ten minutes ago is worse than none.
    staleTime: 10_000,
    refetchInterval: 30_000,
  });

  if (isLoading) return <Loading what="system health" />;
  if (error) return <ErrorNote error={error} />;
  if (!data) return null;

  const missing = Object.entries(data.schema).filter(([, present]) => !present);
  const down = Object.entries(data.checks).filter(([, c]) => !c.ok);

  const findings: { tone: "bad" | "warn"; title: string; body: string }[] = [];

  if (down.length) {
    findings.push({
      tone: "bad",
      title: `${down.map(([n]) => n).join(", ")} unreachable`,
      body: "Storefront traffic may be degraded. Redis being down costs the auth cache and the login throttle; Qdrant being down stops search entirely.",
    });
  }
  if (missing.length) {
    findings.push({
      tone: "bad",
      title: `${missing.length} expected table${missing.length === 1 ? "" : "s"} missing`,
      body: `Missing: ${missing.map(([n]) => n).join(", ")}. Run scripts/migrate.py --apply.`,
    });
  }
  if (data.flags.operator_key_configured) {
    findings.push({
      tone: "warn",
      title: "The break-glass operator key is still configured",
      body: "Anyone holding AICHATBOT_OPERATOR_KEY has owner rights, and their actions are logged as 'break-glass' with no named actor. It exists to bootstrap the first owner account. Remove it from the production .env now that real accounts exist — while it is set, RBAC and the audit trail are decorative.",
    });
  }
  if (!data.flags.quota_enforcement) {
    findings.push({
      tone: "warn",
      title: "Request quotas are not enforced",
      body: "AICHATBOT_QUOTA_ENFORCEMENT is unset, so a tenant at 300% of plan keeps working. Every request limit shown in this console is observational — that caveat comes from here.",
    });
  }

  return (
    <div className="sy">
      <section className="sy-findings">
        <span className="eyebrow">Outstanding</span>
        {findings.length === 0 ? (
          <p className="sy-clear">
            <span className="sy-clear-mark" aria-hidden="true" />
            Nothing outstanding. Every dependency is reachable, every expected
            table exists, and no configuration is left in a bootstrap state.
          </p>
        ) : (
          <ul className="sy-finding-list">
            {findings.map((f) => (
              <li key={f.title} className={`sy-finding is-${f.tone}`}>
                <strong>{f.title}</strong>
                <p>{f.body}</p>
              </li>
            ))}
          </ul>
        )}
      </section>

      <div className="sy-split">
        <section className="sy-card">
          <span className="eyebrow">Dependencies</span>
          <ul className="sy-checks">
            {Object.entries(data.checks).map(([name, c]) => (
              <li key={name} className={c.ok ? "sy-check is-ok" : "sy-check is-bad"}>
                <span className="sy-dot" aria-hidden="true" />
                <span className="sy-check-name">{name}</span>
                <span className="sy-check-state">
                  {c.ok ? "reachable" : "unreachable"}
                  {c.collections !== undefined && ` · ${c.collections} collections`}
                </span>
                {/* The driver's own message. More useful than "failed", and it
                    is what somebody will paste into a search box. */}
                {c.error && <span className="sy-check-err">{c.error}</span>}
              </li>
            ))}
          </ul>
        </section>

        <section className="sy-card">
          <span className="eyebrow">Schema</span>
          <p className="sy-note">
            Probed by name rather than by trying a query — a swallowed "table
            does not exist" and a genuinely empty result look identical
            downstream.
          </p>
          <ul className="sy-tables">
            {Object.entries(data.schema).map(([name, present]) => (
              <li key={name} className={present ? "sy-table is-ok" : "sy-table is-bad"}>
                {name}
              </li>
            ))}
          </ul>
        </section>
      </div>

      <section className="sy-card">
        <span className="eyebrow">Applied migrations</span>
        {data.migrations.length === 0 ? (
          <p className="sy-note">
            No <code>schema_migrations</code> rows. Either nothing has been
            applied through the runner, or the table has not been created yet —
            it is created on the first <code>--apply</code>.
          </p>
        ) : (
          <ol className="sy-migrations">
            {data.migrations.map((m) => (
              <li key={m.version}>
                <span className="sy-mig-v">{m.version}</span>
                <span className="sy-mig-f">{m.filename}</span>
                <span className="sy-mig-t">{when(m.applied_at)}</span>
              </li>
            ))}
          </ol>
        )}
      </section>
    </div>
  );
}
