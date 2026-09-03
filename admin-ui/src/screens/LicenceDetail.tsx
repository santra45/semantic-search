import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import { api } from "../api";
import { useActions } from "../lib/actions";
import { useCan } from "../lib/session";
import { ABSENT, cost, num, productLabel, relative, when } from "../lib/format";
import { ErrorNote, Loading } from "../components/Bits";
import { openRevoke } from "./tenantActions";

/**
 * One key, and the chain it belongs to.
 *
 * ────────────────────────────────────────────────────────────────────────────
 * THE TIMELINE IS THE SCREEN. THE FIELDS ARE THE FOOTNOTE.
 *
 * A licence row has seven columns and none of them are interesting on their
 * own. What an operator comes here to answer is "why did this key stop
 * working", and since issue_licence() DELETES the key it rotates out, the
 * licences table alone cannot answer it — the predecessor is gone.
 * licence_events is the only surviving record that it existed.
 *
 * So the timeline shows the whole chain for the SUBSCRIPTION, not just this
 * licence: keys that came before, what replaced them, and who did it. Events
 * belonging to a different key are marked as such rather than being blended
 * in, because "this key was superseded" and "the key before this one was
 * superseded" are different sentences.
 *
 * AND THE USAGE FIGURES ARE NOT THIS KEY'S. usage_events carries no licence_id
 * — spend is attributed to the subscription — so the numbers here span every
 * key the subscription has ever held. Presenting them as this licence's usage
 * would be wrong the moment anything was rotated, which is exactly when
 * somebody is reading this page.
 * ────────────────────────────────────────────────────────────────────────────
 */

interface Event {
  id: number;
  event: string;
  detail: string | null;
  key_prefix: string | null;
  actor_email: string | null;
  created_at: string | null;
  is_this_licence: boolean;
}

interface LicenceDetailData {
  licence: {
    id: string; key_prefix: string | null; has_plaintext: boolean;
    is_active: boolean; issued_at: string | null;
    expires_at: string | null; revoked_at: string | null;
  };
  subscription: {
    id: string; product_code: string; status: string; plan: string;
    request_limit: number; disabled_reason: string | null;
  };
  site: { id: string; domain: string; environment: string; platform: string };
  client: { id: string; name: string };
  events: Event[];
  usage_for_subscription: {
    requests: number | null; cost: number | null; tokens: number | null;
    first_seen: string | null; last_seen: string | null;
  };
}

export function LicenceDetail() {
  const { licenceId } = useParams();
  const isOwner = useCan("owner");
  const act = useActions([["licence", licenceId], ["licences"], ["overview"]]);

  const { data, isLoading, error } = useQuery<LicenceDetailData>({
    queryKey: ["licence", licenceId],
    queryFn: () => api.get<LicenceDetailData>(`/api/admin/licences/${licenceId}`),
  });

  if (isLoading) return <Loading what="licence" />;
  if (error) return <ErrorNote error={error} />;
  if (!data) return null;

  const { licence, subscription, site, client, events, usage_for_subscription: usage } = data;
  const expired = !!licence.expires_at && new Date(licence.expires_at) <= new Date();
  const dead = !licence.is_active || expired;
  const state = !licence.is_active ? "revoked" : expired ? "expired" : "active";

  return (
    <div className="ld">
      {act.sheetNode}

      <nav className="ld-crumb">
        <Link to="/licences">Licences</Link>
        <span aria-hidden="true">/</span>
        <span className="mono">{licence.key_prefix ?? ABSENT}</span>
      </nav>

      <header className={dead ? "ld-head is-dead" : "ld-head"}>
        <div>
          <h1 className="ld-prefix">{licence.key_prefix ?? ABSENT}</h1>
          <p className="ld-where">
            {productLabel(subscription.product_code)} ·{" "}
            <Link to={`/sites/${site.id}`}>{site.domain}</Link> ·{" "}
            <Link to={`/tenants/${client.id}`}>{client.name}</Link>
            {site.environment === "production"
              ? <>{" "}<span className="pill pill-ok">prod</span></>
              : <>{" "}<span className="pill pill-muted">dev</span></>}
          </p>
        </div>

        <div className="ld-head-right">
          <span className={state === "active" ? "pill pill-ok" : "pill pill-bad"}>{state}</span>
          {isOwner && <Reveal licence={licence} />}
          {isOwner && licence.is_active && (
            <button
              className="td-btn is-danger"
              onClick={() =>
                openRevoke(act, {
                  id: licence.id,
                  key_prefix: licence.key_prefix,
                  domain: site.domain,
                  product_code: subscription.product_code,
                })
              }
            >
              Revoke
            </button>
          )}
        </div>
      </header>

      <dl className="ld-facts">
        <Fact label="Issued" value={when(licence.issued_at)} />
        <Fact
          label="Expires"
          value={licence.expires_at ? when(licence.expires_at) : "no expiry"}
          hint={licence.expires_at ? relative(licence.expires_at) : undefined}
        />
        <Fact label="Revoked" value={licence.revoked_at ? when(licence.revoked_at) : ABSENT} />
        <Fact
          label="Plaintext"
          value={licence.has_plaintext ? "stored" : "never stored"}
          hint={
            licence.has_plaintext
              ? "can be shown again"
              : "predates the column — rotation is the only route to a readable key"
          }
        />
      </dl>

      <div className="ld-split">
        <Timeline events={events} />

        <aside className="ld-side">
          <section className="ld-card">
            <span className="eyebrow">Subscription</span>
            <p className="ld-sub-name">{productLabel(subscription.product_code)}</p>
            <ul className="ld-kv">
              <li><span>Status</span><em>{subscription.status}</em></li>
              <li><span>Plan</span><em>{subscription.plan}</em></li>
              <li>
                <span>Limit</span>
                <em>{num(subscription.request_limit)} req/mo <i>not enforced</i></em>
              </li>
            </ul>
            {subscription.disabled_reason && (
              <p className="ld-reason">
                <span className="eyebrow">Shown to the merchant</span>
                {subscription.disabled_reason}
              </p>
            )}
          </section>

          <section className="ld-card">
            {/* Labelled at the top, not in a footnote. usage_events has no
                licence_id, so these figures cover every key this subscription
                has ever held — which is precisely the distinction somebody
                reading a rotated key's page needs. */}
            <span className="eyebrow">Usage · whole subscription</span>
            <p className="ld-note">
              Spend is attributed to the subscription, not to a key, so this
              spans every licence it has held — including any this one replaced.
            </p>
            <ul className="ld-kv">
              <li><span>Requests</span><em>{num(usage.requests)}</em></li>
              <li><span>Cost</span><em>{cost(usage.cost)}</em></li>
              <li><span>Tokens</span><em>{num(usage.tokens)}</em></li>
              <li><span>First seen</span><em>{when(usage.first_seen)}</em></li>
              <li><span>Last seen</span><em>{when(usage.last_seen)}</em></li>
            </ul>
          </section>
        </aside>
      </div>
    </div>
  );
}

function Fact({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className={value === ABSENT ? "ld-fact is-absent" : "ld-fact"}>
      <dt className="eyebrow">{label}</dt>
      <dd>{value}{hint && <span>{hint}</span>}</dd>
    </div>
  );
}

const EVENT_WORDS: Record<string, string> = {
  issued: "Issued",
  rotated: "Rotated in",
  superseded: "Superseded",
  revoked: "Revoked",
  expired: "Expired",
};

function Timeline({ events }: { events: Event[] }) {
  if (!events.length) {
    return (
      <section className="ld-card">
        <span className="eyebrow">History</span>
        <p className="ld-note">
          No recorded events. licence_events only started being written on
          2026-09-03, so a key issued before then has no history here — its
          existence is the only record.
        </p>
      </section>
    );
  }

  return (
    <section className="ld-card ld-timeline-card">
      <span className="eyebrow">History · whole key chain</span>
      <p className="ld-note">
        Includes keys this subscription held previously. A rotation deletes the
        key it replaces, so these events are the only record those ever existed.
      </p>

      <ol className="ld-timeline">
        {events.map((e) => (
          <li key={e.id} className={e.is_this_licence ? "ld-ev is-this" : "ld-ev"}>
            <span className="ld-ev-dot" aria-hidden="true" />
            <div className="ld-ev-body">
              <div className="ld-ev-top">
                <span className="ld-ev-word">{EVENT_WORDS[e.event] ?? e.event}</span>
                {e.key_prefix && <span className="ld-ev-key">{e.key_prefix}</span>}
                {/* Only marked on the OTHER keys. Tagging every row "this key"
                    would be noise on the common case. */}
                {!e.is_this_licence && (
                  <span className="pill pill-muted">earlier key</span>
                )}
                <span className="ld-ev-time">{when(e.created_at)}</span>
              </div>
              {e.detail && <p className="ld-ev-detail">{e.detail}</p>}
              {e.actor_email && <p className="ld-ev-actor">{e.actor_email}</p>}
              {!e.actor_email && (
                // Events with no actor are the reason licence_events is separate
                // from admin_audit_log — an expiry is nobody's action.
                <p className="ld-ev-actor is-system">no actor — recorded by the system</p>
              )}
            </div>
          </li>
        ))}
      </ol>
    </section>
  );
}

function Reveal({ licence }: { licence: LicenceDetailData["licence"] }) {
  const [shown, setShown] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [failed, setFailed] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  if (!licence.has_plaintext) {
    return <span className="lc-nokey">cannot be shown</span>;
  }
  if (shown) {
    return (
      <span className="lc-shown">
        <code>{shown}</code>
        <button className="td-btn" onClick={() => {
          navigator.clipboard?.writeText(shown).then(
            () => { setCopied(true); setTimeout(() => setCopied(false), 1500); },
            () => setCopied(false));
        }}>{copied ? "Copied" : "Copy"}</button>
        <button className="td-btn" onClick={() => setShown(null)}>Hide</button>
      </span>
    );
  }
  return (
    <>
      <button className="td-btn" disabled={busy} onClick={async () => {
        setBusy(true); setFailed(null);
        try {
          const r = await api.get<{ key: string }>(`/api/admin/licences/${licence.id}/key`);
          setShown(r.key);
        } catch (e) {
          setFailed(e instanceof Error ? e.message : String(e));
        } finally { setBusy(false); }
      }}>{busy ? "…" : "Show key"}</button>
      {failed && <span className="lc-failed">{failed}</span>}
    </>
  );
}
