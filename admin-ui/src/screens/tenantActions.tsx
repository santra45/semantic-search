import { useQuery } from "@tanstack/react-query";
import { api } from "../api";
import type { useActions } from "../lib/actions";
import { num, productLabel, when } from "../lib/format";
import { rungLabel, type PlanRung } from "../lib/plans";
import type { Client, Site, Subscription } from "../lib/tenantTypes";

/**
 * What each write action SAYS.
 *
 * The sheet owns the mechanics — reason field, type-to-confirm, focus trap,
 * error rendering. These own the words, and the words are the part an operator
 * actually reads at the moment of deciding. Two rules run through all of them:
 *
 *   · Describe what happens, in present tense. Never "are you sure?" — that
 *     asks for a feeling when the operator needs a fact.
 *   · The verb is stable through the flow. The button says "Suspend", the toast
 *     says "Suspended". A control that changes its name mid-action makes people
 *     wonder whether they did the thing they meant to.
 */

export type Act = ReturnType<typeof useActions>;

export function openClientDisable(act: Act, client: Client) {
  act.open(
    {
      title: `Suspend ${client.name}`,
      tone: "danger",
      verb: "Suspend customer",
      reasonRequired: true,
      body: (
        <>
          Every store and every module under this customer stops authenticating
          immediately. Their individual states are preserved — restoring the
          customer puts each one back as it was, rather than turning everything
          on.
        </>
      ),
      // Fetched from the server rather than counted off the page, so the
      // operator sees what the BACKEND thinks is affected — including anything
      // the current view has filtered out of sight.
      blastRadius: <BlastRadius path={`/api/admin/clients/${client.id}/blast-radius`} />,
    },
    async (reason) => {
      await api.post(`/api/admin/clients/${client.id}/disable`, { reason });
      return {
        message: `Suspended ${client.name}`,
        detail: "Cached authorisations were evicted, so it takes effect now.",
        undo: {
          label: "Restore",
          run: () => api.post(`/api/admin/clients/${client.id}/enable`, {}),
        },
      };
    },
  );
}

export function openClientEnable(act: Act, client: Client) {
  act.open(
    {
      title: `Restore ${client.name}`,
      verb: "Restore customer",
      body: (
        <>
          Stores and modules return to their own states. Anything switched off
          for its own reasons stays off.
        </>
      ),
    },
    async (reason) => {
      await api.post(`/api/admin/clients/${client.id}/enable`, { reason });
      return { message: `Restored ${client.name}` };
    },
  );
}

export function openSiteDisable(act: Act, site: Site) {
  act.open(
    {
      title: `Suspend ${site.domain}`,
      tone: "danger",
      verb: "Suspend store",
      reasonRequired: true,
      body: (
        <>
          Every module on this store stops working. Other stores belonging to
          this customer are unaffected.
        </>
      ),
    },
    async (reason) => {
      await api.post(`/api/admin/sites/${site.id}/disable`, { reason });
      return {
        message: `Suspended ${site.domain}`,
        undo: { label: "Restore", run: () => api.post(`/api/admin/sites/${site.id}/enable`, {}) },
      };
    },
  );
}

export function openSiteEnable(act: Act, site: Site) {
  act.open(
    {
      title: `Restore ${site.domain}`,
      verb: "Restore store",
      body: <>Modules on this store return to their own states.</>,
    },
    async (reason) => {
      await api.post(`/api/admin/sites/${site.id}/enable`, { reason });
      return { message: `Restored ${site.domain}` };
    },
  );
}

export function openPromote(act: Act, site: Site) {
  act.open(
    {
      title: `Promote ${site.domain} to production`,
      tone: "danger",
      verb: "Promote",
      // Type-to-confirm, because this is not reversible in any useful sense: it
      // changes how every future usage row is attributed and leaves the store's
      // existing czg_test_ keys disagreeing with it.
      confirmText: site.domain,
      confirmLabel: "Type the domain to confirm",
      body: (
        <>
          Spend on this store stops being booked as Czargroup's and starts being
          booked as the merchant's — <code>key_owner</code> follows the site, not
          the key. New keys are issued as <code>czg_live_</code>.
          <br />
          <br />
          Existing keys keep working but will carry the wrong environment in
          their prefix. Reissue them straight after.
        </>
      ),
    },
    async (reason) => {
      const r = await api.post<{ live_licences_needing_reissue: number }>(
        `/api/admin/sites/${site.id}/environment`,
        { environment: "production", confirm_domain: site.domain, reason },
      );
      return {
        message: `${site.domain} is now production`,
        detail: r.live_licences_needing_reissue
          ? `${r.live_licences_needing_reissue} key(s) still say test — reissue them.`
          : undefined,
      };
    },
  );
}

export function openPause(act: Act, sub: Subscription) {
  act.open(
    {
      title: `Pause ${productLabel(sub.product_code)}`,
      tone: "danger",
      verb: "Pause module",
      reasonRequired: true,
      body: (
        <>
          This module stops on <code>{sub.domain}</code> only. The reason you
          give is shown to the <strong>merchant</strong> — it appears in their
          storefront and in the 403 their plugin receives, so write it for them
          rather than for us.
        </>
      ),
    },
    async (reason) => {
      await api.post(`/api/admin/subscriptions/${sub.id}/pause`, { reason });
      return {
        message: `Paused ${productLabel(sub.product_code)}`,
        detail: `on ${sub.domain}`,
        undo: {
          label: "Resume",
          run: () => api.post(`/api/admin/subscriptions/${sub.id}/resume`, {}),
        },
      };
    },
  );
}

export function openResume(act: Act, sub: Subscription) {
  act.open(
    {
      title: `Resume ${productLabel(sub.product_code)}`,
      verb: "Resume module",
      body: (
        <>
          The module serves again on <code>{sub.domain}</code> and the
          merchant-facing pause message is cleared.
          {sub.plan === "trial" && (
            <>
              {" "}It returns to <strong>trial</strong>, not active — a trial plan
              cannot be made active without naming the plan they bought.
            </>
          )}
        </>
      ),
    },
    async (reason) => {
      await api.post(`/api/admin/subscriptions/${sub.id}/resume`, { reason });
      return {
        message: `Resumed ${productLabel(sub.product_code)}`,
        detail: `on ${sub.domain}`,
      };
    },
  );
}

export function openIssue(act: Act, sub: Subscription, rotating: boolean) {
  act.open(
    {
      title: rotating
        ? `Rotate the ${productLabel(sub.product_code)} key`
        : `Issue a ${productLabel(sub.product_code)} key`,
      tone: rotating ? "danger" : "normal",
      verb: rotating ? "Rotate key" : "Issue key",
      reasonRequired: rotating,
      body: rotating ? (
        <>
          <strong>The current key stops working the moment this returns.</strong>{" "}
          Its cached authorisations are evicted immediately, so{" "}
          <code>{sub.domain}</code> breaks until the new key is pasted into the
          module config. The old key is deleted rather than disabled — there is
          nothing to roll back to.
        </>
      ) : (
        <>
          Mints the first key for this module on <code>{sub.domain}</code>. The
          plaintext is stored, so it can be read again from the licence page —
          this is not your only chance to copy it.
        </>
      ),
    },
    async (reason) => {
      const r = await api.post<{ key_prefix: string; superseded: number }>(
        `/api/admin/subscriptions/${sub.id}/licence`,
        { valid_days: 365, reason },
      );
      // No undo offered. A rotation DELETES the row it replaces, so there is
      // nothing to put back, and an affordance that cannot work is worse than
      // none at all.
      return {
        message: rotating ? `Rotated to ${r.key_prefix}` : `Issued ${r.key_prefix}`,
        detail: rotating
          ? `The previous key is dead. Paste the new one into ${sub.domain}.`
          : `Paste it into ${sub.domain}.`,
      };
    },
  );
}

function BlastRadius({ path }: { path: string }) {
  const { data, isLoading, error } = useQuery<{
    sites: number;
    subscriptions: number;
    live_licences: number;
    requests_per_day: number | null;
  }>({ queryKey: ["blast", path], queryFn: () => api.get(path) });

  if (isLoading) return <span className="muted">Checking what this affects…</span>;
  // Failing to load the preview must not block the action — but it must be said
  // out loud, because the operator is about to decide without it.
  if (error || !data)
    return <span className="muted">Could not load the blast radius. Proceeding blind.</span>;

  return (
    <>
      <span className="eyebrow">This will affect</span>
      <div className="cs-blast-figs">
        <span><strong>{num(data.sites)}</strong> sites</span>
        <span><strong>{num(data.subscriptions)}</strong> subscriptions</span>
        <span><strong>{num(data.live_licences)}</strong> live keys</span>
        <span><strong>{num(data.requests_per_day)}</strong> requests/day</span>
      </div>
    </>
  );
}

/* ── Edits ───────────────────────────────────────────────────────────────────
 *
 * These carry a value rather than only a confirmation, and the two plan ladders
 * are deliberately edited from different places: catalogue size lives on the
 * SITE because a store's modules share one Qdrant collection, request quota
 * lives on the SUBSCRIPTION because it is bought per module. One combined
 * "plan" control would silently change the wrong one.
 */

export function openPlanEdit(act: Act, sub: Subscription, rungs: PlanRung[]) {
  act.open(
    {
      title: `Change the plan for ${productLabel(sub.product_code)}`,
      verb: "Change plan",
      fields: [{
        name: "plan",
        label: "Request quota — bought per module",
        type: "select",
        value: sub.plan,
        options: rungs.map((r) => ({ value: r.code, label: rungLabel(r) })),
        hint: "Moving off trial also moves the subscription to active.",
      }],
      body: (
        <>
          Changes the monthly request allowance on <code>{sub.domain}</code>.
          Nothing enforces it today — <code>AICHATBOT_QUOTA_ENFORCEMENT</code> is
          unset — so this records what they bought rather than capping what they
          can do.
        </>
      ),
    },
    async (reason, values) => {
      const r = await api.patch<{ plan: string; request_limit: number }>(
        `/api/admin/subscriptions/${sub.id}/plan`,
        { plan: values.plan, reason },
      );
      return {
        message: `${productLabel(sub.product_code)} is now ${r.plan}`,
        detail: `${r.request_limit.toLocaleString()} requests/mo · still not enforced`,
      };
    },
  );
}

export function openTermEdit(act: Act, sub: Subscription) {
  act.open(
    {
      title: `Extend ${productLabel(sub.product_code)}`,
      verb: "Extend term",
      fields: [{
        name: "extend_days",
        label: "Extend by",
        type: "select",
        value: "365",
        options: [
          { value: "30", label: "30 days" },
          { value: "90", label: "90 days" },
          { value: "365", label: "1 year" },
        ],
      }],
      body: (
        <>
          Extends from the current expiry, not from today — a term with three
          months left gains the extension on top rather than restarting the
          clock and losing them.
        </>
      ),
    },
    async (reason, values) => {
      const r = await api.patch<{ expires_at: string | null }>(
        `/api/admin/subscriptions/${sub.id}/term`,
        { extend_days: Number(values.extend_days), reason },
      );
      return {
        message: `Extended ${productLabel(sub.product_code)}`,
        detail: r.expires_at ? `now expires ${when(r.expires_at)}` : undefined,
      };
    },
  );
}

export function openIndexPlanEdit(act: Act, site: Site, rungs: PlanRung[]) {
  act.open(
    {
      title: `Change the catalogue plan for ${site.domain}`,
      verb: "Change catalogue plan",
      fields: [{
        name: "index_plan",
        label: "Catalogue size — bought once per store",
        type: "select",
        value: site.index_plan,
        options: rungs.map((r) => ({ value: r.code, label: rungLabel(r) })),
        hint: `${site.indexed_items.toLocaleString()} items are already indexed.`,
      }],
      body: (
        <>
          This ceiling <strong>is</strong> enforced — writes past it are refused.
          A downgrade below what is already indexed will be rejected, because a
          store sitting over its own ceiling has no clean way out: nothing
          deletes the catalogue, and every later sync fails a check the merchant
          cannot act on.
        </>
      ),
    },
    async (reason, values) => {
      const r = await api.patch<{ catalogue_limit: number }>(
        `/api/admin/sites/${site.id}/index-plan`,
        { index_plan: values.index_plan, reason },
      );
      return {
        message: `${site.domain} is now on ${values.index_plan}`,
        detail: `ceiling ${r.catalogue_limit.toLocaleString()} items`,
      };
    },
  );
}

export function openRevoke(act: Act, licence: { id: string; key_prefix: string | null; domain: string; product_code: string }) {
  act.open(
    {
      title: `Revoke ${licence.key_prefix ?? "this key"}`,
      tone: "danger",
      verb: "Revoke permanently",
      reasonRequired: true,
      confirmText: licence.key_prefix ?? undefined,
      confirmLabel: "Type the key prefix to confirm",
      body: (
        <>
          <strong>There is no undo.</strong> The key stops working immediately on{" "}
          <code>{licence.domain}</code> and cannot be re-enabled — a revoked
          licence stays revoked. If the module needs to keep running, rotate the
          key instead: that issues a replacement in the same step.
        </>
      ),
    },
    async (reason) => {
      await api.post(`/api/admin/licences/${licence.id}/revoke`, {
        reason,
        confirm_prefix: licence.key_prefix ?? "",
      });
      return {
        message: `Revoked ${licence.key_prefix ?? "the key"}`,
        detail: `${productLabel(licence.product_code)} on ${licence.domain} can no longer authenticate.`,
      };
    },
  );
}
