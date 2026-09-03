import { useEffect, useRef, useState, type ReactNode } from "react";

/**
 * The one dialog every write goes through.
 *
 * ────────────────────────────────────────────────────────────────────────────
 * ONE COMPONENT, NOT ONE PER ACTION — for the same reason the backend puts every
 * mutation through mutate(). A guardrail that each call site has to remember is
 * a guardrail that will eventually be forgotten by one of them, and it will be
 * the destructive one.
 *
 * So the sheet owns all four of §7.4's rules and an action opts into them by
 * configuration rather than by reimplementing them:
 *
 *   reasonRequired   a mandatory sentence, stored in the audit row
 *   confirmText      type-to-confirm, matched exactly
 *   blastRadius      what this will affect, fetched BEFORE the button arms
 *   tone             how loud the thing looks
 *
 * The server enforces all of it again — a blank reason is a 422, a mismatched
 * confirm is a 422 — so this is about not letting somebody fire an action they
 * have not understood, not about being the thing that stops them.
 * ────────────────────────────────────────────────────────────────────────────
 */

/** A single form control on the sheet. Deliberately a tiny vocabulary: these
 *  are confirmations that happen to need one value, not a form builder. */
export interface SheetField {
  name: string;
  label: string;
  type: "select" | "number";
  options?: { value: string; label: string }[];
  value: string;
  hint?: string;
}

export interface ConfirmConfig {
  title: string;
  /** Plain description of what happens. Present tense, no "are you sure". */
  body: ReactNode;
  /** Button label. Same verb the toast will use — "Suspend" → "Suspended". */
  verb: string;
  tone?: "normal" | "danger";
  reasonRequired?: boolean;
  /** When set, the operator must type this exactly. */
  confirmText?: string;
  confirmLabel?: string;
  /** Rendered above the form. Fetch it before opening. */
  blastRadius?: ReactNode;
  /** Edits that need a value — a plan, a term. Empty for pure confirmations. */
  fields?: SheetField[];
  onConfirm: (reason: string, values: Record<string, string>) => Promise<unknown>;
}

export function ConfirmSheet({
  config, onClose,
}: { config: ConfirmConfig; onClose: () => void }) {
  const [reason, setReason] = useState("");
  const [typed, setTyped] = useState("");
  const [values, setValues] = useState<Record<string, string>>(() =>
    Object.fromEntries((config.fields ?? []).map((f) => [f.name, f.value])),
  );
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const firstField = useRef<HTMLTextAreaElement | HTMLInputElement | HTMLSelectElement | null>(null);
  const sheet = useRef<HTMLDivElement>(null);

  useEffect(() => {
    firstField.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !busy) onClose();
      // Focus trap. A modal that lets Tab wander into the page behind it is a
      // modal a keyboard user can fire blind.
      if (e.key === "Tab" && sheet.current) {
        const focusable = sheet.current.querySelectorAll<HTMLElement>(
          "button, input, textarea, select, a[href]",
        );
        if (!focusable.length) return;
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault(); last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault(); first.focus();
        }
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [busy, onClose]);

  const reasonOk = !config.reasonRequired || reason.trim().length >= 3;
  const typedOk = !config.confirmText || typed === config.confirmText;
  // An edit that has not changed anything is not an edit. Disabling it stops a
  // no-op write landing in the audit log as though something happened.
  const changed =
    !config.fields?.length ||
    config.fields.some((f) => values[f.name] !== f.value);
  const ready = reasonOk && typedOk && changed && !busy;

  async function run() {
    if (!ready) return;
    setBusy(true);
    setError(null);
    try {
      await config.onConfirm(reason.trim(), values);
      onClose();
    } catch (e) {
      // The server's own message, verbatim. licensing_service says things like
      // "Activating a trial needs the plan they bought" — better guidance than
      // anything this layer would invent.
      setError(e instanceof Error ? e.message : String(e));
      setBusy(false);
    }
  }

  return (
    <div className="cs-scrim" onMouseDown={(e) => { if (e.target === e.currentTarget && !busy) onClose(); }}>
      <div
        className={config.tone === "danger" ? "cs-sheet is-danger" : "cs-sheet"}
        role="dialog"
        aria-modal="true"
        aria-labelledby="cs-title"
        ref={sheet}
      >
        <h2 id="cs-title">{config.title}</h2>
        <div className="cs-body">{config.body}</div>

        {config.blastRadius && <div className="cs-blast">{config.blastRadius}</div>}

        {config.fields?.map((f, i) => (
          <label className="cs-field" key={f.name}>
            <span>{f.label}</span>
            {f.type === "select" ? (
              <select
                ref={i === 0 && !config.reasonRequired
                  ? (firstField as React.RefObject<HTMLSelectElement>) : undefined}
                value={values[f.name]}
                onChange={(e) => setValues((v) => ({ ...v, [f.name]: e.target.value }))}
              >
                {f.options?.map((o) => (
                  <option key={o.value} value={o.value}>{o.label}</option>
                ))}
              </select>
            ) : (
              <input
                type="number"
                value={values[f.name]}
                onChange={(e) => setValues((v) => ({ ...v, [f.name]: e.target.value }))}
              />
            )}
            {f.hint && <em className="cs-hint">{f.hint}</em>}
          </label>
        ))}

        {config.reasonRequired && (
          <label className="cs-field">
            <span>Reason</span>
            <textarea
              ref={firstField as React.RefObject<HTMLTextAreaElement>}
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              rows={2}
              placeholder="Why, for whoever reads the audit log in six months"
              maxLength={500}
            />
          </label>
        )}

        {config.confirmText && (
          <label className="cs-field">
            <span>
              {config.confirmLabel ?? "Type to confirm"}:{" "}
              <code>{config.confirmText}</code>
            </span>
            <input
              ref={!config.reasonRequired ? (firstField as React.RefObject<HTMLInputElement>) : undefined}
              value={typed}
              onChange={(e) => setTyped(e.target.value)}
              autoComplete="off"
              spellCheck={false}
            />
          </label>
        )}

        {error && <div className="err cs-err">{error}</div>}

        <div className="cs-actions">
          <button className="linkbtn" onClick={onClose} disabled={busy}>Cancel</button>
          <button
            className={config.tone === "danger" ? "cs-go is-danger" : "cs-go"}
            onClick={run}
            disabled={!ready}
          >
            {busy ? "Working…" : config.verb}
          </button>
        </div>
      </div>
    </div>
  );
}
