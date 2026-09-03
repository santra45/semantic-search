import {
  createContext, useCallback, useContext, useMemo, useRef, useState, type ReactNode,
} from "react";

/**
 * Toasts, including the undo affordance the write API is built around.
 *
 * An `action` is optional and deliberately NOT auto-dismissed on the same
 * timer as a plain notice: a toast offering "Undo" that vanishes in three
 * seconds is a promise the UI does not keep. Notices fade; actionable toasts
 * hold for longer and can be dismissed explicitly.
 */

export type ToastKind = "success" | "error" | "info";

export interface Toast {
  id: number;
  kind: ToastKind;
  message: string;
  /** Optional second line — used for the server's own explanation of a refusal. */
  detail?: string;
  action?: { label: string; run: () => void };
}

interface ToastApi {
  push: (t: Omit<Toast, "id">) => number;
  dismiss: (id: number) => void;
  success: (message: string, detail?: string) => number;
  error: (message: string, detail?: string) => number;
}

const ToastContext = createContext<ToastApi | null>(null);

const NOTICE_MS = 4_000;
const ACTION_MS = 10_000;

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const nextId = useRef(1);
  // Tracked so dismiss() can cancel a pending timer — otherwise a toast the
  // user closed by hand fires its removal later against an id that has been
  // reused, and closes somebody else's toast.
  const timers = useRef(new Map<number, ReturnType<typeof setTimeout>>());

  const dismiss = useCallback((id: number) => {
    const t = timers.current.get(id);
    if (t) {
      clearTimeout(t);
      timers.current.delete(id);
    }
    setToasts((cur) => cur.filter((x) => x.id !== id));
  }, []);

  const push = useCallback(
    (t: Omit<Toast, "id">) => {
      const id = nextId.current++;
      setToasts((cur) => [...cur, { ...t, id }]);
      const ms = t.action ? ACTION_MS : NOTICE_MS;
      timers.current.set(
        id,
        setTimeout(() => dismiss(id), ms),
      );
      return id;
    },
    [dismiss],
  );

  const api = useMemo<ToastApi>(
    () => ({
      push,
      dismiss,
      success: (message, detail) => push({ kind: "success", message, detail }),
      error: (message, detail) => push({ kind: "error", message, detail }),
    }),
    [push, dismiss],
  );

  return (
    <ToastContext.Provider value={api}>
      {children}
      <div className="toasts" role="status" aria-live="polite">
        {toasts.map((t) => (
          <div key={t.id} className={`toast toast-${t.kind}`}>
            <div className="toast-body">
              <div>{t.message}</div>
              {t.detail && <div className="toast-detail">{t.detail}</div>}
            </div>
            {t.action && (
              <button
                className="toast-action"
                onClick={() => {
                  t.action!.run();
                  dismiss(t.id);
                }}
              >
                {t.action.label}
              </button>
            )}
            <button className="toast-close" onClick={() => dismiss(t.id)} aria-label="Dismiss">
              ×
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

export function useToast(): ToastApi {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error("useToast() outside <ToastProvider>.");
  return ctx;
}
