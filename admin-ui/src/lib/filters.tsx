import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";

/**
 * The global environment filter, and the reporting window.
 *
 * ────────────────────────────────────────────────────────────────────────────
 * WHY ENVIRONMENT IS A FIRST-CLASS CONTROL RATHER THAN A COLUMN.
 *
 * Every site in the database is `development` today — they are Czargroup demo
 * and test stores running on Czargroup's own API keys. The first real customer
 * does not replace those rows, it lands ALONGSIDE them.
 *
 * So on the day it matters, every unfiltered headline number silently starts
 * including demo traffic, and there is no point at which somebody notices,
 * because the number was always "right". Building the filter now costs a
 * dropdown; retrofitting it costs a period where nobody can say which figures
 * were already wrong.
 *
 * Default is `all` and dev rows carry a visible badge, so nothing is hidden by
 * accident — the filter narrows, it does not quietly exclude.
 * ────────────────────────────────────────────────────────────────────────────
 */

export type EnvFilter = "all" | "development" | "production";

interface Filters {
  environment: EnvFilter;
  setEnvironment: (v: EnvFilter) => void;
  /** Reporting window in days. Clamped server-side to 1..365 regardless. */
  days: number;
  setDays: (v: number) => void;
  /** Appends `environment` only when narrowed, so `all` sends no parameter. */
  envParam: () => string;
}

const FiltersContext = createContext<Filters | null>(null);

const ENV_KEY = "admin.environment";
const DAYS_KEY = "admin.days";

function readStored<T extends string | number>(key: string, fallback: T, coerce: (s: string) => T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw === null ? fallback : coerce(raw);
  } catch {
    // Private mode, disabled storage, a sandboxed iframe. A remembered filter
    // is a convenience; failing to read one must never stop the console
    // rendering.
    return fallback;
  }
}

function store(key: string, value: string) {
  try {
    localStorage.setItem(key, value);
  } catch {
    /* see above */
  }
}

export function FiltersProvider({ children }: { children: ReactNode }) {
  const [environment, setEnvRaw] = useState<EnvFilter>(() =>
    readStored<EnvFilter>(ENV_KEY, "all", (s) =>
      s === "development" || s === "production" ? s : "all",
    ),
  );
  const [days, setDaysRaw] = useState<number>(() =>
    readStored<number>(DAYS_KEY, 30, (s) => {
      const n = Number(s);
      return Number.isFinite(n) && n > 0 && n <= 365 ? n : 30;
    }),
  );

  const setEnvironment = useCallback((v: EnvFilter) => {
    setEnvRaw(v);
    store(ENV_KEY, v);
  }, []);

  const setDays = useCallback((v: number) => {
    setDaysRaw(v);
    store(DAYS_KEY, String(v));
  }, []);

  const envParam = useCallback(
    () => (environment === "all" ? "" : `&environment=${environment}`),
    [environment],
  );

  const value = useMemo(
    () => ({ environment, setEnvironment, days, setDays, envParam }),
    [environment, setEnvironment, days, setDays, envParam],
  );

  return <FiltersContext.Provider value={value}>{children}</FiltersContext.Provider>;
}

export function useFilters(): Filters {
  const ctx = useContext(FiltersContext);
  if (!ctx) throw new Error("useFilters() outside <FiltersProvider>.");
  return ctx;
}
