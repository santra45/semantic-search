import React from "react";
import ReactDOM from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { App } from "./App";
import "./styles.css";
import "./overview.css";
import "./tenants.css";
import "./tenant-detail.css";
import "./confirm.css";
import "./licences.css";
import "./audit.css";
import "./usage.css";
import "./licence-detail.css";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // A 401 is an answer, not a transport failure — retrying it four times
      // just delays the login screen by a few seconds.
      retry: (count, error) =>
        (error as { status?: number })?.status === 401 ? false : count < 2,
      refetchOnWindowFocus: false,
      staleTime: 30_000,
    },
  },
});

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </React.StrictMode>,
);
