import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The console is served from /admin on the SAME origin as the API, so the
// session cookie needs no CORS and no token juggling. Both halves of that
// arrangement are configured here.
export default defineConfig({
  plugins: [react()],

  // Every asset URL is rewritten to /admin/assets/... . Without this the built
  // index.html requests /assets/... , which the API does not serve and the SPA
  // catch-all happily answers with index.html — so the browser gets HTML where
  // it expected JavaScript and the page renders blank with a MIME-type error.
  base: "/admin/",

  build: {
    // Straight into what FastAPI already mounts. Relative to admin-ui/.
    outDir: "../backend/app/static/admin",
    emptyOutDir: true,
    // No sourcemaps: they roughly double build memory on a box that is sharing
    // 1,967 MB with a live API, and an internal console is not worth that.
    sourcemap: false,
  },

  server: {
    // Dev only. `npm run dev` serves the SPA from Vite and forwards the API
    // calls to uvicorn, so the cookie stays same-origin from the browser's
    // point of view and dev behaves like production.
    proxy: {
      "/api": { target: "http://localhost:8000", changeOrigin: false },
    },
  },
});
