import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // xfwd: say who the browser was, so a device reaching this dev server over
      // the network is never taken for this computer (backend/access.py).
      "/api": { target: "http://localhost:8000", xfwd: true },
    },
  },
});
