import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.jsx";
import "./styles/index.css";

// A tab left open while the server rebuilt the page asks for pieces (the map,
// Develop's parts) by names the new build no longer has. Load the new page
// instead of failing — once a minute at most, so a real fault cannot loop.
window.addEventListener("vite:preloadError", (event) => {
  let last = 0;
  try { last = Number(sessionStorage.getItem("photool.reloadedForBuild") || 0); } catch { /* none kept */ }
  if (Date.now() - last < 60000) return;
  try { sessionStorage.setItem("photool.reloadedForBuild", String(Date.now())); } catch { /* reload anyway */ }
  event.preventDefault();
  window.location.reload();
});

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
