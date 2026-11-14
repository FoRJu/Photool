// Bring in a film look as a .cube LUT (a film simulation, say), kept as a style
// and rendered on any RAW like a learned one (backend adjust/film_lut.py).
import { useRef, useState } from "react";

const API = "/api";
const MAX_BYTES = 12 * 1024 * 1024;

export default function FilmLutImport({ onImported }) {
  const input = useRef(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const chosen = async (file) => {
    if (!file) return;
    setError("");
    if (file.size > MAX_BYTES) { setError("That file is too large for a LUT"); return; }
    setBusy(true);
    try {
      const text = await file.text();
      const res = await fetch(`${API}/styles/import-lut`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, filename: file.name }),
      });
      const data = await res.json();
      if (!res.ok) { setError(data.detail || "Could not read that LUT"); return; }
      onImported?.(data.style);
    } catch {
      setError("Could not read that file");
    } finally {
      setBusy(false);
      if (input.current) input.current.value = "";
    }
  };

  return (
    <span className="inline-flex flex-col">
      <input ref={input} type="file" accept=".cube,text/plain" className="hidden" data-testid="film-lut-file"
        onChange={(e) => chosen(e.target.files?.[0])} />
      <button onClick={() => input.current?.click()} disabled={busy} data-testid="film-lut-import"
        title="Use a film look made as a .cube LUT for a linear-contrast picture (sRGB or Display P3), such as a film simulation. It is kept on this computer as a style and set to each frame's own brightness."
        className="underline decoration-neutral-700 hover:text-neutral-300 disabled:opacity-40 text-left">
        {busy ? "Reading the LUT…" : "+ Film LUT (.cube)…"}
      </button>
      {error && <span className="text-red-400/80">{error}</span>}
    </span>
  );
}
