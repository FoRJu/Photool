import { useEffect, useState } from "react";

// ── Two cameras, one moment ─────────────────────────────────────────
//
// Two photos selected in the grid that were taken at the same instant on two
// cameras (a flash going off, a jump at its top, a clock in both frames). One
// camera's clock is taken as right and the other's is moved to agree: a rule
// for that body alone, so two R6 Mark IIs are two clocks (backend
// catalog/clock.py align). Nothing is written to the files.

const API = "/api";

const SPANS = [
  { id: "day", label: "That day", hint: "Only the day this photo's clock said. Safest: a clock is often set again between trips." },
  { id: "onward", label: "From that day on", hint: "Every photo from this body from that day, until a later rule says otherwise." },
  { id: "always", label: "Always", hint: "Every photo this body ever took." },
];

function shownTime(raw) {
  const m = /^(\d{4}):(\d\d):(\d\d) (\d\d):(\d\d):(\d\d)/.exec(raw || "");
  return m ? `${m[1]}-${m[2]}-${m[3]} ${m[4]}:${m[5]}:${m[6]}` : "no time";
}

function seconds(raw) {
  const m = /^(\d{4}):(\d\d):(\d\d) (\d\d):(\d\d):(\d\d)/.exec(raw || "");
  return m ? Date.UTC(+m[1], m[2] - 1, +m[3], +m[4], +m[5], +m[6]) / 1000 : null;
}

export function shiftLabel(s) {
  if (!s) return "on time";
  const a = Math.abs(s);
  const parts = [];
  if (a >= 86400) parts.push(`${Math.floor(a / 86400)} d`);
  if (a % 86400 >= 3600) parts.push(`${Math.floor((a % 86400) / 3600)} h`);
  if (a % 3600 >= 60) parts.push(`${Math.floor((a % 3600) / 60)} min`);
  if (a % 60) parts.push(`${a % 60} s`);
  return `${s > 0 ? "+" : "−"}${parts.join(" ")}`;
}

function Side({ photo, thumb, right, onPick }) {
  return (
    <button onClick={onPick} data-testid="same-moment-side"
      className={`flex-1 min-w-0 rounded border p-2 text-left ${right ? "border-sky-500 bg-sky-950/40" : "border-neutral-700 hover:border-neutral-500"}`}>
      <img src={thumb(photo)} alt={photo.filename} className="w-full h-36 object-contain bg-black rounded" />
      <div className="mt-1.5 truncate text-neutral-200">{photo.filename}</div>
      <div className="text-xs text-neutral-500 truncate">{photo.camera || "Unknown camera"}</div>
      <div className="text-xs font-mono text-neutral-300">{shownTime(photo.capture_date)}</div>
      <div className={`mt-1 text-xs ${right ? "text-sky-300" : "text-neutral-600"}`}>
        {right ? "✓ This clock was right" : "Its clock is moved"}
      </div>
    </button>
  );
}

export default function SameMomentDialog({ first, other, thumb, onClose, onDone }) {
  const [rightIs, setRightIs] = useState(0);
  const [span, setSpan] = useState("day");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const pair = [first, other];
  const reference = pair[rightIs];
  const moved = pair[1 - rightIs];
  const a = seconds(reference.capture_date);
  const b = seconds(moved.capture_date);
  const gap = a !== null && b !== null ? a - b : null;

  useEffect(() => {
    const key = (e) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [onClose]);

  const apply = async () => {
    setBusy(true);
    setError("");
    try {
      const res = await fetch(`${API}/library/clocks/align`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reference: reference.filepath, other: moved.filepath, span }),
      });
      const out = await res.json();
      if (!res.ok) { setError(out.detail || "Not lined up"); return; }
      onDone(out.aligned);
    } catch { setError("The backend is not reachable"); } finally { setBusy(false); }
  };

  return (
    <div className="fixed inset-0 z-[1000] flex items-start justify-center bg-black/60 p-4 overflow-y-auto" data-modal
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="w-full max-w-xl rounded-lg border border-neutral-700 bg-neutral-900 p-4 space-y-3 text-sm" data-testid="same-moment">
        <div className="flex items-start gap-3">
          <div className="flex-1">
            <h2 className="text-base font-medium text-neutral-100">Taken at the same moment</h2>
            <p className="text-xs text-neutral-500">
              Pick the camera whose clock was right. The other camera's clock is moved to agree, for that body
              alone. Nothing is written to the files.
            </p>
          </div>
          <button onClick={onClose} className="text-neutral-500 hover:text-white">✕</button>
        </div>
        <div className="flex gap-2">
          {pair.map((p, i) => (
            <Side key={p.filepath} photo={p} thumb={thumb} right={i === rightIs} onPick={() => setRightIs(i)} />
          ))}
        </div>
        <p className="text-xs text-neutral-300" data-testid="same-moment-gap">
          {gap === null ? "One of these has no time to line up."
            : gap === 0 ? "These already read the same time."
            : `${moved.camera || "The other camera"} is moved ${shiftLabel(gap)}.`}
        </p>
        <div className="space-y-1">
          {SPANS.map((s) => (
            <label key={s.id} className="flex items-baseline gap-2 text-xs cursor-pointer" title={s.hint}>
              <input type="radio" name="same-moment-span" checked={span === s.id} onChange={() => setSpan(s.id)} />
              <span className="text-neutral-200">{s.label}</span>
              <span className="text-neutral-600">{s.hint}</span>
            </label>
          ))}
        </div>
        {error && <p className="text-xs text-red-400">{error}</p>}
        <div className="flex justify-end gap-2">
          <button onClick={onClose} className="px-2.5 py-1 rounded bg-neutral-800 hover:bg-neutral-700">Cancel</button>
          <button disabled={busy || gap === null} onClick={apply} data-testid="same-moment-apply"
            className="px-2.5 py-1 rounded bg-sky-800/60 text-sky-100 hover:bg-sky-700/60 disabled:opacity-40">
            Line up the clocks
          </button>
        </div>
      </div>
    </div>
  );
}
