import { useState, useEffect, useCallback } from "react";

// ── Camera clocks ────────────────────────────────────────────────────
//
// A photograph is dated by the time where it was taken, not by whatever its
// camera's clock was set to (backend catalog/clock.py). A camera that records
// its clock's zone is put right from home or the folder's place; for the rest,
// and for a camera that was somewhere else than its folder says (a DJI in
// China while the Ricoh was at home), a rule says so for some days.
// Nothing is written to the files and nothing moves on disk.

const API = "/api";

function browserZone() {
  try { return Intl.DateTimeFormat().resolvedOptions().timeZone || ""; } catch { return ""; }
}

function offsetLabel(offset) {
  if (!offset) return "no zone recorded";
  if (offset === "+00:00" || offset === "Z") return "UTC";
  return `UTC${offset}`;
}

function ZoneInput({ value, onChange, placeholder, id }) {
  return (
    <input list={id} value={value} onChange={(e) => onChange(e.target.value)} placeholder={placeholder}
      spellCheck={false}
      className="w-full min-w-0 bg-neutral-950 border border-neutral-700 rounded px-2 py-1 text-base sm:text-sm text-neutral-200 placeholder:text-neutral-600" />
  );
}

function RuleForm({ camera, zonesId, onAdd, busy }) {
  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");
  const [clockZone, setClockZone] = useState("");
  const [whereZone, setWhereZone] = useState("");
  const ready = clockZone.trim() || whereZone.trim();
  return (
    <div className="mt-2 rounded border border-neutral-800 p-2 space-y-2 text-xs" data-testid="clock-rule-form">
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-neutral-500 w-24">Days</span>
        <input type="date" value={from} onChange={(e) => setFrom(e.target.value)}
          className="bg-neutral-950 border border-neutral-700 rounded px-1.5 py-0.5 text-neutral-200" />
        <span className="text-neutral-600">to</span>
        <input type="date" value={to} onChange={(e) => setTo(e.target.value)}
          className="bg-neutral-950 border border-neutral-700 rounded px-1.5 py-0.5 text-neutral-200" />
        <span className="text-neutral-600">(as its clock said; empty for always)</span>
      </div>
      <div className="flex items-center gap-2">
        <span className="text-neutral-500 w-24 shrink-0">It was in</span>
        <ZoneInput id={zonesId} value={whereZone} onChange={setWhereZone}
          placeholder="Asia/Shanghai — empty for the folder's place, or home" />
      </div>
      <div className="flex items-center gap-2">
        <span className="text-neutral-500 w-24 shrink-0">Its clock was on</span>
        <ZoneInput id={zonesId} value={clockZone} onChange={setClockZone}
          placeholder="UTC, America/Los_Angeles — empty for what its files say" />
      </div>
      <div className="flex justify-end">
        <button disabled={!ready || busy}
          onClick={() => onAdd({ camera, date_from: from, date_to: to, clock_zone: clockZone.trim(), where_zone: whereZone.trim() })}
          className="px-2.5 py-1 rounded bg-sky-800/60 text-sky-100 hover:bg-sky-700/60 disabled:opacity-40">Add rule</button>
      </div>
    </div>
  );
}

export default function ClockPanel({ onClose, onChanged, onOpenFolder }) {
  const [state, setState] = useState(null);
  const [zones, setZones] = useState([]);
  const [home, setHome] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [adding, setAdding] = useState(null);       // the camera a rule is being added for
  const [showMoved, setShowMoved] = useState(false);

  const take = useCallback((next) => {
    setState(next);
    setHome((h) => h || next.home_zone || "");
  }, []);

  const load = useCallback(async () => {
    try {
      const res = await fetch(`${API}/library/clocks`);
      if (res.ok) take(await res.json());
    } catch { setError("The backend is not reachable"); }
  }, [take]);

  useEffect(() => {
    load();
    fetch(`${API}/library/clocks/zones`).then((r) => r.json()).then((d) => setZones(d.zones || [])).catch(() => {});
  }, [load]);

  useEffect(() => {
    const key = (e) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [onClose]);

  // While the files are read, keep the numbers coming.
  const reading = state?.job?.status === "running";
  useEffect(() => {
    if (!reading) return;
    const timer = setInterval(load, 2000);
    return () => clearInterval(timer);
  }, [reading, load]);

  const send = async (method, path, body) => {
    setBusy(true);
    setError("");
    try {
      const res = await fetch(`${API}${path}`, {
        method, headers: { "Content-Type": "application/json" }, body: body ? JSON.stringify(body) : undefined,
      });
      const out = await res.json();
      if (!res.ok) { setError(out.detail || "Not saved"); return false; }
      setState(out);
      onChanged?.();
      return true;
    } catch { setError("The backend is not reachable"); return false; } finally { setBusy(false); }
  };

  const suggested = !state?.home_zone && browserZone();
  const rulesFor = (camera) => (state?.rules || []).filter((r) => r.camera === camera);

  return (
    <div className="fixed inset-0 z-[1000] flex items-start justify-center bg-black/60 p-4 overflow-y-auto" data-modal
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="w-full max-w-3xl rounded-lg border border-neutral-700 bg-neutral-900 p-4 space-y-4 text-sm" data-testid="clock-panel">
        <datalist id="clock-zones">{zones.map((z) => <option key={z} value={z} />)}</datalist>
        <div className="flex items-start gap-3">
          <div className="flex-1">
            <h2 className="text-base font-medium text-neutral-100">Camera clocks</h2>
            <p className="text-xs text-neutral-500">
              Photos are dated by the time where they were taken. A camera that records its clock's zone is put
              right on its own, from the folder's place or from home. Nothing is written to the files and nothing moves.
            </p>
          </div>
          <button onClick={onClose} className="text-neutral-500 hover:text-white">✕</button>
        </div>

        {state && !state.available && (
          <p className="text-xs text-amber-400">This computer has no time zone database: run setup again (pip install tzdata).</p>
        )}
        {reading && (
          <p className="text-xs text-neutral-400">
            Reading camera clocks: {state.job.done.toLocaleString()} of {state.job.total.toLocaleString()} photos…
          </p>
        )}
        {error && <p className="text-xs text-red-400">{error}</p>}

        <section className="space-y-1.5">
          <h3 className="text-xs uppercase tracking-wide text-neutral-500">Home</h3>
          <div className="flex items-center gap-2 flex-wrap">
            <div className="flex-1 min-w-[14rem]">
              <ZoneInput id="clock-zones" value={home} onChange={setHome} placeholder="America/Los_Angeles" />
            </div>
            <button disabled={busy || !home.trim() || home.trim() === state?.home_zone}
              onClick={() => send("PUT", "/library/clocks/home", { zone: home.trim() })}
              className="px-2.5 py-1 rounded bg-sky-800/60 text-sky-100 hover:bg-sky-700/60 disabled:opacity-40">Save</button>
            {suggested && home !== suggested && (
              <button onClick={() => setHome(suggested)} className="text-xs text-sky-300 hover:text-sky-200">
                Use this computer's: {suggested}
              </button>
            )}
          </div>
          <p className="text-xs text-neutral-600">
            Where a photo was taken when its folder has no place.
            {state?.home_source === "home" && " From the home place."}
            {!state?.home_zone && " Not set yet: nothing is put right until it is."}
          </p>
        </section>

        {state?.moved?.photos > 0 && (
          <section className="space-y-1.5" data-testid="clock-moved">
            <button onClick={() => setShowMoved((v) => !v)} className="text-left text-neutral-200 hover:text-white">
              {state.moved.photos.toLocaleString()} photos in {state.moved.folders.length} folder
              {state.moved.folders.length === 1 ? "" : "s"} were taken on another day than their camera's clock said
              <span className="text-xs text-neutral-500"> {showMoved ? "▴" : "▾"}</span>
            </button>
            {showMoved && (
              <ul className="max-h-64 overflow-y-auto space-y-0.5 text-xs">
                {state.moved.folders.map((f) => (
                  <li key={f.folder} className="flex items-baseline gap-2 flex-wrap">
                    <button onClick={() => onOpenFolder?.(f.folder)} className="font-mono text-neutral-400 hover:text-white">{f.folder}</button>
                    {f.moves.map((m) => (
                      <span key={`${m.camera}${m.from}${m.to}`} className="text-neutral-500">
                        {m.camera || "Unknown camera"}: {m.from} → <span className="text-neutral-200">{m.to}</span> ({m.photos})
                      </span>
                    ))}
                  </li>
                ))}
              </ul>
            )}
          </section>
        )}

        <section className="space-y-2">
          <h3 className="text-xs uppercase tracking-wide text-neutral-500">Cameras</h3>
          {(state?.cameras || []).map((cam) => (
            <div key={cam.camera} className="rounded border border-neutral-800 p-2.5" data-testid="clock-camera">
              <div className="flex items-baseline gap-2 flex-wrap">
                <span className="text-neutral-100">{cam.camera}</span>
                <span className="text-xs text-neutral-500">{cam.photos.toLocaleString()} files</span>
                <button onClick={() => setAdding(adding === cam.camera ? null : cam.camera)}
                  className="ml-auto text-xs text-sky-300 hover:text-sky-200">{adding === cam.camera ? "Cancel" : "+ Rule"}</button>
              </div>
              <div className="mt-1 flex flex-wrap gap-1.5 text-xs">
                {cam.recorded.map((r) => (
                  <span key={r.offset} className={`rounded px-1.5 py-0.5 ${r.offset ? "bg-neutral-800 text-neutral-300" : "text-neutral-600"}`}
                    title={`${r.first} to ${r.last}`}>
                    {offsetLabel(r.offset)} · {r.photos.toLocaleString()}
                  </span>
                ))}
              </div>
              {rulesFor(cam.camera).map((r) => (
                <div key={r.id} className="mt-1.5 flex items-center gap-2 text-xs text-neutral-300" data-testid="clock-rule">
                  <span className="text-neutral-500">{r.date_from || "always"}{r.date_to ? ` → ${r.date_to}` : r.date_from ? " on" : ""}</span>
                  {r.where_zone && <span>was in <span className="text-neutral-100">{r.where_zone}</span></span>}
                  {r.clock_zone && <span>clock on <span className="text-neutral-100">{r.clock_zone}</span></span>}
                  <button disabled={busy} onClick={() => send("DELETE", `/library/clocks/rules/${r.id}`)}
                    className="ml-auto text-neutral-600 hover:text-red-400" title="Remove this rule">✕</button>
                </div>
              ))}
              {adding === cam.camera && (
                <RuleForm camera={cam.camera} zonesId="clock-zones" busy={busy}
                  onAdd={async (rule) => { if (await send("POST", "/library/clocks/rules", rule)) setAdding(null); }} />
              )}
            </div>
          ))}
        </section>
      </div>
    </div>
  );
}
