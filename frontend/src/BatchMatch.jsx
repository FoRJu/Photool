import { useEffect, useMemo, useRef, useState } from "react";
import { clearMatchReference, lastMatchChanged, onLastMatchChange } from "./editClipboard.js";

const API = "/api";
const SAMPLES = 4;
const pad = (n) => `${n} photo${n === 1 ? "" : "s"}`;
const KINDS = [
  { id: "look", label: "Look", title: "Tone and colour, by each channel's levels. Each frame's exposure and white balance stay as they are." },
  { id: "light", label: "Light", title: "Exposure and white balance, from capture settings, shared faces and scene points. Curves and colour stay as they are." },
  { id: "both", label: "Both", title: "Light first, then the look on the frame as lit." },
];

const shootOf = (path) => path.split("/").slice(0, -1).join("/");

// Evenly through the selection, so the first frames don't speak for the whole set.
function samplesOf(filepaths) {
  if (filepaths.length <= SAMPLES) return filepaths;
  return Array.from({ length: SAMPLES }, (_, i) => filepaths[Math.round((i * (filepaths.length - 1)) / (SAMPLES - 1))]);
}

async function rendered(filepath, adjustment) {
  const res = await fetch(`${API}/export/preview`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filepath, size: 512, linear: true, adjustment }),
  });
  if (!res.ok) return null;
  return URL.createObjectURL(await res.blob());
}

/** Match the selected photos to the reference: choose what and how much, see it on a few
 *  of them, then apply to all with one undo. Nothing is saved until Apply. */
export default function BatchMatch({ reference, filepaths, onClose, onDone }) {
  const [kind, setKind] = useState("look");
  const [amount, setAmount] = useState(75);
  const [previews, setPreviews] = useState({});
  const [looking, setLooking] = useState(true);
  const [job, setJob] = useState(null);
  const [error, setError] = useState("");
  const [left, setLeft] = useState(() => new Set());
  const everyone = useMemo(() => filepaths.filter((f) => f !== reference.filepath), [filepaths, reference]);
  const targets = useMemo(() => everyone.filter((f) => !left.has(f)), [everyone, left]);
  // Which to show: the frames most likely to go wrong, or an even spread if that can't be
  // told. Nothing is drawn until it's known, since drawing a frame can take seconds.
  const [examples, setExamples] = useState(null);
  const samples = useMemo(() => (examples || []).map((e) => e.filepath), [examples]);
  const elsewhere = targets.filter((f) => shootOf(f) !== shootOf(reference.filepath)).length;
  const urls = useRef(new Set());
  const open = useRef(true);
  const body = { filepaths: targets, reference: reference.filepath, kind, amount };

  useEffect(() => () => {
    open.current = false;
    for (const url of urls.current) URL.revokeObjectURL(url);
  }, []);

  useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape" && !job) { e.stopPropagation(); onClose(); } };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [job, onClose]);

  useEffect(() => {
    let stale = false;
    fetch(`${API}/photos/edit/match/examples`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filepaths: everyone, reference: reference.filepath }),
    }).then((r) => (r.ok ? r.json() : null)).catch(() => null).then((out) => {
      if (stale) return;
      setExamples(out?.examples?.length ? out.examples
        : samplesOf(everyone).map((filepath) => ({ filepath, why: "" })));
    });
    return () => { stale = true; };
  }, [everyone, reference]);

  // The picture of each sample, before: drawn once.
  useEffect(() => {
    let stale = false;
    (async () => {
      for (const file of samples) {
        const res = await fetch(`${API}/photos/edit?filepath=${encodeURIComponent(file)}`).catch(() => null);
        const own = res && res.ok ? (await res.json()).settings : {};
        const url = await rendered(file, own);
        if (url) urls.current.add(url);
        if (stale || !open.current) return;
        setPreviews((p) => ({ ...p, [file]: { ...p[file], before: url } }));
      }
    })();
    return () => { stale = true; };
  }, [samples]);

  // And after, as the choices stand: nothing saved.
  useEffect(() => {
    if (!samples.length) return undefined;
    let stale = false;
    const timer = setTimeout(async () => {
      setLooking(true);
      setError("");
      try {
        const res = await fetch(`${API}/photos/edit/match`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ...body, filepaths: samples, dry_run: true }),
        });
        const out = await res.json();
        if (stale) return;
        if (!res.ok) { setError(out.detail || "Could not match"); return; }
        for (const file of samples) {
          const found = out.results[file];
          if (!found) continue;
          const url = await rendered(file, found.settings);
          if (url) urls.current.add(url);
          if (stale) return;
          setPreviews((p) => ({ ...p, [file]: { ...p[file], after: url, note: found.note } }));
        }
      } catch {
        if (!stale) setError("The server did not answer");
      } finally {
        if (!stale) setLooking(false);
      }
    }, 300);
    return () => { stale = true; clearTimeout(timer); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [kind, amount, samples]);

  // A running job, followed until it ends.
  useEffect(() => {
    if (!job?.id || job.status !== "running") return undefined;
    const timer = setInterval(async () => {
      try {
        const res = await fetch(`${API}/photos/edit/match/progress/${job.id}`);
        if (!res.ok) return;
        const state = await res.json();
        setJob({ id: job.id, ...state });
        if (state.status !== "running") {
          onDone(state, kind);
          lastMatchChanged();
        }
      } catch { /* try again on the next tick */ }
    }, 700);
    return () => clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job?.id, job?.status]);

  const apply = async () => {
    setError("");
    try {
      const res = await fetch(`${API}/photos/edit/match`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
      });
      const out = await res.json();
      if (!res.ok) { setError(out.detail || "Could not start"); return; }
      setJob({ id: out.job_id, status: "running", done: 0, total: out.total });
    } catch { setError("The server did not answer"); }
  };
  const running = job?.status === "running";

  return (
    <div data-modal data-testid="batch-match" className="fixed inset-0 z-[80] bg-black/80 flex items-center justify-center p-3"
      onClick={(e) => { if (e.target === e.currentTarget && !job) onClose(); }}>
      <div className="w-full max-w-4xl max-h-full overflow-y-auto bg-neutral-900 border border-neutral-700 rounded-lg p-4 space-y-3">
        <div className="flex items-start gap-3">
          <img src={`${API}/thumb/${reference.filepath.split("/").map(encodeURIComponent).join("/")}?size=grid`} alt=""
            className="w-20 h-20 object-cover rounded border border-amber-500/60" />
          <div className="min-w-0 flex-1">
            <div className="text-sm text-neutral-200">Match {pad(targets.length)} to <span className="text-amber-300">{reference.name}</span></div>
            <div className="text-xs text-neutral-500 truncate">as it is edited now · {shootOf(reference.filepath)}</div>
            {elsewhere > 0 && (
              <div className="text-xs text-amber-300/90 mt-1" data-testid="other-shoot">
                {elsewhere === targets.length ? "All of these are" : `${elsewhere} of these are`} from another shoot.
                Match copies a picture's content as well as its look, so it often works less well across shoots. Check the examples below.
              </div>
            )}
          </div>
          <button disabled={!!job} onClick={() => { clearMatchReference(); onClose(); }}
            className="text-xs text-neutral-500 hover:text-neutral-300 disabled:opacity-40">Forget reference</button>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <div className="flex gap-1" role="radiogroup" aria-label="What to match">
            {KINDS.map((k) => (
              <button key={k.id} role="radio" aria-checked={kind === k.id} disabled={!!job} title={k.title}
                data-testid={`match-kind-${k.id}`} onClick={() => setKind(k.id)}
                className={`px-3 py-1 rounded border text-sm disabled:opacity-50 ${kind === k.id
                  ? "border-amber-400 text-white bg-neutral-800" : "border-neutral-700 text-neutral-400 hover:text-white"}`}>
                {k.label}
              </button>
            ))}
          </div>
          {kind !== "light" && (
            <label className="flex items-center gap-2 text-sm text-neutral-400 flex-1 min-w-[12rem]">
              Strength
              <input type="range" min="0" max="100" step="5" value={amount} disabled={!!job} data-testid="match-amount"
                onChange={(e) => setAmount(Number(e.target.value))} className="flex-1 accent-amber-400" />
              <span className="w-9 text-right text-neutral-300">{amount}%</span>
            </label>
          )}
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3" data-testid="match-samples">
          {(examples || []).map(({ filepath: file, why }) => {
            const p = previews[file] || {};
            const out = left.has(file);
            return (
              <div key={file} className="relative" data-testid="match-example">
                {!job && (
                  <button data-testid="leave-out" onClick={() => setLeft((s) => {
                    const next = new Set(s);
                    if (next.has(file)) next.delete(file); else next.add(file);
                    return next;
                  })}
                    title={out ? "Put this photo back in the batch" : "Leave this photo out of the batch"}
                    className="absolute right-1 top-1 z-10 px-1.5 rounded bg-black/70 text-xs leading-5 text-neutral-200 hover:text-white">
                    {out ? "↩ Put back" : "✕ Leave out"}
                  </button>
                )}
                <div className={`grid grid-cols-2 gap-px bg-neutral-800 ${out ? "opacity-25" : ""}`}>
                  {["before", "after"].map((side) => (
                    <div key={side} className="relative aspect-[3/2] bg-black">
                      {p[side] && <img src={p[side]} alt="" className="w-full h-full object-contain" data-testid={`match-${side}`} />}
                      <span className="absolute left-1 top-1 text-[10px] px-1 rounded bg-black/60 text-neutral-300">{side === "before" ? "Now" : "Matched"}</span>
                    </div>
                  ))}
                </div>
                <div className="text-[11px] text-neutral-500 truncate mt-0.5" title={p.note || ""}>
                  {why && <span className="text-amber-300/80">{why} · </span>}
                  {file.split("/").pop()}{out ? " · left out" : p.note ? ` · ${p.note}` : ""}
                </div>
              </div>
            );
          })}
        </div>
        {everyone.length > samples.length && (
          <div className="text-xs text-neutral-600">The photos most likely to go wrong, of the {everyone.length} selected.</div>
        )}

        {error && <div className="text-sm text-red-400" data-testid="match-error">{error}</div>}
        <div className="flex items-center gap-2">
          {job ? (
            <div className="flex-1 text-sm text-neutral-300" data-testid="match-progress">
              {running ? `Matching… ${job.done} of ${job.total}` : `Finished ${job.done} of ${job.total}`}
              <div className="h-1 mt-1 bg-neutral-800 rounded overflow-hidden">
                <div className="h-full bg-amber-400" style={{ width: `${job.total ? (100 * job.done) / job.total : 0}%` }} />
              </div>
            </div>
          ) : (
            <div className="flex-1 text-xs text-neutral-500">
              {looking ? "Working out the examples…" : "Each photo keeps the rest of its own edit. You can take back the whole batch later from the folder view."}
            </div>
          )}
          {running ? (
            <button onClick={() => fetch(`${API}/photos/edit/match/cancel/${job.id}`, { method: "POST" })}
              className="px-3 py-1 rounded border border-neutral-700 text-neutral-300 hover:text-white text-sm">Stop</button>
          ) : job ? (
            <button onClick={onClose} className="px-3 py-1 rounded border border-neutral-700 text-neutral-300 hover:text-white text-sm">Close</button>
          ) : (
            <>
              <button onClick={onClose} className="px-3 py-1 rounded border border-neutral-700 text-neutral-300 hover:text-white text-sm">Cancel</button>
              <button onClick={apply} disabled={!targets.length} data-testid="match-apply"
                className="px-3 py-1 rounded bg-amber-500 text-black font-medium hover:bg-amber-400 text-sm disabled:opacity-40">
                Match {pad(targets.length)}
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

/** The latest match, on the folder's toolbar for as long as it can be taken back: past the
 *  selection, the folder, a reload and a restart. A photo edited since keeps that work. */
export function LastMatch({ onRestored }) {
  const [last, setLast] = useState(null);
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    const ask = () => fetch(`${API}/photos/edit/match/last`).then((r) => (r.ok ? r.json() : null))
      .then((d) => setLast(d?.last || null)).catch(() => {});
    ask();
    return onLastMatchChange(ask);
  }, []);
  const undo = async () => {
    setBusy(true);
    try {
      const res = await fetch(`${API}/photos/edit/match/undo`, { method: "POST" });
      const out = await res.json();
      if (!res.ok) { setNote(out.detail || "Could not take it back"); return; }
      for (const [fp, stored] of Object.entries(out.restored)) onRestored?.(fp, stored);
      const n = Object.keys(out.restored).length;
      const names = out.kept.map((f) => f.split("/").pop());
      setNote(`Took back ${pad(n)}`
        + (names.length ? ` · kept ${names.length} edited since: ${names.slice(0, 3).join(", ")}${names.length > 3 ? ", …" : ""}` : ""));
      setLast(null);
    } catch { setNote("The backend is not reachable"); } finally { setBusy(false); }
  };
  if (!last) return note ? <span className="text-xs text-green-300/80" data-testid="last-match-note">{note}</span> : null;
  return (
    <span className="flex items-center gap-1.5 text-xs text-neutral-400" data-testid="last-match">
      Last match: {pad(last.photos)} to {last.reference.split("/").pop()}
      <button disabled={busy} onClick={undo} data-testid="last-match-undo"
        title="Take this match back. A photo you've edited since keeps that work."
        className="px-1.5 py-0.5 rounded border border-neutral-700 text-amber-200 hover:text-white disabled:opacity-40">Undo</button>
      <button disabled={busy} onClick={() => { fetch(`${API}/photos/edit/match/last`, { method: "DELETE" }).catch(() => {}); setLast(null); }}
        title="Keep this match, and stop offering to take it back" className="text-neutral-600 hover:text-neutral-300">✕</button>
    </span>
  );
}
