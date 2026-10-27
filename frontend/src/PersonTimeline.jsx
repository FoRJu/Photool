import { useState, useEffect, useCallback, useRef } from "react";
import CollageMaker from "./CollageMaker.jsx";

// ── A person's timeline ─────────────────────────────────────────────
// Their best face for every month (or year) they were photographed, oldest
// first, and a player that steps through them with the face held in one place:
// a child growing up, frame by frame. The pick for each period is the server's
// (backend faces/timeline.py).

const API = "/api";

function portraitUrl(p, width = 1200) {
  return `${API}/faces/portrait/${p.filepath}?index=${p.face_index}&width=${width}`;
}

function photoUrl(p) {
  const cut = p.filepath.lastIndexOf("/");
  return `${API}/thumb/${p.filepath.slice(0, cut)}/${encodeURIComponent(p.filepath.slice(cut + 1))}?size=preview`;
}

// "May 2019", or "2019" for a year.
function label(p) {
  const [y, m] = p.period.split("-");
  if (!m) return y;
  return new Date(+y, +m - 1, 1).toLocaleDateString(undefined, { month: "long", year: "numeric" });
}

// Seconds per picture, to choose from; the first is the default. A timeline
// reads as a flip book, so faster than a slideshow.
const PACES = [2, 1, 3, 5];

// Up to nine periods spread evenly from the first to the last: a collage of
// someone growing up starts with the earliest and ends with the latest.
function spread(items, n = 9) {
  if (items.length <= n) return items;
  const out = [];
  for (let i = 0; i < n; i++) out.push(items[Math.round((i * (items.length - 1)) / (n - 1))]);
  return out;
}

function TimelinePlayer({ periods, start, name, onClose }) {
  const [index, setIndex] = useState(start);
  const [playing, setPlaying] = useState(true);
  const [pace, setPace] = useState(PACES[0]);
  const [faces, setFaces] = useState(true);       // aligned portraits, or the whole photos
  const [shown, setShown] = useState([]);         // [{key, url, at}], the last on top
  const rootRef = useRef(null);

  const go = useCallback((step) => {
    setIndex((i) => Math.min(Math.max(i + step, 0), periods.length - 1));
  }, [periods.length]);

  useEffect(() => {
    const p = periods[index];
    if (!p) return undefined;
    let live = true;
    const url = faces ? portraitUrl(p) : photoUrl(p);
    const img = new Image();
    const place = () => {
      if (live) setShown((s) => [...s.slice(-1), { key: `${index}-${faces}-${Date.now()}`, url, at: index }]);
    };
    img.onload = place;
    img.onerror = place;
    img.src = url;
    // The next few are fetched ahead: a portrait the server has not cut yet
    // takes longer than a flip-book frame stays up.
    for (const ahead of periods.slice(index + 1, index + 4)) new Image().src = faces ? portraitUrl(ahead) : photoUrl(ahead);
    return () => { live = false; };
  }, [index, periods, faces]);

  // The picture on screen, which is behind `index` while the next one loads;
  // the date and the clock go by it.
  const at = shown.length ? shown[shown.length - 1].at : -1;

  // Plays to the last period and stops there: a timeline has an end. Each
  // picture gets its full time once it is up, however long it took to come.
  useEffect(() => {
    if (!playing || at !== index) return undefined;
    if (index >= periods.length - 1) { setPlaying(false); return undefined; }
    const timer = setTimeout(() => go(1), pace * 1000);
    return () => clearTimeout(timer);
  }, [playing, pace, index, at, go, periods.length]);

  const close = useCallback(() => {
    if (document.fullscreenElement) document.exitFullscreen?.().catch(() => {});
    onClose();
  }, [onClose]);

  const play = useCallback(() => {
    setPlaying((on) => {
      if (!on && index >= periods.length - 1) setIndex(0);
      return !on;
    });
  }, [index, periods.length]);

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "Escape") close();
      else if (e.key === "ArrowRight") go(1);
      else if (e.key === "ArrowLeft") go(-1);
      else if (e.key === " ") play();
      else if (e.key === "f" || e.key === "F") {
        if (document.fullscreenElement) document.exitFullscreen?.().catch(() => {});
        else rootRef.current?.requestFullscreen?.().catch(() => {});
      } else return;
      e.preventDefault();
      e.stopPropagation();
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [go, close, play]);

  const p = periods[at];
  return (
    <div ref={rootRef} className="fixed inset-0 z-[1100] bg-black select-none" data-testid="timeline-player"
      onClick={(e) => e.stopPropagation()}>
      <style>{`@keyframes photool-fade { from { opacity: 0 } to { opacity: 1 } }`}</style>
      <div className="absolute inset-0" onClick={(e) => {
        const rect = e.currentTarget.getBoundingClientRect();
        const x = (e.clientX - rect.left) / rect.width;
        if (x < 0.25) go(-1); else if (x > 0.75) go(1); else play();
      }}>
        {shown.map((s) => (
          <img key={s.key} src={s.url} alt="" className="absolute inset-0 w-full h-full object-contain"
            style={{ animation: `photool-fade ${Math.min(700, pace * 400)}ms ease-out` }} />
        ))}
      </div>
      <div className="absolute inset-x-0 top-0 p-3 flex items-center gap-3 text-sm bg-gradient-to-b from-black/70 to-transparent">
        <span className="text-neutral-200 truncate">{name}</span>
        <span className="text-neutral-500 tabular-nums">{Math.max(at, 0) + 1} / {periods.length}</span>
        <span className="flex-1" />
        <div className="flex rounded border border-white/20 overflow-hidden text-xs">
          {[[true, "Faces"], [false, "Whole photos"]].map(([k, text]) => (
            <button key={text} onClick={() => setFaces(k)}
              className={`px-2 py-0.5 ${faces === k ? "bg-white/20 text-white" : "text-neutral-400 hover:text-neutral-200"}`}>{text}</button>
          ))}
        </div>
        <button onClick={play} className="px-2 py-0.5 rounded bg-white/10 hover:bg-white/20 text-neutral-100">
          {playing ? "❚❚ Pause" : "▶ Play"}
        </button>
        <select value={pace} onChange={(e) => setPace(+e.target.value)}
          className="bg-black/60 border border-white/20 rounded px-1 py-0.5 text-neutral-200" title="Seconds per picture">
          {[...PACES].sort((a, b) => a - b).map((s) => <option key={s} value={s}>{s} s</option>)}
        </select>
        <button onClick={() => (document.fullscreenElement ? document.exitFullscreen?.() : rootRef.current?.requestFullscreen?.())?.catch?.(() => {})}
          title="Full screen (F)" className="px-2 py-0.5 rounded bg-white/10 hover:bg-white/20 text-neutral-100">⛶</button>
        <button onClick={close} title="Close (Esc)" className="px-2 py-0.5 rounded bg-white/10 hover:bg-white/20 text-neutral-100">✕</button>
      </div>
      {p && (
        <div className="absolute left-0 bottom-0 p-5 pointer-events-none">
          <p className="text-3xl sm:text-4xl font-light text-white drop-shadow-lg" data-testid="timeline-date">{label(p)}</p>
          <p className="text-xs text-neutral-400 mt-1">{p.photos} photo{p.photos === 1 ? "" : "s"} · {p.filepath.split("/").slice(-2, -1)[0]}</p>
        </div>
      )}
      {/* Where in the years this picture is. */}
      <div className="absolute inset-x-0 bottom-0 h-1 bg-white/10">
        <div className="h-full bg-amber-400/80 transition-all" style={{ width: `${((Math.max(at, 0) + 1) / periods.length) * 100}%` }} />
      </div>
    </div>
  );
}

export default function PersonTimeline({ person, onClose }) {
  const [period, setPeriod] = useState("month");
  const [data, setData] = useState(null);
  const [playFrom, setPlayFrom] = useState(null);
  const [collage, setCollage] = useState(false);
  const name = person.name || "Unnamed group";

  useEffect(() => {
    let live = true;
    setData(null);
    fetch(`${API}/faces/people/${person.id}/timeline?period=${period}`)
      .then((r) => (r.ok ? r.json() : { periods: [], undated: 0 }))
      .then((d) => { if (live) setData(d); })
      .catch(() => { if (live) setData({ periods: [], undated: 0 }); });
    return () => { live = false; };
  }, [person.id, period]);

  useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape" && playFrom == null && !collage) { e.stopPropagation(); onClose(); } };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [onClose, playFrom, collage]);

  const periods = data?.periods || [];

  return (
    <div data-modal="timeline" className="fixed inset-0 z-[1050] bg-black/85 flex items-stretch sm:items-start justify-center sm:p-4 overflow-auto"
      onClick={(e) => { e.stopPropagation(); onClose(); }}>
      <div className="w-full max-w-6xl bg-neutral-950 sm:border border-neutral-800 sm:rounded-lg flex flex-col min-h-full sm:min-h-0"
        onClick={(e) => e.stopPropagation()}>
        <div className="sticky top-0 z-10 bg-neutral-950/95 backdrop-blur border-b border-neutral-800 px-4 py-3 flex items-center gap-3 flex-wrap">
          <h3 className="text-lg font-medium">{name} over the years</h3>
          <span className="text-xs text-neutral-500">
            {data ? `${periods.length} ${period}${periods.length === 1 ? "" : "s"}` : "Loading…"}
            {periods.length > 1 && ` · ${label(periods[0])} – ${label(periods[periods.length - 1])}`}
            {data?.undated > 0 && ` · ${data.undated} undated photo${data.undated === 1 ? "" : "s"} left out`}
          </span>
          <div className="ml-auto flex items-center gap-2">
            <div className="flex rounded border border-neutral-800 overflow-hidden text-xs">
              {[["month", "By month"], ["year", "By year"]].map(([k, text]) => (
                <button key={k} onClick={() => setPeriod(k)}
                  className={`px-2 py-1 ${period === k ? "bg-neutral-800 text-white" : "text-neutral-500 hover:text-neutral-300"}`}>{text}</button>
              ))}
            </div>
            <button onClick={() => setCollage(true)} disabled={periods.length < 2} data-testid="timeline-collage"
              title="A collage of them growing up: up to nine, spread from the first to the latest (swap any in)"
              className="px-2.5 py-1 rounded border border-neutral-700 text-sm text-neutral-200 hover:border-amber-600 disabled:opacity-40">▦ Collage</button>
            <button onClick={() => setPlayFrom(0)} disabled={!periods.length} data-testid="timeline-play"
              className="px-2.5 py-1 rounded bg-amber-600/80 hover:bg-amber-500 text-black text-sm disabled:opacity-40">▶ Play</button>
            <button onClick={onClose} className="px-2 py-1 text-sm text-neutral-400 hover:text-white">Close</button>
          </div>
        </div>
        <div className="p-3">
          <div className="grid gap-2" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(min(160px, 42vw), 1fr))" }}>
            {periods.map((p, i) => (
              <button key={p.period} onClick={() => setPlayFrom(i)} title={`Play from ${label(p)}`}
                className="text-left group" data-testid="timeline-tile">
                <img src={portraitUrl(p, 360)} loading="lazy" alt=""
                  className="w-full aspect-[3/2] object-cover rounded bg-neutral-900 border border-transparent group-hover:border-neutral-600" />
                <p className={`mt-0.5 text-xs ${i === 0 || periods[i - 1].period.slice(0, 4) !== p.period.slice(0, 4)
                  ? "text-amber-300" : "text-neutral-300"}`}>{label(p)}</p>
                <p className="text-[10px] text-neutral-600">{p.photos} photo{p.photos === 1 ? "" : "s"}</p>
              </button>
            ))}
          </div>
          {data && !periods.length && <p className="text-sm text-neutral-500">No dated photos of {name} yet.</p>}
        </div>
      </div>
      {collage && (
        <div onClick={(e) => e.stopPropagation()}>
          <CollageMaker title={`${name} over the years`} filepaths={periods.map((p) => p.filepath)}
            start={spread(periods).map((p) => p.filepath)} initialFormat="square" onClose={() => setCollage(false)} />
        </div>
      )}
      {playFrom != null && periods.length > 0 && (
        <TimelinePlayer periods={periods} start={playFrom} name={name} onClose={() => setPlayFrom(null)} />
      )}
    </div>
  );
}
