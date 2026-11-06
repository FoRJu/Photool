import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import Slideshow from "./Slideshow.jsx";
import { recall, remember } from "./remembered.js";

// ── Exploring the library ───────────────────────────────────────────
// Ways in besides a folder and a day (backend catalog/explore.py):
//   Rediscover   good frames not looked at in a long while, and a day at random
//   Looks like   scenes the library is full of — birthdays, the beach, snow —
//                the closest matches by smart search, each a year at a time.
//                Smart search measures resemblance, not fact: no counts are
//                claimed, and "Not this" teaches a shelf what does not belong.
//   Timeline     the whole library by year and month, with a scrubber
// Every picture is drawn as the library shows it: a photo whose cover is one of
// its virtual copies shows that copy.

const API = "/api";
const MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July",
  "August", "September", "October", "November", "December"];
const MONTH_SHORT = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

const plural = (n, word) => `${(n || 0).toLocaleString()} ${word}${n === 1 ? "" : "s"}`;

// The same key the grid uses: it changes when the edit (or the cover) does, so
// the browser asks again instead of showing what it kept.
function editKey(p) {
  const cover = (p && p.cover) || 0;
  const edit = cover ? p.cover_edit : p && p.edit;
  const keys = edit ? Object.keys(edit).filter((k) => Number(edit[k]) !== 0).sort() : [];
  // A photo with copies always says which version it shows: the browser keeps
  // pictures by address, and the main edit and a copy must never share one.
  const which = cover || (p && p.copies) ? `c${cover}` : "";
  if (!keys.length) return which;
  let h = 5381;
  for (const ch of keys.map((k) => `${k}=${JSON.stringify(edit[k])}`).join(";")) {
    h = ((h << 5) + h + ch.charCodeAt(0)) | 0;
  }
  return `${which}e${(h >>> 0).toString(36)}`;
}

export function photoThumb(p, size = "grid") {
  if (!p) return "";
  const cut = p.filepath.lastIndexOf("/");
  const key = editKey(p);
  return `${API}/thumb/${p.filepath.slice(0, cut)}/${encodeURIComponent(p.filepath.slice(cut + 1))}`
    + `?size=${size}${key ? `&v=${key}` : ""}`;
}

function dayText(day, opts = { day: "numeric", month: "short", year: "numeric" }) {
  if (!day) return "";
  const [y, m, d] = day.split("-").map(Number);
  return new Date(y, m - 1, d).toLocaleDateString(undefined, opts);
}

function agoText(day) {
  if (!day) return "";
  const years = new Date().getFullYear() - Number(day.slice(0, 4));
  return years <= 0 ? "This year" : years === 1 ? "A year ago" : `${years} years ago`;
}

function useJson(url, keep = null) {
  const [data, setData] = useState(() => (keep ? recall(keep) : null));
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    if (!url) return undefined;
    let live = true;
    setFailed(false);
    fetch(url)
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then((d) => {
        if (!live) return;
        setData(d);
        if (keep) remember(keep, d);
      })
      .catch(() => { if (live) setFailed(true); });
    return () => { live = false; };
  }, [url, keep]);
  return [data, failed];
}

function Strip({ photos, onPick, label = null, height = "h-24 sm:h-28" }) {
  return (
    <div className="flex gap-1.5 overflow-x-auto pb-1">
      {photos.map((p, i) => (
        <button key={`${p.filepath}#${i}`} onClick={() => onPick(i)}
          className="relative shrink-0 rounded overflow-hidden border border-neutral-800 hover:border-neutral-500">
          <img src={photoThumb(p)} alt="" loading="lazy" className={`${height} w-auto object-cover bg-neutral-900`} />
          {label && (
            <span className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/80 to-transparent text-[10px] text-neutral-200 px-1.5 pb-1 pt-3 text-left">
              {label(p)}
            </span>
          )}
          {p.cover > 0 && (
            <span className="absolute top-1 right-1 px-1 rounded bg-amber-500/90 text-black text-[9px] leading-tight"
              title="A virtual copy, chosen as the version shown">★</span>
          )}
        </button>
      ))}
    </div>
  );
}

// ── Rediscover ──────────────────────────────────────────────────────

export function Rediscover({ onPlay, onOpen }) {
  // The same all day; Shuffle deals again (and is kept for the day on this device).
  const today = new Date().toLocaleDateString("en-CA");
  const [shuffle, setShuffle] = useState(() => {
    const kept = recall("rediscover.shuffle");
    return kept?.day === today ? kept.n : 0;
  });
  const [data] = useJson(`${API}/library/rediscover${shuffle ? `?shuffle=${shuffle}` : ""}`);
  const again = () => {
    const n = shuffle + 1;
    setShuffle(n);
    remember("rediscover.shuffle", { day: today, n });
  };
  if (!data || (!data.photos.length && !data.day)) return null;
  const day = data.day;
  const dayTitle = day ? `${day.ago === 1 ? "A year ago" : `${day.ago} years ago`} · ${dayText(day.date, { weekday: "long", day: "numeric", month: "long", year: "numeric" })}` : "";
  return (
    <section className="space-y-2" data-testid="rediscover">
      <div className="flex items-baseline gap-3 flex-wrap">
        <h2 className="text-sm font-medium text-neutral-200">Rediscover</h2>
        <span className="text-xs text-neutral-500">Good photos nobody has opened in a long while</span>
        {data.photos.length > 1 && (
          <button onClick={() => onPlay(data.photos, 0, "Rediscover")}
            className="text-xs text-sky-300 hover:text-sky-200">▶ Play</button>
        )}
        <button onClick={again} className="text-xs text-neutral-400 hover:text-white" data-testid="rediscover-shuffle"
          title="Deal another set">↻ Shuffle</button>
      </div>
      {data.photos.length > 0 && (
        <Strip photos={data.photos} onPick={(i) => onPlay(data.photos, i, "Rediscover")}
          label={(p) => agoText(p.date)} />
      )}
      {day && (
        <div className="space-y-1">
          <div className="flex items-baseline gap-2 text-xs flex-wrap">
            <span className="text-neutral-400">A day at random:</span>
            <span className="text-neutral-200">{dayTitle}</span>
            <span className="text-neutral-600">· {plural(day.total, "photo")}</span>
            <button onClick={() => onPlay(day.photos, 0, dayTitle)} className="text-sky-300 hover:text-sky-200">▶ Play the day</button>
            <button onClick={() => onOpen(day.folder)} className="text-neutral-500 hover:text-neutral-300 underline decoration-neutral-700">
              Open the folder
            </button>
          </div>
          <Strip photos={day.photos.slice(0, 10)} onPick={(i) => onPlay(day.photos, i, dayTitle)} height="h-16 sm:h-20" />
        </div>
      )}
    </section>
  );
}

// ── Collections ─────────────────────────────────────────────────────

export function Collections({ onOpen, onSearch }) {
  const [data] = useJson(`${API}/library/collections`, "collections");
  const [open, setOpen] = useState(null);
  if (!data) return null;
  if (!data.available) {
    return (
      <p className="text-xs text-neutral-600" data-testid="collections-off">
        Looks like — birthdays, the beach, snow, every year — appears here once smart search is on
        (✦ Describe, beside the search box).
      </p>
    );
  }
  if (!data.collections.length) return null;
  return (
    <section className="space-y-2" data-testid="collections">
      <div className="flex items-baseline gap-3">
        <h2 className="text-sm font-medium text-neutral-200">Looks like</h2>
        <span className="text-xs text-neutral-500">The closest matches by what the pictures show. A guess, not a label.</span>
      </div>
      <div className="flex gap-2 overflow-x-auto pb-1">
        {data.collections.map((c) => (
          <button key={c.key} onClick={() => setOpen(c)} data-testid={`collection-${c.key}`}
            className="shrink-0 w-36 sm:w-40 text-left rounded-lg overflow-hidden border border-neutral-800 hover:border-neutral-500 bg-neutral-900/50">
            <div className="grid grid-cols-2 gap-px bg-neutral-950 aspect-square">
              {[0, 1, 2, 3].map((i) => (
                c.covers[i]
                  ? <img key={i} src={photoThumb(c.covers[i])} alt="" loading="lazy" className="w-full h-full object-cover" />
                  : <div key={i} className="bg-neutral-900" />
              ))}
            </div>
            <div className="px-2 py-1.5">
              <div className="text-sm text-neutral-100 truncate">{c.title}</div>
              <div className="text-[11px] text-neutral-500">Across {plural(c.years, "year")}</div>
            </div>
          </button>
        ))}
      </div>
      {open && (
        <CollectionPage entry={open} onClose={() => setOpen(null)}
          onOpenFolder={(f) => { setOpen(null); onOpen(f); }}
          onSearch={onSearch && ((q) => { setOpen(null); onSearch(q); })} />
      )}
    </section>
  );
}

// Esc closes the page, unless something over it (a slideshow) has the key.
function useEscape(onClose, active = true) {
  useEffect(() => {
    if (!active) return undefined;
    const key = (e) => { if (e.key === "Escape") { e.stopPropagation(); onClose(); } };
    window.addEventListener("keydown", key, true);
    return () => window.removeEventListener("keydown", key, true);
  }, [onClose, active]);
}

function CollectionPage({ entry, onClose, onOpenFolder, onSearch }) {
  const [loaded, failed] = useJson(`${API}/library/collections/${entry.key}`);
  // Photos said not to belong go at once; the server keeps that for next time.
  const [gone, setGone] = useState(() => new Set());
  const notThis = (p) => {
    setGone((g) => new Set(g).add(p.filepath));
    fetch(`${API}/library/collections/${entry.key}/hide`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filepath: p.filepath }),
    }).catch(() => {});
  };
  const data = useMemo(() => (loaded ? {
    ...loaded,
    years: loaded.years.map((y) => ({ ...y, photos: y.photos.filter((p) => !gone.has(p.filepath)) }))
      .filter((y) => y.photos.length),
  } : null), [loaded, gone]);
  const [slides, setSlides] = useState(null);       // { photos, start, title }
  const closeSlides = useCallback(() => setSlides(null), []);
  useEscape(onClose, !slides);
  const all = useMemo(() => (data ? data.years.flatMap((y) => y.photos) : []), [data]);
  return (
    <div data-modal="collection" className="fixed inset-0 z-[1000] bg-black/85 flex items-stretch sm:items-start justify-center sm:p-4 overflow-auto"
      onClick={onClose} data-testid="collection-page">
      <div className="bg-neutral-950 sm:border border-neutral-800 sm:rounded-lg w-full max-w-6xl p-4 space-y-4 min-h-full sm:min-h-0"
        onClick={(e) => e.stopPropagation()}>
        <div className="flex items-baseline gap-3 flex-wrap">
          <h2 className="text-lg text-neutral-100">{entry.title}</h2>
          {data && <span className="text-sm text-neutral-500">Closest matches over {plural(data.years.length, "year")}</span>}
          {all.length > 1 && (
            <button onClick={() => setSlides({ photos: all, start: 0, title: entry.title })}
              className="text-sm text-sky-300 hover:text-sky-200">▶ Play the best of every year</button>
          )}
          {onSearch && (
            <button onClick={() => onSearch(entry.query)} className="text-sm text-neutral-400 hover:text-white"
              title={`Everything that looks like this, as a search: ${entry.query}`}>See more like this in the grid</button>
          )}
          <button onClick={onClose} className="ml-auto text-neutral-400 hover:text-white text-lg leading-none" aria-label="Close">✕</button>
        </div>
        {data && data.years.length > 3 && (
          <div className="flex flex-wrap gap-1">
            {data.years.map((y) => (
              <a key={y.year} href={`#coll-${y.year}`}
                onClick={(e) => { e.preventDefault(); document.getElementById(`coll-${y.year}`)?.scrollIntoView({ behavior: "smooth", block: "start" }); }}
                className="px-2 py-0.5 rounded border border-neutral-800 text-xs text-neutral-300 hover:border-neutral-500">
                {y.year || "Undated"}
              </a>
            ))}
          </div>
        )}
        {failed && <p className="text-sm text-red-400">This collection could not be read.</p>}
        {!data && !failed && <p className="text-sm text-neutral-500">Looking through the library…</p>}
        {data && data.years.map((y) => {
          const title = `${entry.title} · ${y.year || "Undated"}`;
          return (
            <section key={y.year} id={`coll-${y.year}`} className="space-y-1.5 scroll-mt-4">
              <div className="flex items-baseline gap-2 text-sm">
                <span className="text-neutral-100">{y.year || "Undated"}</span>
                                <button onClick={() => setSlides({ photos: y.photos, start: 0, title })} className="text-xs text-sky-300 hover:text-sky-200">▶ Play</button>
              </div>
              <div className="grid gap-1.5" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(min(150px, 30vw), 1fr))" }}>
                {y.photos.map((p, i) => (
                  <div key={p.filepath} className="relative group">
                    <button onClick={() => setSlides({ photos: y.photos, start: i, title })}
                      className="block w-full aspect-square rounded overflow-hidden border border-neutral-800 hover:border-neutral-500">
                      <img src={photoThumb(p)} alt="" loading="lazy" className="w-full h-full object-cover bg-neutral-900" />
                    </button>
                    <button onClick={() => notThis(p)} data-testid="collection-not-this"
                      title={`Not ${entry.title.toLowerCase()}: take it out of this shelf`}
                      className="absolute top-1 right-1 text-[10px] text-neutral-200 bg-black/70 hover:bg-red-900/80 rounded px-1.5 py-0.5 opacity-0 group-hover:opacity-100 focus:opacity-100">
                      Not this</button>
                    <button onClick={() => onOpenFolder(p.folder)}
                      className="absolute bottom-1 left-1 right-1 truncate text-left text-[10px] text-neutral-200 bg-black/60 rounded px-1 opacity-0 group-hover:opacity-100"
                      title={`Open ${p.folder}`}>{dayText(p.date) || p.folder}</button>
                  </div>
                ))}
              </div>
            </section>
          );
        })}
      </div>
      {slides && (
        <div onClick={(e) => e.stopPropagation()}>
          <Slideshow photos={slides.photos} start={slides.start} title={slides.title} onClose={closeSlides} />
        </div>
      )}
    </div>
  );
}

// ── The timeline ────────────────────────────────────────────────────
// Decades scrolled, not days paged. Two zooms: a year at a time (its best
// frames, and a bar per month to jump in) and a month at a time (its best few
// and its shoots, each one click from its folder). The scrubber at the side
// goes straight to any year, by tap or by dragging along it.

const ZOOM_KEY = "photool.timeline.zoom";

function Scrubber({ years, current, onGo }) {
  const box = useRef(null);
  const dragging = useRef(false);
  const at = (clientY) => {
    const r = box.current?.getBoundingClientRect();
    if (!r || !years.length) return null;
    const f = Math.min(0.9999, Math.max(0, (clientY - r.top) / r.height));
    return years[Math.floor(f * years.length)];
  };
  // Many years: only every fifth (and the ends) is labelled, the rest are ticks.
  const dense = years.length > 24;
  return (
    <div ref={box} data-testid="timeline-scrubber"
      className="fixed right-0 top-16 bottom-24 sm:bottom-8 z-20 w-9 sm:w-11 flex flex-col justify-between py-1 select-none touch-none
        rounded-l-lg bg-neutral-950/80 backdrop-blur border border-r-0 border-neutral-800 cursor-ns-resize"
      onPointerDown={(e) => { dragging.current = true; e.currentTarget.setPointerCapture?.(e.pointerId); const y = at(e.clientY); if (y) onGo(y); }}
      onPointerMove={(e) => { if (dragging.current) { const y = at(e.clientY); if (y) onGo(y); } }}
      onPointerUp={() => { dragging.current = false; }}
      onPointerCancel={() => { dragging.current = false; }}>
      {years.map((y, i) => {
        const labelled = !dense || y === current || i === 0 || i === years.length - 1 || y % 5 === 0;
        return (
          <div key={y} className={`text-center leading-none ${y === current ? "text-amber-300 font-semibold" : "text-neutral-500"}`}
            style={{ fontSize: labelled ? "10px" : "6px" }}>
            {labelled ? (y === current || !dense ? y : `'${String(y).slice(2)}`) : "·"}
          </div>
        );
      })}
    </div>
  );
}

function MonthBar({ year, onMonth }) {
  const byMonth = new Map(year.months.map((m) => [Number(m.month.slice(5, 7)) - 1, m]));
  const most = Math.max(1, ...year.months.map((m) => m.total));
  return (
    <div className="grid grid-cols-12 gap-0.5 items-end h-10 max-w-md">
      {MONTH_SHORT.map((name, i) => {
        const m = byMonth.get(i);
        return (
          <button key={name} disabled={!m} onClick={() => m && onMonth(m.month)}
            title={m ? `${MONTH_NAMES[i]}: ${plural(m.total, "photo")}` : MONTH_NAMES[i]}
            className="flex flex-col items-center justify-end h-full group disabled:cursor-default">
            <span className={`w-full rounded-sm ${m ? "bg-sky-700 group-hover:bg-sky-500" : "bg-neutral-900"}`}
              style={{ height: m ? `${Math.max(12, (m.total / most) * 100)}%` : "8%" }} />
            <span className="text-[8px] text-neutral-600 mt-0.5">{name[0]}</span>
          </button>
        );
      })}
    </div>
  );
}

export function Timeline({ onOpen, onPlay, titleOf = (f) => f }) {
  const [data, failed] = useJson(`${API}/library/timeline`);
  const [zoom, setZoom] = useState(() => {
    try { return localStorage.getItem(ZOOM_KEY) || "years"; } catch { return "years"; }
  });
  const [current, setCurrent] = useState(null);
  // The scrubber shows once the timeline reaches the top of the screen, so it
  // never sits over the page's own controls above it.
  const [reached, setReached] = useState(false);
  const rootRef = useRef(null);
  const pending = useRef(null);          // where to scroll once a zoom has drawn
  const chooseZoom = (z) => {
    setZoom(z);
    try { localStorage.setItem(ZOOM_KEY, z); } catch { /* this visit only */ }
  };
  const years = useMemo(() => (data ? data.years : []), [data]);
  const yearList = useMemo(() => years.map((y) => y.year), [years]);

  const go = useCallback((id) => {
    document.getElementById(id)?.scrollIntoView({ block: "start" });
  }, []);
  const goYear = useCallback((year) => { setCurrent(year); go(`tl-${year}`); }, [go]);
  const openMonth = (month) => { pending.current = `tl-${month}`; chooseZoom("months"); };
  useEffect(() => {
    if (pending.current) {
      const id = pending.current;
      pending.current = null;
      requestAnimationFrame(() => go(id));
    }
  }, [zoom, go]);

  // The year at the top of the screen, for the scrubber to light.
  useEffect(() => {
    if (!years.length) return undefined;
    let frame = 0;
    const look = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        let seen = years[0].year;
        for (const y of years) {
          const el = document.getElementById(`tl-${y.year}`);
          if (el && el.getBoundingClientRect().top <= 120) seen = y.year;
        }
        setCurrent(seen);
        const top = rootRef.current?.getBoundingClientRect().top;
        setReached(top !== undefined && top < 160);
      });
    };
    look();
    window.addEventListener("scroll", look, { passive: true });
    return () => { window.removeEventListener("scroll", look); cancelAnimationFrame(frame); };
  }, [years, zoom]);

  if (failed) return <p className="text-sm text-red-400">The timeline could not be read.</p>;
  if (!data) return <p className="text-sm text-neutral-500">Laying out the years…</p>;
  if (!years.length) return <p className="text-sm text-neutral-500">No dated photos yet.</p>;

  return (
    <div ref={rootRef} className="pr-10 sm:pr-12 space-y-6" data-testid="timeline">
      <div className="flex items-center gap-2 text-sm">
        <span className="text-neutral-500">Show</span>
        <div className="flex rounded border border-neutral-800 overflow-hidden">
          {[["years", "Years"], ["months", "Months"]].map(([k, label]) => (
            <button key={k} onClick={() => chooseZoom(k)} data-testid={`timeline-zoom-${k}`}
              className={`px-3 py-1 ${zoom === k ? "bg-neutral-800 text-white" : "text-neutral-500 hover:text-neutral-300"}`}>{label}</button>
          ))}
        </div>
        <span className="text-xs text-neutral-600 hidden sm:inline">
          {zoom === "years" ? "Each year's best, one a day. A month's bar opens that month." : "Each month's best few, and its shoots."}
        </span>
      </div>

      {years.map((y) => (
        <section key={y.year} id={`tl-${y.year}`} className="space-y-3 scroll-mt-2">
          <h3 className="sticky top-0 z-10 bg-neutral-950/90 backdrop-blur py-1 flex items-baseline gap-3">
            <span className="text-lg text-neutral-100">{y.year}</span>
            <span className="text-xs text-neutral-500">{plural(y.total, "photo")} · {plural(y.months.length, "month")}</span>
            {y.photos.length > 1 && (
              <button onClick={() => onPlay(y.photos, 0, String(y.year))} className="text-xs text-sky-300 hover:text-sky-200">▶ Play</button>
            )}
          </h3>
          {zoom === "years" ? (
            <div className="space-y-2">
              <Strip photos={y.photos} onPick={(i) => onPlay(y.photos, i, String(y.year))}
                label={(p) => dayText(p.date, { day: "numeric", month: "short" })} />
              <MonthBar year={y} onMonth={openMonth} />
            </div>
          ) : (
            [...y.months].map((m) => {
              const title = `${MONTH_NAMES[Number(m.month.slice(5, 7)) - 1]} ${m.month.slice(0, 4)}`;
              return (
                <div key={m.month} id={`tl-${m.month}`} className="space-y-1.5 scroll-mt-10 pl-2 border-l border-neutral-800">
                  <div className="flex items-baseline gap-2 text-sm">
                    <span className="text-neutral-200">{title}</span>
                    <span className="text-xs text-neutral-500">{plural(m.total, "photo")} · {plural(m.days, "day")}</span>
                    {m.photos.length > 1 && (
                      <button onClick={() => onPlay(m.photos, 0, title)} className="text-xs text-sky-300 hover:text-sky-200">▶ Play</button>
                    )}
                  </div>
                  <Strip photos={m.photos} onPick={(i) => onPlay(m.photos, i, title)} height="h-20 sm:h-24" />
                  {m.shoots.length > 0 && (
                    <div className="flex flex-wrap gap-1.5">
                      {m.shoots.map((s) => (
                        <button key={s.folder} onClick={() => onOpen(s.folder)} title={`Open ${s.folder}`}
                          className="flex items-center gap-2 pr-2 rounded border border-neutral-800 hover:border-neutral-500 bg-neutral-900/50 max-w-[16rem]">
                          <img src={photoThumb(s.cover)} alt="" loading="lazy" className="w-10 h-10 object-cover rounded-l" />
                          <span className="text-xs text-neutral-300 truncate">{titleOf(s.folder)}</span>
                          <span className="text-[10px] text-neutral-600 shrink-0">{s.total}</span>
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              );
            })
          )}
        </section>
      ))}
      {reached && <Scrubber years={yearList} current={current} onGo={goYear} />}
    </div>
  );
}
