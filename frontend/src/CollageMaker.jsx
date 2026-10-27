import { useState, useEffect, useRef } from "react";

// ── Collage ─────────────────────────────────────────────────────────
// Two to nine photos on one fixed canvas — a phone story, a square, a print —
// each cut to fill its cell around its people (backend export/collage.py). The
// layouts offered are the ones that need least cutting; the order is the
// person's. The page shows a small render of exactly what will be saved.
//
// Given more than nine (a basket, a person's timeline), the first nine go in
// and the rest wait below, to be swapped in.

const API = "/api";

const FORMATS = [
  ["story", "Story 9:16"],
  ["square", "Square"],
  ["4x6", "4×6 print"],
  ["5x7", "5×7 print"],
  ["8x10", "8×10 print"],
  ["wide", "3:2 screen"],
];
const TURNS = new Set(["4x6", "5x7", "8x10", "wide"]);
const GAPS = [["None", 0], ["Thin", 8], ["Wide", 24]];
const MARGINS = [["None", 0], ["Thin", 20], ["Wide", 60]];
const BACKGROUNDS = [["white", "#fff"], ["cream", "#faf9f6"], ["black", "#000"], ["kraft", "#c4a47a"],
  ["sky", "#d6e6f5"], ["blush", "#f6dad6"], ["sage", "#d6e2cd"], ["navy", "#1e293b"]];
// What each photo is: cut edge to edge, framed on a wall, a print (or a taped
// snapshot, a pinned card, a stamp) lying loose, or a frame on a strip of film.
const STYLES = [
  ["clean", "Clean", "Edge to edge"],
  ["rounded", "Rounded", "Soft corners"],
  ["gallery", "Gallery", "Framed and matted, as on a wall"],
  ["polaroid", "Polaroid", "Instant prints on a table"],
  ["taped", "Taped", "Snapshots held down with washi tape"],
  ["pinned", "Pinned", "Cards pinned to a board"],
  ["stamps", "Stamps", "Postage stamps"],
  ["film", "Film", "Frames on strips of film"],
];
const PRINT_STYLES = new Set(["polaroid", "taped", "pinned", "stamps"]);
// Which layouts a style offers: arrangements of loose prints, strips of film, or grids.
const family = (style) => (PRINT_STYLES.has(style) ? "prints" : style === "film" ? "film" : "grid");
// Patterns are drawn in the background colour; some have a colour they are usually seen in.
const PATTERNS = [
  ["", "None"], ["dots", "Dots"], ["stripes", "Stripes"], ["grid", "Grid paper"],
  ["linen", "Linen", "#e8e2d6"], ["wood", "Wood", "#8a5a3a"], ["cork", "Cork", "#b98f62"], ["paper", "Paper", "#efe6d2"],
];

function thumbUrl(filepath) {
  const cut = filepath.lastIndexOf("/");
  return `${API}/thumb/${filepath.slice(0, cut)}/${encodeURIComponent(filepath.slice(cut + 1))}?size=grid`;
}

// A layout as a small drawing: its cells, to choose by eye.
const PRINT_NAMES = {
  scatter: "Scatter: all one size across the page",
  feature: "Feature: the first photo large in the middle",
  pile: "Pile: tossed together",
  fan: "Fan: spread like a hand of cards, the first in the middle",
  cascade: "Cascade: overlapping down the page, the first on top",
  "film-across": "Strips across the page",
  "film-down": "Strips down the page",
};

function LayoutIcon({ layout, aspect, on, onClick }) {
  const h = 54;
  const w = Math.round(h * aspect);
  const prints = Array.isArray(layout.angles);
  return (
    <button onClick={onClick} data-testid="collage-layout"
      title={prints || PRINT_NAMES[layout.by] ? PRINT_NAMES[layout.by] || layout.by
        : (String(layout.by).startsWith("hero") ? "The first photo large · "
          : layout.by === "centre" ? "Centre: the first photo large in the middle · "
            : layout.by === "mosaic" ? "Mosaic: the first photo across a corner · " : "")
        + (layout.crop < 0.02 ? "Every photo whole" : `About ${Math.round(layout.crop * 100)}% of each photo cut away`)}
      className={`shrink-0 p-1 rounded border ${on ? "border-amber-500 bg-amber-500/10" : "border-neutral-800 hover:border-neutral-600"}`}>
      <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
        <rect width={w} height={h} fill="#262626" />
        {prints
          // Prints, turned as they will lie, in the order they are laid down.
          ? (layout.order || layout.cells.map((_, i) => i)).map((i) => {
            const [x, y, cw, ch] = layout.cells[i];
            const cx = (x + cw / 2) * w, cy = (y + ch / 2) * h;
            return (
              <rect key={i} x={x * w} y={y * h} width={Math.max(1, cw * w)} height={Math.max(1, ch * h)}
                transform={`rotate(${-(layout.angles[i] || 0)} ${cx} ${cy})`}
                fill={on ? "#fde68a" : "#d4d4d4"} stroke="#262626" strokeWidth={0.6} />
            );
          })
          : layout.cells.map(([x, y, cw, ch], i) => (
            <rect key={i} x={x * w + 0.75} y={y * h + 0.75} width={Math.max(1, cw * w - 1.5)} height={Math.max(1, ch * h - 1.5)}
              fill={on ? "#f59e0b" : "#737373"} opacity={0.85} />
          ))}
      </svg>
    </button>
  );
}

export default function CollageMaker({ filepaths, onClose, title = "Collage", start = null, initialFormat = "story" }) {
  const [order, setOrder] = useState(() => (start || filepaths).slice(0, 9));
  const pool = filepaths.filter((fp) => !order.includes(fp));
  const [format, setFormat] = useState(initialFormat);
  const [portrait, setPortrait] = useState(false);
  const [gap, setGap] = useState(8);
  const [margin, setMargin] = useState(0);
  const [background, setBackground] = useState("white");
  const [style, setStyle] = useState("clean");
  const [pattern, setPattern] = useState("");
  const [backPhoto, setBackPhoto] = useState("");   // a photo as the surface instead
  const [backBlur, setBackBlur] = useState(true);
  const [options, setOptions] = useState(null);     // {width, height, layouts}
  const [layout, setLayout] = useState(0);
  const [preview, setPreview] = useState(null);
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState("");
  const previewRef = useRef(null);

  const body = (extra = {}) => ({
    filepaths: order, format, portrait, layout, gap, margin, background, style, pattern,
    background_photo: backPhoto, background_blur: backBlur, ...extra,
  });

  // The layouts for these photos on this canvas; the first is the least cut.
  useEffect(() => {
    let live = true;
    setNote("");
    fetch(`${API}/library/collage/layouts`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body()),
    })
      .then((r) => r.json().then((d) => ({ ok: r.ok, d })))
      .then(({ ok, d }) => {
        if (!live) return;
        if (!ok) { setNote(d.detail || "No layout for these"); setOptions(null); return; }
        setOptions(d);
        setLayout(0);
      })
      .catch(() => { if (live) setNote("The backend is not reachable"); });
    return () => { live = false; };
  }, [order, format, portrait, gap, margin, family(style)]); // eslint-disable-line react-hooks/exhaustive-deps

  // A small render of exactly what Save will make, redrawn a beat after a change.
  useEffect(() => {
    if (!options) return undefined;
    let live = true;
    const timer = setTimeout(async () => {
      try {
        const res = await fetch(`${API}/library/collage`, {
          method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body({ preview: true })),
        });
        if (!res.ok || !live) return;
        const url = URL.createObjectURL(await res.blob());
        if (previewRef.current) URL.revokeObjectURL(previewRef.current);
        previewRef.current = url;
        setPreview(url);
      } catch { /* the last preview stays */ }
    }, 150);
    return () => { live = false; clearTimeout(timer); };
  }, [options, layout, background, style, pattern, backPhoto, backBlur]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => () => { if (previewRef.current) URL.revokeObjectURL(previewRef.current); }, []);

  useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape") { e.stopPropagation(); onClose(); } };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [onClose]);

  const move = (i, step) => setOrder((o) => {
    const j = i + step;
    if (j < 0 || j >= o.length) return o;
    const next = [...o];
    [next[i], next[j]] = [next[j], next[i]];
    return next;
  });
  const drop = (i) => setOrder((o) => (o.length > 2 ? o.filter((_, k) => k !== i) : o));

  const save = async () => {
    setBusy(true);
    setNote("");
    try {
      const res = await fetch(`${API}/library/collage`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body()),
      });
      if (!res.ok) { setNote((await res.json().catch(() => ({}))).detail || "Not made"); return; }
      const url = URL.createObjectURL(await res.blob());
      const a = document.createElement("a");
      const name = `Photool collage ${format}${TURNS.has(format) && portrait ? " portrait" : ""}`
        + `${style !== "clean" ? ` ${style}` : ""}.jpg`;
      a.href = url;
      a.download = name;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 10000);
      setNote(`Saved ${name} to your downloads`);
    } catch { setNote("The backend is not reachable"); } finally { setBusy(false); }
  };

  const aspect = options ? options.width / options.height : 9 / 16;
  const chip = (on) => `px-2 py-0.5 rounded border text-xs ${on
    ? "bg-neutral-700 border-neutral-500 text-white" : "border-neutral-800 text-neutral-400 hover:text-neutral-200"}`;

  return (
    <div data-modal="collage" className="fixed inset-0 z-[1000] bg-black/85 flex items-stretch sm:items-start justify-center sm:p-4 overflow-auto"
      onClick={onClose}>
      <div className="w-full max-w-5xl bg-neutral-950 sm:border border-neutral-800 sm:rounded-lg p-4 flex flex-col md:flex-row gap-4"
        onClick={(e) => e.stopPropagation()}>
        {/* The picture as it will be saved. */}
        <div className="md:w-1/2 flex items-start justify-center">
          <div className="relative w-full flex items-center justify-center bg-neutral-900 rounded"
            style={{ aspectRatio: aspect, maxHeight: "75vh", maxWidth: `calc(75vh * ${aspect})` }}>
            {preview ? <img src={preview} alt="" data-testid="collage-preview" className="w-full h-full object-contain rounded" />
              : <span className="text-sm text-neutral-500">Laying it out…</span>}
          </div>
        </div>
        <div className="md:w-1/2 space-y-4 text-sm">
          <div className="flex items-center gap-2">
            <h3 className="text-lg font-medium">{title}</h3>
            <span className="text-xs text-neutral-500">{order.length} photos{options && ` · ${options.width}×${options.height} preview`}</span>
            <button onClick={onClose} className="ml-auto text-neutral-400 hover:text-white">Close</button>
          </div>

          <section className="space-y-1.5">
            <p className="text-xs text-neutral-500">Size</p>
            <div className="flex gap-1.5 flex-wrap">
              {FORMATS.map(([k, label]) => (
                <button key={k} onClick={() => setFormat(k)} className={chip(format === k)} data-testid="collage-format">{label}</button>
              ))}
              {TURNS.has(format) && (
                <button onClick={() => setPortrait((p) => !p)} className={chip(portrait)}
                  title="Turn the page upright">{portrait ? "▯ Portrait" : "▭ Landscape"}</button>
              )}
            </div>
          </section>

          {options && (
            <section className="space-y-1.5">
              <p className="text-xs text-neutral-500">{{
                prints: "Layout — scatter, feature, pile, fan or cascade",
                film: "Layout — strips across the page, or down it",
                grid: "Layout — the first cuts least off the photos; the later ones make the first photo large",
              }[family(style)]}</p>
              <div className="flex gap-2 overflow-x-auto pb-1">
                {options.layouts.map((l, i) => (
                  <LayoutIcon key={i} layout={l} aspect={aspect} on={layout === i} onClick={() => setLayout(i)} />
                ))}
              </div>
            </section>
          )}

          <section className="space-y-1.5">
            <p className="text-xs text-neutral-500">Style</p>
            <div className="flex gap-1.5 flex-wrap">
              {STYLES.map(([k, label, hint]) => (
                <button key={k} title={hint} onClick={() => {
                  setStyle(k);
                  // Loose prints want a little table round them.
                  if (PRINT_STYLES.has(k) && margin < 20) setMargin(20);
                }} className={chip(style === k)} data-testid="collage-style">{label}</button>
              ))}
            </div>
          </section>

          <section className="flex gap-6 flex-wrap">
            {family(style) === "grid" && (
              <div className="space-y-1.5">
                <p className="text-xs text-neutral-500">Space between</p>
                <div className="flex gap-1">{GAPS.map(([label, v]) => (
                  <button key={label} onClick={() => setGap(v)} className={chip(gap === v)}>{label}</button>))}</div>
              </div>
            )}
            <div className="space-y-1.5">
              <p className="text-xs text-neutral-500">Border</p>
              <div className="flex gap-1">{MARGINS.map(([label, v]) => (
                <button key={label} onClick={() => setMargin(v)} className={chip(margin === v)}>{label}</button>))}</div>
            </div>
          </section>

          <section className="space-y-1.5" data-testid="collage-background">
            <p className="text-xs text-neutral-500">
              Background{backPhoto ? " — a photo" : pattern ? ` — ${PATTERNS.find(([k]) => k === pattern)?.[1].toLowerCase()} in this colour` : ""}
            </p>
            <div className="flex gap-1.5 flex-wrap items-center">
              {BACKGROUNDS.map(([k, color]) => (
                <button key={k} onClick={() => { setBackground(k); setBackPhoto(""); }} title={k} aria-label={k}
                  className={`w-6 h-6 rounded-full border ${background === k && !backPhoto ? "ring-2 ring-amber-500 border-transparent" : "border-neutral-600"}`}
                  style={{ background: color }} />))}
              {/* Any colour at all. */}
              <label title="Any colour" className={`relative w-6 h-6 rounded-full border overflow-hidden cursor-pointer ${
                background.startsWith("#") && !backPhoto ? "ring-2 ring-amber-500 border-transparent" : "border-neutral-600"}`}
                style={{ background: background.startsWith("#") ? background
                  : "conic-gradient(#f87171, #fbbf24, #4ade80, #60a5fa, #c084fc, #f87171)" }}>
                <input type="color" data-testid="collage-colour"
                  value={background.startsWith("#") ? background : (BACKGROUNDS.find(([k]) => k === background)?.[1] || "#ffffff").replace(/^#(.)(.)(.)$/, "#$1$1$2$2$3$3")}
                  onChange={(e) => { setBackground(e.target.value); setBackPhoto(""); }}
                  className="absolute inset-0 opacity-0 cursor-pointer" />
              </label>
            </div>
            <div className="flex gap-1 flex-wrap">
              {PATTERNS.map(([k, label, natural]) => (
                <button key={k || "none"} onClick={() => {
                  setPattern(k);
                  setBackPhoto("");
                  if (natural) setBackground(natural);
                }} className={chip(pattern === k && !backPhoto)} data-testid="collage-pattern">{label}</button>
              ))}
            </div>
            {backPhoto ? (
              <div className="flex items-center gap-2">
                <img src={thumbUrl(backPhoto)} alt="" className="w-10 h-10 object-cover rounded" />
                <span className="text-xs text-neutral-400">Photo background</span>
                <button onClick={() => setBackBlur((b) => !b)} className={chip(backBlur)}
                  title="Blurred, the photos on it stand out">Blur</button>
                <button onClick={() => setBackPhoto("")} className="text-xs text-neutral-500 hover:text-white" aria-label="No photo background">✕</button>
              </div>
            ) : (
              <p className="text-[11px] text-neutral-600">Or use one of the photos below as the background: its ▣ button.</p>
            )}
          </section>

          <section className="space-y-1.5">
            <p className="text-xs text-neutral-500">Order — the layout fills in this order</p>
            <div className="flex gap-1.5 flex-wrap">
              {order.map((fp, i) => (
                <div key={fp} className="relative group w-16" data-testid="collage-photo">
                  <img src={thumbUrl(fp)} alt="" className="w-16 h-16 object-cover rounded bg-neutral-900" />
                  <span className="absolute top-0.5 left-0.5 text-[10px] bg-black/70 text-white rounded px-1">{i + 1}</span>
                  <div className="flex justify-between text-xs text-neutral-400">
                    <button onClick={() => move(i, -1)} disabled={i === 0} className="hover:text-white disabled:opacity-30" aria-label="Earlier">◀</button>
                    <button onClick={() => setBackPhoto(fp)} data-testid="collage-as-background"
                      className={`hover:text-amber-300 ${backPhoto === fp ? "text-amber-400" : ""}`}
                      aria-label="Use as background" title="Use as the background">▣</button>
                    {order.length > 2 && <button onClick={() => drop(i)} className="hover:text-red-400" aria-label="Leave out" title="Leave out">✕</button>}
                    <button onClick={() => move(i, 1)} disabled={i === order.length - 1} className="hover:text-white disabled:opacity-30" aria-label="Later">▶</button>
                  </div>
                </div>
              ))}
            </div>
          </section>

          {pool.length > 0 && (
            <section className="space-y-1.5">
              <p className="text-xs text-neutral-500">
                Not in it ({pool.length}){order.length < 9 ? " — tap one to add it" : " — leave one out above to make room"}
              </p>
              <div className="flex gap-1.5 overflow-x-auto pb-1">
                {pool.map((fp) => (
                  <div key={fp} className="shrink-0 relative">
                    <button onClick={() => order.length < 9 && setOrder((o) => [...o, fp])} data-testid="collage-pool"
                      disabled={order.length >= 9} title={fp}
                      className="rounded overflow-hidden border border-neutral-800 hover:border-amber-600 disabled:opacity-40 disabled:hover:border-neutral-800">
                      <img src={thumbUrl(fp)} alt="" loading="lazy" className="w-14 h-14 object-cover bg-neutral-900" />
                    </button>
                    {/* A photo not in the collage makes the best background. */}
                    <button onClick={() => setBackPhoto(fp)} title="Use as the background" aria-label="Use as background"
                      className={`absolute bottom-0.5 right-0.5 text-[10px] leading-none px-0.5 rounded bg-black/70 ${
                        backPhoto === fp ? "text-amber-400" : "text-neutral-300 hover:text-amber-300"}`}>▣</button>
                  </div>
                ))}
              </div>
            </section>
          )}

          <div className="flex items-center gap-3">
            <button onClick={save} disabled={busy || !options} data-testid="collage-save"
              className="px-3 py-1.5 rounded bg-amber-600/80 hover:bg-amber-500 text-black disabled:opacity-40">
              {busy ? "Making it…" : "⤓ Save collage"}
            </button>
            {note && <span className="text-xs text-emerald-400">{note}</span>}
          </div>

        </div>
      </div>
    </div>
  );
}
