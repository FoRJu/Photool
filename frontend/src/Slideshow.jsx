import { useCallback, useEffect, useRef, useState } from "react";
import { pullFocus } from "./slideshowFocus.js";
import { recall, remember } from "./remembered.js";

const API = "/api";

// How long each picture stays, to choose from; the first is the default.
const PACES = [5, 3, 8];
// The pause after the pull back has finished, before the next picture.
const PAUSE = 0.9;

// The picture as the grid would show it, edit included (the server draws the
// edit; the hash only makes the browser ask again when the edit changes).
function slideUrl(photo) {
  const cut = photo.filepath.lastIndexOf("/");
  let v = "";
  const edit = photo.edit || {};
  const keys = Object.keys(edit).filter((k) => Number(edit[k]) !== 0).sort();
  if (keys.length) {
    let h = 5381;
    for (const ch of keys.map((k) => `${k}=${edit[k]}`).join(";")) h = ((h << 5) + h + ch.charCodeAt(0)) | 0;
    v = `&v=e${(h >>> 0).toString(36)}`;
  }
  return `${API}/thumb/${photo.filepath.slice(0, cut)}/${encodeURIComponent(photo.filepath.slice(cut + 1))}?size=preview${v}`;
}

function caption(photo) {
  const m = /^(\d{4}):(\d\d):(\d\d)/.exec(photo.capture_date || "");
  const when = m
    ? new Date(+m[1], +m[2] - 1, +m[3]).toLocaleDateString(undefined, { day: "numeric", month: "long", year: "numeric" })
    : "";
  return [when, photo.folder].filter(Boolean).join(" · ");
}

// Where the slow zoom of each picture drifts to, in turn, so one after another
// does not all move the same way.
const DRIFTS = ["center", "left top", "right bottom", "right top", "left bottom"];

// The picture's own fill behind it: the same frame, blurred and darkened, so a
// portrait on a landscape screen has a background instead of black bars — and
// it is the picture's own, so nothing of another picture shows through them.
// The fill is enlarged a little to keep the blur's soft edge off the frame.
const BACKDROP = { filter: "blur(20px) brightness(0.6)", transform: "scale(1.1)" };

// The faces of a photograph, fetched once per picture (and once per edit: the
// URL carries the edit's hash). The catalog answers with boxes as fractions of
// the finished picture; a picture not scanned, or with no face on it, still
// pulls back gently from its middle.
async function facesOf(filepath) {
  try {
    const res = await fetch(`${API}/faces/photo/${filepath}`);
    if (!res.ok) return null;
    const data = await res.json();
    return pullFocus(data.faces);
  } catch {
    return null;
  }
}

/**
 * A folder, a search or a memory, full screen, one picture at a time: each
 * fades in over the last and drifts slowly while it is shown, on a blurred,
 * darkened fill of itself, so every picture has a background and no two are
 * ever seen through one another. Pull back (Z, kept on this device) opens each
 * picture from its main person instead: the catalog's biggest face tells where
 * the calm zoom out starts and how far it opens, it holds that framing for a
 * moment, settles on the whole scene, and rests there briefly before the next.
 * Space pauses, ←/→ step, Esc leaves; on a phone, tap the left or right.
 */
export default function Slideshow({ photos, start = 0, title = "", onClose }) {
  const [index, setIndex] = useState(() => Math.min(Math.max(start, 0), photos.length - 1));
  const [playing, setPlaying] = useState(true);
  const [pace, setPace] = useState(PACES[0]);
  // Pull back from the faces rather than drift in: off until it is turned on,
  // then kept on this device.
  const [pullBack, setPullBack] = useState(() => !!recall("slideshow.pullBack"));
  // The frames on screen, oldest first: {key, url, size, drift, focus, active}.
  // The picture showing, and the one it replaced underneath while the fade runs.
  const [frames, setFrames] = useState([]);
  // The element's size, to map a face's fractions of the picture (object-contain
  // letterboxes it) into the box the zoom turns around.
  const [screen, setScreen] = useState(() => (typeof window === "undefined"
    ? { w: 1280, h: 720 }
    : { w: window.innerWidth, h: window.innerHeight }));
  const [chrome, setChrome] = useState(true);
  const hideRef = useRef(null);
  const rootRef = useRef(null);
  const seq = useRef(0);
  const faceCache = useRef(new Map());

  // The crossfade between two pictures: the long dissolve it always had.
  const fade = 900;
  // The chosen Calm move: a quarter longer than the pace, with the framing held
  // for a moment and a short pause after it before the next picture.
  const move = pace * 1.25;

  useEffect(() => {
    const onResize = () => setScreen({ w: window.innerWidth, h: window.innerHeight });
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  const go = useCallback((step) => {
    setIndex((i) => (i + step + photos.length) % photos.length);
  }, [photos.length]);

  // A picture's faces, from the cache or the catalog; never a hang, since a
  // slow answer falls back to pulling back from the middle.
  const focusFor = useCallback((photo) => {
    const key = slideUrl(photo);
    let found = faceCache.current.get(key);
    if (!found) {
      found = Promise.race([
        facesOf(photo.filepath),
        new Promise((resolve) => setTimeout(() => resolve(null), 2000)),
      ]).then((focus) => focus || pullFocus(null));
      faceCache.current.set(key, found);
    }
    return found;
  }, []);

  // The picture to show: loaded first, then laid over the one before.
  useEffect(() => {
    const photo = photos[index];
    if (!photo) return undefined;
    let live = true;
    const img = new Image();
    const url = slideUrl(photo);
    const place = async () => {
      // With Pull back on, the frame waits for the faces too, so it opens from
      // them rather than jumping to them once it is on screen.
      const focus = pullBack ? await focusFor(photo) : null;
      if (!live) return;
      const key = `s${seq.current++}`;
      const size = { w: img.naturalWidth || 0, h: img.naturalHeight || 0 };
      setFrames((fs) => [...fs.slice(-1), { key, url, size, at: index, drift: DRIFTS[index % DRIFTS.length], focus, active: false }]);
      const next = photos[(index + 1) % photos.length];
      if (next) {
        new Image().src = slideUrl(next);
        if (pullBack) focusFor(next);         // ready before it is shown
      }
    };
    img.onload = () => { place(); };
    img.onerror = () => { place(); };
    img.src = url;
    return () => { live = false; };
  }, [index, photos, pullBack, focusFor]);

  // A new frame starts transparent and is made visible one paint later, so the
  // crossfade actually runs. A frame replaced while it was still arriving
  // simply never arrives; the newest one always does.
  useEffect(() => {
    const last = frames[frames.length - 1];
    if (!last || last.active) return undefined;
    let second = 0;
    const first = requestAnimationFrame(() => {
      second = requestAnimationFrame(() => {
        setFrames((fs) => fs.map((f) => (f.key === last.key ? { ...f, active: true } : f)));
      });
    });
    return () => {
      cancelAnimationFrame(first);
      cancelAnimationFrame(second);
    };
  }, [frames]);

  // The picture the new one is laid over is completely covered once the
  // crossfade is done. Letting it go then leaves one frame to composite while
  // the slow move runs, which is what keeps a long zoom out smooth.
  useEffect(() => {
    if (frames.length < 2) return undefined;
    const top = frames[frames.length - 1];
    if (!top.active) return undefined;
    const timer = setTimeout(() => {
      setFrames((fs) => (fs.length > 1 ? fs.slice(-1) : fs));
    }, (top.focus ? 300 : fade) + 250);
    return () => clearTimeout(timer);
  }, [frames, fade]);

  // The move starts when the frame is actually on screen (with Pull back on its
  // faces may be fetched first), and the next picture waits for the move and
  // then a short pause, so the zoom out is seen whole before it goes.
  const lastFrame = frames[frames.length - 1];
  const ready = Boolean(lastFrame && lastFrame.active && lastFrame.at === index);
  useEffect(() => {
    if (!playing || photos.length < 2) return undefined;
    if (pullBack && !ready) return undefined;
    const wait = (pullBack ? move + PAUSE : pace) * 1000;
    const timer = setTimeout(() => go(1), wait);
    return () => clearTimeout(timer);
  }, [playing, pace, index, go, photos.length, pullBack, ready, move]);

  const togglePull = useCallback(() => {
    setPullBack((on) => {
      remember("slideshow.pullBack", !on);
      return !on;
    });
  }, []);

  const poke = useCallback(() => {
    setChrome(true);
    clearTimeout(hideRef.current);
    hideRef.current = setTimeout(() => setChrome(false), 2500);
  }, []);
  useEffect(() => { poke(); return () => clearTimeout(hideRef.current); }, [poke]);

  const close = useCallback(() => {
    if (document.fullscreenElement) document.exitFullscreen?.().catch(() => {});
    onClose();
  }, [onClose]);

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "Escape") close();
      else if (e.key === "ArrowRight") go(1);
      else if (e.key === "ArrowLeft") go(-1);
      else if (e.key === " ") setPlaying((p) => !p);
      else if (e.key === "z" || e.key === "Z") togglePull();
      else if (e.key === "f" || e.key === "F") {
        if (document.fullscreenElement) document.exitFullscreen?.().catch(() => {});
        else rootRef.current?.requestFullscreen?.().catch(() => {});
      } else return;
      e.preventDefault();
      e.stopPropagation();
      poke();
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [go, close, poke, togglePull]);

  // The picture's slow move while it is shown. Pull back: hold the framing on
  // the people for a moment, then dolly out to the whole scene, with an ease
  // rather than a constant speed. It is one CSS animation of `transform` only,
  // so the browser runs it on the compositor rather than React re-creating a
  // transition. Otherwise, the drift in it always had.
  const motion = (f) => {
    if (f.focus) {
      let x = f.focus.x;
      let y = f.focus.y;
      if (f.size.w > 0 && f.size.h > 0) {
        // object-contain letterboxes the picture in the element, so its
        // fractions are mapped into the element's own box.
        const inside = Math.min(screen.w / f.size.w, screen.h / f.size.h);
        const left = (screen.w - f.size.w * inside) / 2;
        const top = (screen.h - f.size.h * inside) / 2;
        x = (left + f.focus.x * f.size.w * inside) / screen.w;
        y = (top + f.focus.y * f.size.h * inside) / screen.h;
      }
      const clamp = (v) => Math.min(1, Math.max(0, v));
      return {
        "--pull-from": String(f.focus.from),
        transformOrigin: `${clamp(x) * 100}% ${clamp(y) * 100}%`,
        animation: `photool-pull ${move}s cubic-bezier(0.16, 1, 0.3, 1) both`,
        willChange: "transform",
      };
    }
    return { animation: `photool-drift ${pace + 2}s linear forwards`, transformOrigin: f.drift,
      willChange: "transform" };
  };

  const photo = photos[index];
  const tap = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    if (x < 0.25) go(-1);
    else if (x > 0.75) go(1);
    else setPlaying((p) => !p);
    poke();
  };

  return (
    <div ref={rootRef} className="fixed inset-0 z-50 bg-black select-none" data-testid="slideshow"
      onMouseMove={poke} style={{ cursor: chrome ? "default" : "none" }}>
      <style>{`
        @keyframes photool-drift { from { transform: scale(1) } to { transform: scale(1.08) } }
        @keyframes photool-pull {
          0%, 15% { transform: scale(var(--pull-from, 1.15)) }
          100% { transform: scale(1) }
        }
      `}</style>
      <div className="absolute inset-0 overflow-hidden" onClick={tap}>
        {frames.map((f) => (
          // A frame is the picture already composed on its own blurred fill, so
          // the fade between two frames never shows black bars, and no picture
          // is seen through another's letterbox.
          <div key={f.key} className="absolute inset-0 overflow-hidden bg-black"
            // A pull back is at its most open as the picture appears, so the
            // picture comes in quickly and the zoom out is what is seen.
            style={{ opacity: f.active ? 1 : 0,
              transition: `opacity ${f.focus ? 300 : fade}ms ease-out` }}>
            <img src={f.url} alt="" aria-hidden="true" draggable={false}
              className="absolute inset-0 w-full h-full object-cover"
              style={{ ...BACKDROP, pointerEvents: "none" }} />
            <img src={f.url} alt="" draggable={false} className="absolute inset-0 w-full h-full object-contain"
              style={motion(f)} />
          </div>
        ))}
      </div>
      <div className={`absolute inset-x-0 top-0 p-3 flex items-center gap-3 text-sm bg-gradient-to-b from-black/70 to-transparent transition-opacity ${chrome ? "opacity-100" : "opacity-0 pointer-events-none"}`}>
        <span className="text-neutral-200 truncate">{title}</span>
        <span className="text-neutral-500 tabular-nums">{index + 1} / {photos.length}</span>
        <span className="flex-1" />
        <button onClick={() => setPlaying((p) => !p)} className="px-2 py-0.5 rounded bg-white/10 hover:bg-white/20 text-neutral-100">
          {playing ? "❚❚ Pause" : "▶ Play"}
        </button>
        <button onClick={togglePull} aria-pressed={pullBack}
          title="Open each picture from the people in it and pull back to the whole scene (Z). Off: the slow drift in."
          className={`px-2 py-0.5 rounded border ${pullBack
            ? "bg-amber-400/20 border-amber-400/60 text-amber-200"
            : "border-transparent bg-white/10 hover:bg-white/20 text-neutral-100"}`}>
          ⤢ Pull back
        </button>
        <select value={pace} onChange={(e) => setPace(+e.target.value)}
          className="bg-black/60 border border-white/20 rounded px-1 py-0.5 text-neutral-200"
          title={pullBack ? "Seconds the pull back takes; a short pause follows before the next picture"
            : "Seconds per picture"}>
          {[...PACES].sort((a, b) => a - b).map((p) => <option key={p} value={p}>{p} s</option>)}
        </select>
        <button onClick={() => (document.fullscreenElement ? document.exitFullscreen?.() : rootRef.current?.requestFullscreen?.())?.catch?.(() => {})}
          title="Full screen (F)" className="px-2 py-0.5 rounded bg-white/10 hover:bg-white/20 text-neutral-100">⛶</button>
        <button onClick={close} title="Close (Esc)" className="px-2 py-0.5 rounded bg-white/10 hover:bg-white/20 text-neutral-100">✕</button>
      </div>
      {photo && (
        <div className={`absolute inset-x-0 bottom-0 p-3 text-xs text-neutral-300 bg-gradient-to-t from-black/70 to-transparent transition-opacity ${chrome ? "opacity-100" : "opacity-0"}`}>
          {caption(photo)}
        </div>
      )}
    </div>
  );
}
