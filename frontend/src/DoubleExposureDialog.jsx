import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { editKey, makeDoubleExposure, sameEdit, secondOf, versionThumb, versionsOf } from "./doubleExposure.js";

// ── Two photos into one ─────────────────────────────────────────────
// The pairing window. The grid opens it on two selected photos, to make a double exposure;
// Develop opens it again on one already made, to rethink it.
//
// The two photos side by side, each with its versions as thumbnails, the one the grid
// shows (its cover) chosen to start with; the second also as shot, its raw light toned
// with the first. ⇄ swaps which one keeps the copy. Below, the pair is drawn four ways, one
// per blend, and Shake it up lays the second on differently in each. A blend clicked comes
// up large, to be placed by hand: drag it, scroll or pinch to enlarge it, mirror it. The
// feel is set here; the fine numbers (amount, exposure, where it shows) are Develop's.

const API = "/api";
const TILE = 420;
const STAGE = 1100;
// The copy's own edit, as the base when the window is opened on one.
const COPY = "copy";

const BLENDS = [
  { mode: "add", label: "Add", hint: "The two lights added, as film does. Bright where either is." },
  { mode: "average", label: "Average", hint: "Added and brought back down: the brightness of one photo." },
  { mode: "lighten", label: "Bright", hint: "The brighter of the two at each point. A dark sky takes the other photo." },
  { mode: "darken", label: "Dark", hint: "The darker of the two at each point. A bright sky takes the other photo." },
];

const STILL = { scale: 100, x: 0, y: 0, flip: false };
// How it is laid on, besides the placement and the blend: Develop's numbers, kept as they are.
const HOW = { amount: 100, exposure: 0, reach: "", invert: false };

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

function shaken() {
  const scale = Math.round(100 + Math.random() * 80);
  // Moved no further than the enlargement allows, so the frame stays covered.
  const room = (scale - 100) / 2;
  const move = () => Math.round((Math.random() * 2 - 1) * room);
  return { scale, x: move(), y: move(), flip: Math.random() < 0.5 };
}

const withoutSecond = (settings) => {
  const { second, ...rest } = settings || {};
  return rest;
};

// Pictures drawn by the server, kept for the window's life by what was asked.
function useRenders() {
  const [, setDrawn] = useState(0);
  const kept = useRef(new Map());
  useEffect(() => () => kept.current.forEach((u) => u && u !== "…" && URL.revokeObjectURL(u)), []);
  return useCallback((body) => {
    if (!body) return null;
    const key = JSON.stringify(body);
    const have = kept.current.get(key);
    if (have !== undefined) return have === "…" ? null : have;
    kept.current.set(key, "…");
    fetch(`${API}/export/preview`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: key,
    })
      .then((r) => (r.ok ? r.blob() : null))
      .then((blob) => {
        if (!blob) { kept.current.delete(key); return; }
        kept.current.set(key, URL.createObjectURL(blob));
        setDrawn((n) => n + 1);
      })
      .catch(() => kept.current.delete(key));
    return null;
  }, []);
}

function Tile({ src, label, picked, star, onClick, testid, title }) {
  return (
    <button onClick={onClick} data-testid={testid} title={title}
      className={`relative shrink-0 w-16 rounded overflow-hidden border text-left ${picked ? "border-amber-400 ring-1 ring-amber-400" : "border-neutral-800 hover:border-neutral-500"}`}>
      <img src={src} alt="" loading="lazy" className="w-16 h-16 object-cover bg-neutral-900" />
      <div className="px-1 py-0.5 text-[10px] leading-tight text-neutral-300 truncate bg-neutral-950">
        {star && <span className="text-amber-300" title="The version the grid shows">★ </span>}{label}
      </div>
    </button>
  );
}

function Side({ title, photo, info, chosen, onChoose, asShot, copy, skip, stamp, testid }) {
  // A double exposure is not offered as either half: its own second would be lost.
  const versions = (info?.versions || []).filter((v) => v.version !== skip && !v.settings.second);
  const current = typeof chosen === "number" ? versions.find((v) => v.version === chosen) : null;
  const big = chosen === null
    ? `${versionThumb(photo.filepath, 0, "preview")}&camera=1`
    : chosen === COPY ? copy?.src
      : versionThumb(photo.filepath, chosen, "preview", current ? `${editKey(current.settings)}.${stamp}` : stamp);
  return (
    <div className="min-w-0 flex flex-col gap-2" data-testid={testid}>
      <div className="text-[11px] uppercase tracking-wide text-neutral-500">{title}</div>
      <div className="h-32 sm:h-44 rounded overflow-hidden bg-neutral-900 border border-neutral-800">
        {big && <img src={big} alt={photo.filename} className="w-full h-full object-contain" />}
      </div>
      <div className="text-xs text-neutral-300 truncate" title={photo.filepath}>{photo.filename}</div>
      <div className="flex gap-1.5 overflow-x-auto pb-1">
        {copy && (
          <Tile src={copy.src} label={copy.label} picked={chosen === COPY} onClick={() => onChoose(COPY)} testid="dx-this-copy"
            title="This copy's own edit, as it is now." />
        )}
        {asShot && (
          <Tile src={`${versionThumb(photo.filepath, 0, "grid")}&camera=1`} label="As shot" picked={chosen === null}
            onClick={() => onChoose(null)} testid="dx-as-shot"
            title="Its raw light, without any edit of its own: the first photo's edit tones the two as one picture, as a camera's multiple exposure does." />
        )}
        {info === null && <div className="text-[11px] text-neutral-500 self-center px-2">Finding its versions…</div>}
        {versions.map((v) => (
          <Tile key={v.version} src={versionThumb(photo.filepath, v.version, "grid", `${editKey(v.settings)}.${stamp}`)}
            label={v.name} picked={chosen === v.version} star={v.version === info.cover}
            onClick={() => onChoose(v.version)} testid={`dx-version-${v.version}`}
            title={asShot ? "This version as it is edited: its crop, colour and look come with it." : "Starts from this version's edit."} />
        ))}
      </div>
    </div>
  );
}

// The chosen blend, large, to be placed by hand. The server's picture is shown when it is
// drawn; while the second is moved, a quick stand-in made in the browser follows the hand.
function Stage({ real, base, over, mode, amount, place, onPlace }) {
  const box = useRef(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const [overSize, setOverSize] = useState(null);
  const [baseAspect, setBaseAspect] = useState(1.5);
  const [moving, setMoving] = useState(false);
  const pointers = useRef(new Map());
  const start = useRef(null);
  const placeRef = useRef(place);
  placeRef.current = place;

  useEffect(() => {
    const el = box.current;
    if (!el) return undefined;
    const seen = new ResizeObserver(() => setSize({ w: el.clientWidth, h: el.clientHeight }));
    seen.observe(el);
    // Scrolling enlarges it; the page must not scroll instead.
    const wheel = (e) => {
      e.preventDefault();
      const p = placeRef.current;
      onPlace({ ...p, scale: Math.round(clamp(p.scale * Math.exp(-e.deltaY * 0.0015), 100, 400)) });
    };
    el.addEventListener("wheel", wheel, { passive: false });
    return () => { seen.disconnect(); el.removeEventListener("wheel", wheel); };
  }, [onPlace]);

  const spread = () => {
    const [a, b] = [...pointers.current.values()];
    return Math.hypot(a.x - b.x, a.y - b.y) || 1;
  };
  const down = (e) => {
    e.currentTarget.setPointerCapture?.(e.pointerId);
    pointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    start.current = { place: placeRef.current, at: { x: e.clientX, y: e.clientY },
      spread: pointers.current.size === 2 ? spread() : null };
    setMoving(true);
  };
  const move = (e) => {
    if (!pointers.current.has(e.pointerId) || !start.current) return;
    pointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    const s = start.current;
    if (pointers.current.size === 2 && s.spread) {
      onPlace({ ...s.place, scale: Math.round(clamp(s.place.scale * spread() / s.spread, 100, 400)) });
    } else if (pointers.current.size === 1 && size.w) {
      onPlace({
        ...s.place,
        x: Math.round(clamp(s.place.x + ((e.clientX - s.at.x) / size.w) * 100, -100, 100)),
        y: Math.round(clamp(s.place.y + ((e.clientY - s.at.y) / size.h) * 100, -100, 100)),
      });
    }
  };
  const up = (e) => {
    pointers.current.delete(e.pointerId);
    if (pointers.current.size) {
      // One finger left of a pinch carries on as a drag from here.
      const [p] = [...pointers.current.values()];
      start.current = { place: placeRef.current, at: p, spread: null };
    } else {
      start.current = null;
      setMoving(false);
    }
  };

  // The second laid on as adjust/double.py places it: just covering the frame at 100%.
  let layer = null;
  if (overSize && size.w) {
    const k = Math.max(size.w / overSize.w, size.h / overSize.h) * place.scale / 100;
    const w = overSize.w * k, h = overSize.h * k;
    const a = amount / 100;
    layer = {
      left: size.w * (0.5 + place.x / 100) - w / 2, top: size.h * (0.5 + place.y / 100) - h / 2, width: w, height: h,
      transform: place.flip ? "scaleX(-1)" : "none",
      mixBlendMode: { add: "plus-lighter", average: "normal", lighten: "lighten", darken: "darken" }[mode],
      opacity: mode === "average" ? a / (1 + a) : Math.min(1, a),
    };
  }
  const quick = moving || !real;
  return (
    <div ref={box} data-testid="dx-stage" onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerCancel={up}
      onDoubleClick={() => onPlace({ ...placeRef.current, x: 0, y: 0, scale: 100 })}
      title="Drag to move, scroll to enlarge, double-click to put it back in the middle at 100%."
      className="relative w-full rounded overflow-hidden bg-neutral-900 border border-neutral-800 cursor-grab active:cursor-grabbing select-none touch-none"
      style={{ aspectRatio: baseAspect, isolation: "isolate" }}>
      <img src={base || undefined} alt="" draggable={false} className="absolute inset-0 w-full h-full"
        style={{ visibility: base ? "visible" : "hidden" }}
        onLoad={(e) => setBaseAspect(e.currentTarget.naturalWidth / e.currentTarget.naturalHeight)} />
      <img src={over} alt="" draggable={false} className="absolute max-w-none"
        style={{ ...(layer || { visibility: "hidden" }), ...(quick ? {} : { visibility: "hidden" }) }}
        onLoad={(e) => setOverSize({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })} />
      {real && <img src={real} alt="" draggable={false} className="absolute inset-0 w-full h-full"
        style={{ visibility: quick ? "hidden" : "visible" }} data-testid="dx-stage-real" />}
      {quick && !moving && <div className="absolute bottom-1 right-2 text-[10px] text-neutral-400 animate-pulse">Drawing…</div>}
    </div>
  );
}

export default function DoubleExposureDialog({ first, other, onMade, onClose, editing = null }) {
  const [swapped, setSwapped] = useState(false);
  const [base, over] = swapped ? [other, first] : [first, other];
  const now = editing?.settings.second || null;
  const [info, setInfo] = useState({});          // filepath -> {versions, cover}
  const [chosen, setChosen] = useState(() => (now ? { [other.filepath]: now.version ?? null } : {}));
  const startMode = now ? Math.max(0, BLENDS.findIndex((b) => b.mode === now.mode)) : null;
  const [tiles, setTiles] = useState(() => BLENDS.map(() => (now
    ? { scale: now.scale ?? 100, x: now.x ?? 0, y: now.y ?? 0, flip: !!now.flip } : STILL)));
  const [picked, setPicked] = useState(startMode);
  const how = useMemo(() => (now
    ? { amount: now.amount ?? 100, exposure: now.exposure ?? 0, reach: now.reach || "", invert: !!now.invert } : HOW), [now]);
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState("");
  const [stamp] = useState(() => Date.now().toString(36));
  const render = useRenders();
  const stageRef = useRef(null);

  useEffect(() => {
    let gone = false;
    for (const photo of [first, other]) {
      versionsOf(photo.filepath)
        .then((d) => { if (!gone) setInfo((was) => ({ ...was, [photo.filepath]: d })); })
        .catch(() => { if (!gone) setInfo((was) => ({ ...was, [photo.filepath]: { versions: [{ version: 0, name: "Main", settings: photo.edit || {} }], cover: 0 } })); });
    }
    return () => { gone = true; };
  }, [first, other]);

  useEffect(() => {
    // Every key is the window's while it is open: an arrow must not change the photo behind it.
    const key = (e) => {
      e.stopPropagation();
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", key, true);
    return () => window.removeEventListener("keydown", key, true);
  }, [onClose]);

  // The photo the copy is of, in Develop: its own edit, and the version it was made from.
  const editingPhoto = editing ? first.filepath : null;
  const copyEdit = useMemo(() => (editing ? withoutSecond(editing.settings) : null), [editing]);
  const origin = useMemo(() => {
    const mine = editing && info[first.filepath];
    if (!mine) return 0;
    const found = mine.versions.find((v) => v.version !== editing.version && !v.settings.second && sameEdit(v.settings, copyEdit));
    return found ? found.version : mine.cover;
  }, [editing, info, first.filepath, copyEdit]);

  const versionFor = (photo, role) => {
    const raw = chosen[photo.filepath];
    const mine = photo.filepath === editingPhoto;
    const cover = info[photo.filepath]?.cover ?? 0;
    if (role === "base") {
      if (raw === undefined || raw === null) return mine ? COPY : cover;
      return raw;
    }
    if (raw === undefined) return mine ? origin : cover;
    return raw === COPY ? origin : raw;
  };
  const baseInfo = info[base.filepath];
  const overInfo = info[over.filepath];
  const baseVersion = versionFor(base, "base");
  const overVersion = versionFor(over, "over");
  const settingsOf = (photo, version) => (version === COPY ? copyEdit
    : version === null ? {} : info[photo.filepath]?.versions.find((v) => v.version === version)?.settings);
  const baseSettings = baseInfo ? settingsOf(base, baseVersion) : null;
  const overSettings = overInfo ? settingsOf(over, overVersion) : null;
  const ready = !!(baseSettings && overSettings);
  const second = useMemo(
    () => (ready ? secondOf(over.filepath, overVersion, overSettings) : null),
    [ready, over.filepath, overVersion, overSettings],
  );
  const laid = (i) => ({ ...how, mode: BLENDS[i].mode, ...tiles[i], ...second });
  const ask = (i, size) => (second
    ? { filepath: base.filepath, size, linear: true, adjustment: { ...withoutSecond(baseSettings), second: laid(i) } }
    : null);

  // The stage's real picture waits until the hand has rested a moment.
  const [settled, setSettled] = useState(null);
  const stageAsk = picked === null ? null : ask(picked, STAGE);
  const stageKey = stageAsk && JSON.stringify(stageAsk);
  useEffect(() => {
    const t = setTimeout(() => setSettled(stageKey), 250);
    return () => clearTimeout(t);
  }, [stageKey]);
  const real = stageAsk && settled === stageKey ? render(stageAsk) : null;
  const stageBase = ready && picked !== null
    ? render({ filepath: base.filepath, size: STAGE, linear: true, adjustment: withoutSecond(baseSettings) }) : null;
  const overSrc = overVersion === null
    ? `${versionThumb(over.filepath, 0, "preview")}&camera=1`
    : versionThumb(over.filepath, overVersion, "preview", overSettings ? `${editKey(overSettings)}.${stamp}` : stamp);

  const pick = (i) => {
    setPicked(i);
    requestAnimationFrame(() => stageRef.current?.scrollIntoView({ block: "nearest", behavior: "smooth" }));
  };
  const placeAt = useCallback((p) => setTiles((was) => was.map((t, i) => (i === picked ? p : t))), [picked]);

  const apply = async () => {
    if (!second || picked === null || busy) return;
    setBusy(true);
    setNote("");
    const done = laid(picked);
    try {
      if (editing && !swapped) {
        editing.onApply({ second: done, baseEdit: baseVersion === COPY ? null : withoutSecond(baseSettings) });
      } else {
        onMade(base.filepath, await makeDoubleExposure(base.filepath, baseVersion === COPY ? 0 : baseVersion, done));
      }
    } catch (e) {
      setNote(e.message || "Could not make the double exposure");
      setBusy(false);
    }
  };

  const choose = (photo) => (version) => setChosen((was) => ({ ...was, [photo.filepath]: version }));
  const still = tiles.every((p) => p.scale === 100 && !p.x && !p.y && !p.flip);
  const thisCopy = editing ? {
    // Drawn without its second, which this window lays on again.
    src: render({ filepath: first.filepath, size: TILE, linear: true, adjustment: copyEdit }),
    label: editing.version ? "This copy" : "As it is",
  } : null;
  const replacing = editing && !swapped && baseVersion !== COPY
    ? baseInfo?.versions.find((v) => v.version === baseVersion)?.name : null;
  const place = picked === null ? null : tiles[picked];

  return (
    <div className="fixed inset-0 z-[950] bg-black/80 flex items-center justify-center p-4" data-modal
      onClick={onClose} data-testid="double-dialog">
      <div className="bg-neutral-950 border border-neutral-800 rounded-lg max-w-4xl w-full max-h-[92vh] overflow-y-auto p-4"
        onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center gap-3 mb-3">
          <h2 className="text-sm text-neutral-200">◐ Double exposure</h2>
          <span className="text-[11px] text-neutral-500 hidden sm:inline">
            Pick the versions, then a blend, then place it by hand.
          </span>
          <button className="ml-auto text-neutral-400 hover:text-white px-2" onClick={onClose} title="Close (Esc)">✕</button>
        </div>

        <div className="grid grid-cols-[1fr_auto_1fr] gap-3 items-start">
          <Side title="Keeps the copy" photo={base} info={baseInfo ?? null} chosen={baseVersion}
            onChoose={choose(base)} stamp={stamp} testid="dx-base"
            copy={base.filepath === editingPhoto ? thisCopy : null} skip={base.filepath === editingPhoto ? editing.version : undefined} />
          <button onClick={() => setSwapped(!swapped)} data-testid="dx-swap"
            title="Swap them: the other photo keeps the copy and this one is laid over it."
            className="self-center mt-8 w-9 h-9 rounded-full border border-neutral-700 text-neutral-300 hover:text-white hover:border-amber-400">⇄</button>
          <Side title="Laid over it" photo={over} info={overInfo ?? null} chosen={overVersion}
            onChoose={choose(over)} asShot stamp={stamp} testid="dx-over"
            skip={over.filepath === editingPhoto ? editing.version : undefined} />
        </div>
        {replacing && (
          <div className="text-[11px] text-amber-300/90 mt-1" data-testid="dx-replacing">
            This copy's own edit will be replaced by {replacing}'s. Choose “{thisCopy?.label}” to keep it.
          </div>
        )}
        {editing && swapped && (
          <div className="text-[11px] text-amber-300/90 mt-1" data-testid="dx-new-copy">
            Swapped, a new copy is made of {base.filename}. This copy stays as it is.
          </div>
        )}

        <div className="flex items-center gap-2 mt-4 mb-2">
          <div className="text-[11px] uppercase tracking-wide text-neutral-500">Blends</div>
          <button onClick={() => setTiles(BLENDS.map(shaken))} data-testid="dx-shake"
            title="Lay the second photo on differently in each: larger, moved, mirrored."
            className="text-[11px] px-2 py-0.5 rounded border border-neutral-700 text-neutral-300 hover:text-white hover:border-neutral-500">🎲 Shake it up</button>
          {!still && (
            <button onClick={() => setTiles(BLENDS.map(() => STILL))}
              className="text-[11px] px-2 py-0.5 rounded border border-neutral-800 text-neutral-500 hover:text-neutral-200">Straight</button>
          )}
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2" data-testid="dx-blends">
          {BLENDS.map((blend, i) => {
            const src = render(ask(i, TILE));
            return (
              <button key={blend.mode} onClick={() => pick(i)} disabled={!second} title={blend.hint}
                data-testid={`dx-blend-${blend.mode}`}
                className={`group rounded overflow-hidden border text-left ${picked === i ? "border-amber-400 ring-1 ring-amber-400" : "border-neutral-800 hover:border-amber-400"}`}>
                <div className="aspect-[3/2] bg-neutral-900 flex items-center justify-center">
                  {src
                    ? <img src={src} alt={blend.label} className="w-full h-full object-contain" />
                    : <span className="text-[11px] text-neutral-600 animate-pulse">Drawing…</span>}
                </div>
                <div className="px-2 py-1 text-xs text-neutral-300 group-hover:text-white flex items-center">
                  {blend.label}
                  {(tiles[i].scale !== 100 || tiles[i].flip) && (
                    <span className="ml-auto text-[10px] text-neutral-500">
                      {tiles[i].scale}%{tiles[i].flip ? " · mirrored" : ""}
                    </span>
                  )}
                </div>
              </button>
            );
          })}
        </div>

        {picked === null ? (
          <div className="text-[11px] text-neutral-500 mt-3">Click the blend you like to place it by hand.</div>
        ) : (
          <div ref={stageRef} className="mt-4 space-y-2">
            <Stage real={real} base={stageBase} over={overSrc} mode={BLENDS[picked].mode} amount={how.amount}
              place={place} onPlace={placeAt} />
            <div className="flex flex-wrap items-center gap-2 text-xs text-neutral-400">
              <span className="hidden sm:inline">Drag to move · scroll or pinch to enlarge · double-click to centre</span>
              <span className="sm:hidden">Drag to move · pinch to enlarge · double-tap to centre</span>
              <label className="flex items-center gap-1.5 ml-auto">
                Size
                <input type="range" min={100} max={400} value={place.scale} data-testid="dx-size" className="w-28 accent-amber-400"
                  onChange={(e) => placeAt({ ...place, scale: +e.target.value })} />
                <span className="w-9 text-right text-neutral-300">{place.scale}%</span>
              </label>
              <button onClick={() => placeAt({ ...place, flip: !place.flip })} data-testid="dx-mirror"
                className={`px-2 py-0.5 rounded border ${place.flip ? "border-amber-400 text-amber-200" : "border-neutral-700 text-neutral-300 hover:text-white"}`}>⇋ Mirror</button>
              <button onClick={() => placeAt({ ...place, x: 0, y: 0, scale: 100 })} data-testid="dx-centre"
                className="px-2 py-0.5 rounded border border-neutral-700 text-neutral-300 hover:text-white">Centre</button>
            </div>
            <div className="flex items-center gap-3 pt-1">
              <div className="text-[11px] text-neutral-500 min-w-0">
                {editing && !swapped
                  ? "Amount, exposure and where it shows stay as they are in Develop."
                  : "A new copy is made; the photo's own edit stays as it is. Amount, exposure and where it shows are in Develop."}
              </div>
              <button onClick={apply} disabled={!second || busy} data-testid="dx-apply"
                className="ml-auto shrink-0 px-3 py-1.5 rounded bg-amber-500 text-black text-sm font-medium hover:bg-amber-400 disabled:opacity-50">
                {busy ? "Making it…" : editing && !swapped ? "Apply" : "Open in Develop"}
              </button>
            </div>
          </div>
        )}
        {note && <div className="text-xs text-red-400 mt-2">{note}</div>}
      </div>
    </div>
  );
}
