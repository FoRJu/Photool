// White balance in Kelvin and tint, with a grey picker (backend adjust/whitebalance.py).
//
// The edit holds two absolute keys: `wb_kelvin` (2000..50000) and `wb_tint`
// (-150..150), Lightroom's scales. Absent is as shot. The panel always writes both
// together, and removes both to go back to as shot. A JPEG's own Kelvin is not
// known, so there the same keys are shown relative, -100..100 either way.
//
// The old `temperature` and `tint` (a relative trim applied after) are no longer
// controls here: an edit that carries them says so in one line, with a Clear.
import { useEffect, useRef, useState } from "react";

const API = "/api";

const K_MIN = 2000;
const K_MAX = 50000;
const M_LOW = 1e6 / K_MAX;     // mireds at the warm (right) end
const M_HIGH = 1e6 / K_MIN;    // mireds at the cool (left) end
const T_MAX = 150;
// Not RAW: one step of the relative Temp is this many mireds about the file's own white.
const MIRED_STEP = 1.2;

// Higher Kelvin is a warmer picture (Lightroom's convention): blue to amber.
const TEMP_TRACK = "linear-gradient(to right, #3b6fd8, #9fb8e6 45%, #e8d9b0 55%, #e0a030)";
const TINT_TRACK = "linear-gradient(to right, #3fa34d, #b9d9a8 45%, #e2b8de 55%, #c03ec0)";

const PRESETS = [
  { label: "Daylight", kelvin: 5500, tint: 10 },
  { label: "Cloudy", kelvin: 6500, tint: 10 },
  { label: "Shade", kelvin: 7500, tint: 10 },
  { label: "Tungsten", kelvin: 2850, tint: 0 },
  { label: "Fluorescent", kelvin: 3800, tint: 21 },
  { label: "Flash", kelvin: 5500, tint: 0 },
];

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const signed = (v) => `${v > 0 ? "+" : ""}${v}`;
// Kelvin to the nearest step a person would set: 10 K, or 100 K above 10000.
const roundKelvin = (k) => {
  const step = k >= 10000 ? 100 : 10;
  return clamp(Math.round(k / step) * step, K_MIN, K_MAX);
};

// The two scales each control is shown on. `toFraction` and `fromFraction` place a
// stored value along the bar; `show` writes it; `nudge` is one arrow-key step.
function scales(raw, asShot) {
  const m0 = 1e6 / asShot.kelvin;
  if (raw) {
    return {
      temp: {
        toFraction: (k) => (M_HIGH - 1e6 / k) / (M_HIGH - M_LOW),
        fromFraction: (f) => roundKelvin(1e6 / (M_HIGH - clamp(f, 0, 1) * (M_HIGH - M_LOW))),
        show: (k) => `${Math.round(k)} K`,
        nudge: (k, dir) => roundKelvin(1e6 / (1e6 / k - dir * 2)),
      },
      tint: {
        toFraction: (t) => (t + T_MAX) / (2 * T_MAX),
        fromFraction: (f) => Math.round(clamp(f, 0, 1) * 2 * T_MAX - T_MAX),
        show: (t) => signed(Math.round(t)),
        nudge: (t, dir) => clamp(Math.round(t) + dir, -T_MAX, T_MAX),
      },
    };
  }
  // Relative: -100..100, 0 at the file's own white.
  const kOf = (s) => clamp(1e6 / (m0 - MIRED_STEP * s), K_MIN, K_MAX);
  const sOf = (k) => clamp((m0 - 1e6 / k) / MIRED_STEP, -100, 100);
  const tOf = (s) => clamp(asShot.tint + s, -T_MAX, T_MAX);
  const uOf = (t) => clamp(t - asShot.tint, -100, 100);
  return {
    temp: {
      toFraction: (k) => (sOf(k) + 100) / 200,
      fromFraction: (f) => Math.round(kOf(Math.round(clamp(f, 0, 1) * 200 - 100))),
      show: (k) => signed(Math.round(sOf(k))),
      nudge: (k, dir) => Math.round(kOf(Math.round(sOf(k)) + dir)),
    },
    tint: {
      toFraction: (t) => (uOf(t) + 100) / 200,
      fromFraction: (f) => Math.round(tOf(Math.round(clamp(f, 0, 1) * 200 - 100))),
      show: (t) => signed(Math.round(uOf(t))),
      nudge: (t, dir) => Math.round(tOf(Math.round(uOf(t)) + dir)),
    },
  };
}

// One bar: its track painted, a mark where the value is, its name and value on it.
// A press or a drag sets it; a double-click or a double-tap puts both back to as shot.
function Bar({ label, testid, track, scale, value, changed, title, onSet, onReset }) {
  const ref = useRef(null);
  const lastTap = useRef(0);
  const set = (e) => {
    const r = ref.current.getBoundingClientRect();
    onSet(scale.fromFraction((e.clientX - r.left) / r.width));
  };
  const f = clamp(scale.toFraction(value), 0, 1);
  return (
    <div ref={ref} tabIndex={0} role="slider" aria-label={label} aria-valuetext={scale.show(value)}
      data-testid={testid} title={`${title}\n\nDouble-click or double-tap: as shot.`}
      onPointerDown={(e) => {
        if (e.button !== 0) return;
        const now = e.timeStamp;
        if (now - lastTap.current < 320) { lastTap.current = 0; onReset(); return; }
        lastTap.current = now;
        e.currentTarget.setPointerCapture(e.pointerId);
        set(e);
      }}
      onPointerMove={(e) => { if (e.currentTarget.hasPointerCapture(e.pointerId)) set(e); }}
      onKeyDown={(e) => {
        const dir = { ArrowRight: 1, ArrowUp: 1, ArrowLeft: -1, ArrowDown: -1 }[e.key];
        if (!dir) return;
        e.preventDefault();
        onSet(scale.nudge(value, dir * (e.shiftKey ? 10 : 1)));
      }}
      className="relative h-8 rounded overflow-hidden cursor-ew-resize select-none touch-none bg-neutral-800 focus:outline-none focus-visible:ring-1 focus-visible:ring-amber-400">
      <div className="absolute inset-0 opacity-45" style={{ background: track }} />
      <div className={`absolute inset-y-0 w-0.5 ${changed ? "bg-amber-300" : "bg-white/80"}`}
        style={{ left: `calc(${f * 100}% - 1px)` }} />
      <div className="relative h-full flex items-center justify-between px-2 text-[11px] pointer-events-none">
        <span className={changed ? "text-amber-200" : "text-neutral-200"}>{label}</span>
        <span className={`font-mono ${changed ? "text-amber-200" : "text-neutral-300"}`}>{scale.show(value)}</span>
      </div>
    </div>
  );
}

const button = "min-h-8 text-[11px] px-2 rounded border";
const idle = "border-neutral-700 text-neutral-300 hover:text-white hover:border-neutral-500";
const lit = "border-amber-400 text-amber-300";

export default function WhiteBalance({ filepath, settings, onChange, onApply, picking, setPicking, pickRef, pickBody }) {
  // What the file knows of its own white: {raw, as_shot: {kelvin, tint}}.
  const [info, setInfo] = useState(null);
  const [note, setNote] = useState("");
  useEffect(() => {
    let live = true;
    setInfo(null);
    setNote("");
    fetch(`${API}/export/develop/white-balance`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ filepath }),
    })
      .then((res) => (res.ok ? res.json() : Promise.reject()))
      .then((data) => { if (live) setInfo(data); })
      .catch(() => { if (live) setInfo({ raw: false, as_shot: { kelvin: 6500, tint: 10 } }); });
    return () => { live = false; };
  }, [filepath]);

  const custom = Number(settings.wb_kelvin) > 0;
  const asShot = info?.as_shot || { kelvin: 6500, tint: 10 };
  const kelvin = custom ? Number(settings.wb_kelvin) : asShot.kelvin;
  const tint = custom ? Number(settings.wb_tint) || 0 : asShot.tint;

  // Both keys, always together. The first move from as shot writes both in one
  // step; after that a drag moves the one key, as any slider does.
  const setBoth = (k, t, label) => onApply(["wb_kelvin", "wb_tint"], { wb_kelvin: k, wb_tint: t }, label);
  const shown = scales(!!info?.raw, asShot);
  const setTemp = (k) => (custom ? onChange("wb_kelvin", k)
    : setBoth(k, asShot.tint, `Temp ${shown.temp.show(k)}`));
  const setTint = (t) => (custom ? onChange("wb_tint", t)
    : setBoth(asShot.kelvin, t, `Tint ${shown.tint.show(t)}`));
  const asShotAgain = () => { if (custom) onApply(["wb_kelvin", "wb_tint"], {}, "White balance as shot"); };

  // The picker: the next click or tap on the picture asks for the white that makes it grey.
  pickRef.current = picking ? async (u, v) => {
    setPicking(false);
    setNote("");
    try {
      const res = await fetch(`${API}/export/develop/white-point`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(pickBody(u, v)),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) { setNote(data.detail || "Could not read that spot."); return; }
      setBoth(data.kelvin, data.tint, "White balance picked");
    } catch {
      setNote("The server did not answer.");
    }
  } : null;
  useEffect(() => () => { pickRef.current = null; }, [pickRef]);

  const oldWarmth = Number(settings.temperature) || 0;
  const oldTint = Number(settings.tint) || 0;

  if (!info) return <div className="text-[10px] text-neutral-600" data-testid="wb-loading">Reading the white balance…</div>;
  const s = shown;
  const near = (p) => custom && Math.abs(kelvin - p.kelvin) < 1 && Math.abs(tint - p.tint) < 1;

  return (
    <div className="space-y-1.5" data-testid="white-balance">
      <Bar label="Temp" testid="wb-temp" track={TEMP_TRACK} scale={s.temp} value={kelvin} changed={custom}
        title={info.raw
          ? "The colour of the light, in Kelvin, as Lightroom's Temp. Higher makes the picture warmer."
          : "Warmer or cooler than the file's own white. This file's Kelvin is not known, so it is shown relative."}
        onSet={setTemp} onReset={asShotAgain} />
      <Bar label="Tint" testid="wb-tint" track={TINT_TRACK} scale={s.tint} value={tint} changed={custom}
        title="Green (left) or magenta (right), as Lightroom's Tint."
        onSet={setTint} onReset={asShotAgain} />
      <div className="text-[10px] text-neutral-500" data-testid="wb-as-shot">
        {info.raw ? `As shot ${asShot.kelvin} K · ${signed(asShot.tint)}` : "As shot: this file's own white"}
      </div>
      <div className="flex flex-wrap gap-1">
        <button onClick={asShotAgain} title="The white balance the camera recorded"
          className={`${button} ${custom ? idle : lit}`}>As shot</button>
        {info.raw && PRESETS.map((p) => (
          <button key={p.label} onClick={() => setBoth(p.kelvin, p.tint, `White balance ${p.label}`)}
            title={`${p.kelvin} K · ${signed(p.tint)}`} className={`${button} ${near(p) ? lit : idle}`}>
            {p.label}
          </button>
        ))}
        <button onClick={() => { setNote(""); setPicking(!picking); }} data-testid="wb-pick"
          title="Then click or tap something in the picture that should be neutral grey or white."
          className={`${button} ${picking ? lit : idle}`}>
          {picking ? "Click something grey…" : "Pick grey"}
        </button>
      </div>
      {note && <div className="text-[10px] text-amber-400" data-testid="wb-note">{note}</div>}
      {(oldWarmth !== 0 || oldTint !== 0) && (
        <div className="flex items-center gap-1 text-[10px] text-neutral-500" data-testid="wb-earlier">
          <span title="A relative warmth and tint from before Temp and Tint, applied on top of them">
            Earlier warmth {signed(oldWarmth)}, tint {signed(oldTint)} ·
          </span>
          <button onClick={() => onApply(["temperature", "tint"], {}, "Earlier warmth cleared")}
            className="min-h-8 px-1 underline hover:text-neutral-300">Clear</button>
        </div>
      )}
    </div>
  );
}
