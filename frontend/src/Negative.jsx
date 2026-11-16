// A scanned negative, turned back into the scene (backend adjust/negative.py).
//
// Being a negative is a fact about the photograph, not part of its edit: nothing
// here touches the edit or its undo history. The description is kept by the
// catalog and the frame is inverted where it is decoded, so every tool after this
// one (white balance, Auto, masks, the loupe, the export) works on the scene.
import { useEffect, useRef, useState } from "react";

const API = "/api";

const SCAN_EXT = /\.(tiff?|psd|psb)$/i;

const FROM = {
  border: "from the film's border",
  picture: "from the darkest part of the picture",
  picked: "picked by you",
  roll: "from another frame",
};

const button = "min-h-8 text-[11px] px-2 rounded border";
const idle = "border-neutral-700 text-neutral-300 hover:text-white hover:border-neutral-500";
const lit = "border-amber-400 text-amber-300";

// A short key for a description, so a thumbnail's address changes with it.
export function negativeStamp(d) {
  if (!d) return "";
  let h = 5381;
  for (const ch of JSON.stringify(d)) h = ((h << 5) + h + ch.charCodeAt(0)) | 0;
  return (h >>> 0).toString(36);
}

// The base's colour as a swatch: linear values, roughly display-encoded.
const swatch = (base) => `rgb(${base.map((v) => Math.round(255 * Math.min(1, Math.max(0, v)) ** (1 / 2.2))).join(",")})`;

function Slider({ label, min, max, value, title, testid, onCommit }) {
  const [v, setV] = useState(value);
  useEffect(() => { setV(value); }, [value]);
  const commit = () => { if (Number(v) !== Number(value)) onCommit(Number(v)); };
  return (
    <label className="flex items-center gap-2 text-xs" title={`${title}\n\nDouble-click: back to ${label === "Contrast" ? 0 : 100}.`}>
      <span className="w-20 shrink-0 text-neutral-400 select-none"
        onDoubleClick={() => onCommit(label === "Contrast" ? 0 : 100)}>{label}</span>
      <input type="range" min={min} max={max} step={1} value={v} data-testid={testid}
        onChange={(e) => setV(Number(e.target.value))}
        onPointerUp={commit} onKeyUp={commit} onBlur={commit}
        className="flex-1 min-w-0 accent-amber-400" />
      <span className="w-8 text-right font-mono text-neutral-300">{Math.round(v)}</span>
    </label>
  );
}

export default function Negative({ filepath, photos, picked, picking, setPicking, pickRef, pickBody, onChanged }) {
  const [info, setInfo] = useState(null);         // {negative, tone}
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [offer, setOffer] = useState(null);       // the photos "Give to the roll" would go to
  const live = useRef(filepath);

  useEffect(() => {
    live.current = filepath;
    setInfo(null);
    setNote("");
    setOffer(null);
    fetch(`${API}/export/develop/negative?filepath=${encodeURIComponent(filepath)}`)
      .then((res) => (res.ok ? res.json() : Promise.reject()))
      .then((data) => { if (live.current === filepath) setInfo(data); })
      .catch(() => { if (live.current === filepath) setInfo({ negative: null, tone: null }); });
  }, [filepath]);

  const d = info?.negative || null;

  const post = async (body, filepaths = [filepath]) => {
    setBusy(true);
    setNote("");
    try {
      const res = await fetch(`${API}/export/develop/negative`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filepaths, ...body }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) { setNote(data.detail || "Could not change it."); return null; }
      if (filepath in data && live.current === filepath) {
        setInfo((old) => ({ ...(old || {}), negative: data[filepath] }));
      }
      onChanged?.(data);
      return data;
    } catch {
      setNote("The server did not answer.");
      return null;
    } finally {
      setBusy(false);
    }
  };

  const choose = (kind) => {
    if (kind === (d?.kind || null)) return;
    post({ kind });
  };

  // The picker: the next click on the picture (shown as the scan while picking)
  // reads the film base there.
  pickRef.current = picking ? async (u, v) => {
    setPicking(false);
    setNote("");
    try {
      const res = await fetch(`${API}/export/develop/negative/base-at`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(pickBody(u, v)),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) { setNote(data.detail || "Could not read that spot."); return; }
      post({ kind: d?.kind || "colour", base: data.base, base_from: "picked" });
    } catch {
      setNote("The server did not answer.");
    }
  } : null;
  useEffect(() => () => { pickRef.current = null; }, [pickRef]);

  // Where "Give to the roll" goes: the selection when there is one, else the
  // other scans in the folder (or, with none, the folder's other photos).
  const rollTargets = () => {
    const others = (list) => list.filter((p) => p !== filepath);
    const chosen = others(picked || []);
    if (chosen.length) return { paths: chosen, one: "selected photo", many: "selected photos" };
    const all = (photos || []).map((p) => p.filepath);
    const scans = others(all.filter((p) => SCAN_EXT.test(p)));
    if (scans.length) return { paths: scans, one: "other scan in this folder", many: "other scans in this folder" };
    return { paths: others(all), one: "other photo in this folder", many: "other photos in this folder" };
  };
  const giveToRoll = async () => {
    if (!offer) return;
    const { paths } = offer;
    setOffer(null);
    const done = await post({ kind: d.kind, base: d.base, base_from: "roll", contrast: d.contrast,
      balance: d.balance }, paths);
    if (done) setNote(`Given to ${paths.length} photo${paths.length === 1 ? "" : "s"}; each measured its own range.`);
  };

  if (!info) return <div className="text-[10px] text-neutral-600">Reading…</div>;

  return (
    <div className="space-y-1.5" data-testid="negative">
      <div className="flex flex-wrap gap-1">
        {[{ kind: null, label: "Not a negative" }, { kind: "colour", label: "Colour negative" },
          { kind: "bw", label: "Black & white negative" }].map((c) => (
          <button key={c.label} disabled={busy} onClick={() => choose(c.kind)} data-testid={`negative-${c.kind || "none"}`}
            title={c.kind ? "Turn this scan back into the scene it recorded. The film base is found on its own." : "Show the scan as it is"}
            className={`${button} ${(d?.kind || null) === c.kind ? lit : idle} disabled:opacity-50`}>
            {c.label}
          </button>
        ))}
      </div>
      {d && (
        <>
          <div className="flex items-center gap-2 text-[11px] text-neutral-400" data-testid="negative-base">
            <span className="inline-block w-5 h-5 rounded border border-neutral-600 shrink-0"
              style={{ background: swatch(d.base) }} title={`Film base ${d.base.map((v) => v.toFixed(3)).join(" · ")}`} />
            <span>Film base {FROM[d.base_from] || ""}</span>
          </div>
          {d.base_from === "picture" && (
            <div className="text-[10px] text-neutral-500">
              A frame with no true black in it comes out a little dark: pick the base on a frame that shows its border and give it to this one.
            </div>
          )}
          <div className="flex flex-wrap gap-1">
            <button onClick={() => { setNote(""); setPicking(!picking); }} disabled={busy} data-testid="negative-pick"
              title="Then click unexposed film: the border, or the gap between two frames. The scan is shown as it is while you pick."
              className={`${button} ${picking ? lit : idle}`}>
              {picking ? "Click unexposed film…" : "Pick film base"}
            </button>
            <button onClick={() => post({ kind: d.kind, remeasure: true })} disabled={busy} data-testid="negative-measure"
              title="Measure again, inside the crop: crop off the holder and the border, then measure again. A base you picked or gave is kept."
              className={`${button} ${idle}`}>Measure again</button>
            <button onClick={() => { const t = rollTargets(); setOffer(t.paths.length ? t : null);
              if (!t.paths.length) setNote("No other photos to give it to."); }} disabled={busy} data-testid="negative-roll"
              title="Give this frame's film base, kind, Contrast and Layer balance to the selected photos, or to the other scans of this folder. Each measures its own range."
              className={`${button} ${offer ? lit : idle}`}>Give to the roll</button>
          </div>
          {offer && (
            <div className="flex flex-wrap items-center gap-1 text-[11px] text-neutral-300" data-testid="negative-roll-offer">
              <span>Give this film base to {offer.paths.length} {offer.paths.length === 1 ? offer.one : offer.many}?</span>
              <button onClick={giveToRoll} className={`${button} ${lit}`}>Give</button>
              <button onClick={() => setOffer(null)} className={`${button} ${idle}`}>Cancel</button>
            </div>
          )}
          <Slider label="Contrast" min={-100} max={100} value={d.contrast} testid="negative-contrast"
            title="The film's contrast. Also what corrects a scan whose values were read with the wrong curve."
            onCommit={(v) => post({ kind: d.kind, contrast: v })} />
          {d.kind === "colour" && (
            <Slider label="Layer balance" min={0} max={100} value={d.balance} testid="negative-balance"
              title="How far the three colour layers are evened out, so blacks and whites are neutral. Bring it down for a sunset or a scene that really was all one colour."
              onCommit={(v) => post({ kind: d.kind, balance: v })} />
          )}
        </>
      )}
      {info.tone && (
        <div className="text-[10px] text-neutral-600" data-testid="negative-tone">
          {info.tone === "linear" ? "Read as a linear scan." : "Read as a gamma-encoded scan."}
        </div>
      )}
      {note && <div className="text-[10px] text-amber-400" data-testid="negative-note">{note}</div>}
    </div>
  );
}
