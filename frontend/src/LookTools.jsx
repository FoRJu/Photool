// The row under the looks while one is on the frame: how much of it, whether it
// is this camera's look, and the look as a .cube LUT for another editor.
import { useRef, useState } from "react";

const API = "/api";
export const LUT_SIZE = 33;

// The file name a download is offered under; the server makes its own safe too.
const safeName = (name) => (name || "look").replace(/[\\/:*?"<>|\x00-\x1f]+/g, "_").trim() || "look";

export default function LookTools({ look, amount, onAmount, camera, isCameras, onCamera, onLut, note }) {
  const [lutNote, setLutNote] = useState("");
  const [busy, setBusy] = useState(false);
  const lastTap = useRef(0);
  if (!look) return null;

  const back = () => onAmount?.(100);
  const doubleTap = () => {
    const now = Date.now();
    if (now - lastTap.current < 350) back();
    lastTap.current = now;
  };

  const saveLut = async () => {
    setBusy(true);
    setLutNote("");
    try {
      const res = await fetch(`${API}/export/develop/presets/${encodeURIComponent(look.name)}/lut?size=${LUT_SIZE}`);
      if (!res.ok) {
        let why = "Could not make a LUT of this look";
        try { why = (await res.json()).detail || why; } catch { /* plain text */ }
        setLutNote(why);
        return;
      }
      const left = res.headers.get("X-Lut-Left-Out") || "";
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${safeName(look.name)}.cube`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      const text = `Saved ${safeName(look.name)}.cube${left ? ` · not in it: ${left}` : ""}`;
      setLutNote(text);
      onLut?.(text, left);
    } catch {
      setLutNote("The server did not answer");
    } finally {
      setBusy(false);
    }
  };

  const shown = note || lutNote;
  return (
    <div className="rounded border border-neutral-800 bg-neutral-900/40 px-2 py-1 space-y-1" data-testid="look-tools">
      <div className="flex items-center gap-2 min-h-8">
        <label htmlFor="look-amount" className="text-[11px] text-neutral-500 shrink-0"
          title="How much of the look: 0 none, 100 as saved, 150 more">Amount</label>
        <input id="look-amount" type="range" min={0} max={150} step={1} value={amount}
          data-testid="look-amount" aria-label={`Amount of the look “${look.name}”`}
          onChange={(e) => onAmount?.(Number(e.target.value))}
          onDoubleClick={back} onTouchEnd={doubleTap}
          className="flex-1 min-w-0 h-8 accent-amber-400" />
        <button onClick={back} onTouchEnd={doubleTap} title="Double-click: back to 100"
          className="text-[11px] tabular-nums w-9 h-8 text-right text-neutral-300 hover:text-white"
          onDoubleClick={back} tabIndex={-1}>{amount}</button>
      </div>
      {look.own && (
        <div className="flex flex-wrap items-center gap-1">
          {camera && (
            <button onClick={() => onCamera?.(!isCameras)} data-testid="look-camera"
              aria-pressed={!!isCameras}
              title={isCameras ? `The look for ${camera}: click to take the star off`
                : `Make this the look for ${camera}: one look per camera`}
              className={`text-[11px] h-8 px-2 rounded border max-w-full truncate ${isCameras
                ? "border-amber-500 text-amber-300" : "border-neutral-700 text-neutral-300 hover:text-white hover:border-neutral-500"}`}>
              {isCameras ? "★" : "☆"} For {camera}
            </button>
          )}
          <button onClick={saveLut} disabled={busy} data-testid="look-lut"
            title="Download this look as a .cube LUT, for a video editor or another photo app"
            className="text-[11px] h-8 px-2 rounded border border-neutral-700 text-neutral-300 hover:text-white hover:border-neutral-500 disabled:opacity-40">
            {busy ? "LUT…" : "LUT"}
          </button>
        </div>
      )}
      {shown && <div className="text-[10px] text-neutral-500" data-testid="look-tools-note">{shown}</div>}
    </div>
  );
}
