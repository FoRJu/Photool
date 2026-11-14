// Lens blur that ends at an outline of what is sharp (backend adjust/outline.py).
//
// The depth map is soft at an edge, so on its own the blur gives a person's edge a
// distance between theirs and the wall's. A click here outlines the thing clicked
// with SAM 2.1, sets the focus to its distance, and from then on the blur ends at
// that outline. The edit holds it as `blur_outline`: {matte, parts, model}, a kept
// selection as an object mask's is. It belongs to its frame: it is not pasted, nor
// kept in a saved look.
import { useEffect, useState } from "react";

const API = "/api";

const button = "min-h-8 text-[11px] px-2 rounded border";
const idle = "border-neutral-700 text-neutral-300 hover:text-white hover:border-neutral-500";
const lit = "border-amber-400 text-amber-300";

// SAM 2.1 is asked for once a visit, before the first outline: the first time
// ever it is a download.
let modelReady = false;

async function readyModel(setNote, setWait) {
  if (modelReady) return true;
  const slow = setTimeout(() => setWait("Getting SAM 2.1, about 330 MB, the first time only…"), 500);
  try {
    const res = await fetch(`${API}/export/develop/object-model`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model: "sam2.1" }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      setNote(data.detail || "Could not get SAM 2.1.");
      return false;
    }
    modelReady = true;
    return true;
  } catch {
    setNote("The server did not answer.");
    return false;
  } finally {
    clearTimeout(slow);
    setWait("");
  }
}

export default function BlurOutline({ filepath, settings, onApply, picking, setPicking, pickRef, pickBody }) {
  const [note, setNote] = useState("");
  const [wait, setWait] = useState("");
  const [busy, setBusy] = useState(false);
  useEffect(() => { setNote(""); setBusy(false); }, [filepath]);

  const held = settings.blur_outline?.matte ? settings.blur_outline : null;
  const count = held ? Math.max(1, (held.parts || []).length) : 0;

  // The next click or tap on the picture outlines what is there, and adds it.
  pickRef.current = picking ? async (u, v) => {
    setPicking(false);
    setNote("");
    setBusy(true);
    try {
      if (!(await readyModel(setNote, setWait))) return;
      const res = await fetch(`${API}/export/develop/blur-outline`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...pickBody(u, v), parts: held?.parts || [] }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) { setNote(data.detail || "Could not outline that."); return; }
      onApply(["blur_outline", "blur_focus"], { blur_outline: data.outline, blur_focus: data.depth },
        held ? "Added to the blur's outline" : "Blur outlined");
    } catch {
      setNote("The server did not answer.");
    } finally {
      setBusy(false);
    }
  } : null;
  useEffect(() => () => { pickRef.current = null; }, [pickRef]);

  return (
    <div className="space-y-1.5" data-testid="blur-outline-panel">
      <div className="flex flex-wrap items-center gap-1">
        <button onClick={() => { setNote(""); setPicking(!picking); }} data-testid="blur-outline" disabled={busy}
          title="Then click what should be sharp: it is outlined, the focus goes to its distance, and the blur ends where it ends."
          className={`${button} ${picking ? lit : idle} disabled:opacity-50`}>
          {busy ? "Outlining…" : picking ? "Click it on the picture…" : held ? "Add to the outline" : "Outline what is sharp"}
        </button>
        {held && (
          <button onClick={() => onApply(["blur_outline"], {}, "Blur outline cleared")} data-testid="blur-outline-clear"
            title="The blur goes back to ending where the depth map says" className={`${button} ${idle}`}>
            Clear
          </button>
        )}
      </div>
      {held && (
        <div className="flex items-center gap-2 text-[10px] text-neutral-400" data-testid="blur-outline-held">
          <img src={`${API}/export/develop/object-matte/${held.matte}`} alt="The outline"
            className="w-24 h-auto bg-neutral-950 rounded border border-neutral-800" />
          <span>Outlined: {count} {count === 1 ? "thing" : "things"}</span>
        </div>
      )}
      {wait && <div className="text-[10px] text-neutral-500" data-testid="blur-outline-wait">{wait}</div>}
      {note && <div className="text-[10px] text-amber-400" data-testid="blur-outline-note">{note}</div>}
    </div>
  );
}
