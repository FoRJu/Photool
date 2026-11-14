// Lens blur that ends at an outline of what is sharp (backend adjust/outline.py).
//
// The depth map is soft at an edge, so on its own the blur gives a person's edge a
// distance between theirs and the wall's. Here what is sharp is outlined with the model
// object masks use (SAM 2.1, or SAM 3 once it has been chosen there), the focus goes to
// its distance, and from then on the blur ends at that outline. It can be a click on a
// thing, everyone in the frame at once, or with SAM 3 everything a word names. The edit
// holds it as `blur_outline`: {matte, parts, model, people?}, a kept selection as an
// object mask's is. It belongs to its frame: it is not pasted, nor kept in a saved look.
import { useEffect, useState } from "react";

const API = "/api";

const button = "min-h-8 text-[11px] px-2 rounded border";
const idle = "border-neutral-700 text-neutral-300 hover:text-white hover:border-neutral-500";
const lit = "border-amber-400 text-amber-300";

// The model is asked for once a visit, before the first outline: the first time ever
// SAM 2.1 is a download. SAM 3 is used only when it is already here (its licence is
// agreed to where object masks fetch it), so nothing of that size is fetched from here.
const ready = {};

async function readyModel(chosen, setNote, setWait) {
  let model = "sam2.1";
  let finds = false;
  try {
    const res = await fetch(`${API}/export/develop/object-models`);
    const list = res.ok ? (await res.json()).models : [];
    const mine = list.find((m) => m.model === chosen);
    if (mine?.available) { model = mine.model; finds = !!mine.finds; }
  } catch { /* SAM 2.1 */ }
  if (ready[model]) return { model, finds };
  const slow = setTimeout(() => setWait(model === "sam2.1"
    ? "Getting SAM 2.1, about 330 MB, the first time only…" : "Loading SAM 3…"), 500);
  try {
    const res = await fetch(`${API}/export/develop/object-model`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      setNote(data.detail || "Could not get the model.");
      return null;
    }
    ready[model] = true;
    return { model, finds };
  } catch {
    setNote("The server did not answer.");
    return null;
  } finally {
    clearTimeout(slow);
    setWait("");
  }
}

// What the outline holds, in a few words: "the people, a dog and 2 things".
function described(held) {
  const parts = held.parts || [];
  const names = [...new Set(parts.filter((p) => p.text).map((p) => (p.text === "person" ? "the people" : p.text)))];
  if (held.people && !names.includes("the people")) names.unshift("the people");
  const clicked = parts.filter((p) => !p.text).length;
  if (clicked) names.push(`${clicked} ${clicked === 1 ? "thing" : "things"}`);
  if (!names.length) return "1 thing";
  return names.length === 1 ? names[0] : `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;
}

export default function BlurOutline({ filepath, settings, onApply, picking, setPicking, pickRef, pickBody, objectModel }) {
  const [note, setNote] = useState("");
  const [wait, setWait] = useState("");
  const [busy, setBusy] = useState("");
  const [finds, setFinds] = useState(false);
  const [name, setName] = useState("");
  // Naming things is SAM 3's, and offered only when it is the object masks' model and here.
  useEffect(() => {
    let live = true;
    fetch(`${API}/export/develop/object-models`).then((r) => (r.ok ? r.json() : { models: [] }))
      .then(({ models }) => {
        const mine = models.find((m) => m.model === objectModel);
        if (live) setFinds(!!(mine?.available && mine.finds));
      }).catch(() => {});
    return () => { live = false; };
  }, [objectModel]);
  useEffect(() => { setNote(""); setBusy(""); }, [filepath]);

  const held = settings.blur_outline?.matte ? settings.blur_outline : null;

  const outline = async (what, ask) => {
    setNote("");
    setBusy(what);
    try {
      const got = await readyModel(objectModel, setNote, setWait);
      if (!got) return;
      setFinds(got.finds);
      const res = await fetch(`${API}/export/develop/blur-outline`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...ask, model: got.model, outline: held || {} }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) { setNote(data.detail || "Could not outline that."); return; }
      onApply(["blur_outline", "blur_focus"], { blur_outline: data.outline, blur_focus: data.depth },
        held ? "Added to the blur's outline" : "Blur outlined");
      if (what === "name") setName("");
    } catch {
      setNote("The server did not answer.");
    } finally {
      setBusy("");
    }
  };

  // The next click or tap on the picture outlines what is there, and adds it.
  pickRef.current = picking ? (u, v) => {
    setPicking(false);
    outline("point", { ...pickBody(u, v), kind: "point" });
  } : null;
  useEffect(() => () => { pickRef.current = null; }, [pickRef]);

  const everyone = held && (held.people || (held.parts || []).some((p) => p.text === "person"));

  return (
    <div className="space-y-1.5" data-testid="blur-outline-panel">
      <div className="flex flex-wrap items-center gap-1">
        <button onClick={() => { setNote(""); setPicking(!picking); }} data-testid="blur-outline" disabled={!!busy}
          title="Then click what should be sharp: it is outlined, the focus goes to its distance, and the blur ends where it ends."
          className={`${button} ${picking ? lit : idle} disabled:opacity-50`}>
          {busy === "point" ? "Outlining…" : picking ? "Click it on the picture…" : held ? "Add to the outline" : "Outline what is sharp"}
        </button>
        {!everyone && (
          <button onClick={() => { setPicking(false); outline("people", { ...pickBody(0.5, 0.5), kind: "people" }); }}
            data-testid="blur-outline-people" disabled={!!busy}
            title="Everyone in the frame, outlined at once: the focus goes to the middle of where they stand."
            className={`${button} ${idle} disabled:opacity-50`}>
            {busy === "people" ? "Finding everyone…" : "Outline the people"}
          </button>
        )}
        {held && (
          <button onClick={() => onApply(["blur_outline"], {}, "Blur outline cleared")} data-testid="blur-outline-clear"
            disabled={!!busy} title="The blur goes back to ending where the depth map says"
            className={`${button} ${idle} disabled:opacity-50`}>
            Clear
          </button>
        )}
      </div>
      {finds && (
        <form className="flex items-center gap-1" data-testid="blur-outline-name-form"
          onSubmit={(e) => { e.preventDefault(); if (name.trim()) outline("name", { ...pickBody(0.5, 0.5), kind: "text", text: name }); }}>
          <input value={name} onChange={(e) => setName(e.target.value)} disabled={!!busy} maxLength={60}
            placeholder="Or name it: dog, bicycle…" data-testid="blur-outline-name" aria-label="What to outline, by name"
            className="min-w-0 flex-1 min-h-8 text-[11px] px-2 rounded border border-neutral-700 bg-neutral-900 text-neutral-200 placeholder:text-neutral-600" />
          <button type="submit" disabled={!!busy || !name.trim()} data-testid="blur-outline-name-go"
            title="Every one of them SAM 3 finds is outlined" className={`${button} ${idle} disabled:opacity-50`}>
            {busy === "name" ? "Finding…" : "Outline"}
          </button>
        </form>
      )}
      {held && (
        <div className="flex items-center gap-2 text-[10px] text-neutral-400" data-testid="blur-outline-held">
          <img src={`${API}/export/develop/object-matte/${held.matte}`} alt="The outline"
            className="w-24 h-auto bg-neutral-950 rounded border border-neutral-800" />
          <span>Outlined: {described(held)}</span>
        </div>
      )}
      {wait && <div className="text-[10px] text-neutral-500" data-testid="blur-outline-wait">{wait}</div>}
      {note && <div className="text-[10px] text-amber-400" data-testid="blur-outline-note">{note}</div>}
    </div>
  );
}
