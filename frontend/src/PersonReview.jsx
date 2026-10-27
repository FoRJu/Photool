import { useState, useEffect, useCallback, useMemo, useRef } from "react";
import PersonTimeline from "./PersonTimeline.jsx";

// ── Checking one person's faces ─────────────────────────────────────
// The grouping is right most of the time and wrong in one recognisable way: a
// face that sits a little apart from everyone else in the group. So the faces
// are shown least alike first, the doubtful ones are marked, and the answer is
// one or two clicks: "not them" (remembered, so no rescan puts it back) or
// "this is …" someone else. The names on the photos follow the faces.

const API = "/api";

function faceCropUrl(filepath, index, size = 112) {
  return `${API}/faces/crop/${filepath}?index=${index}&size=${size}`;
}

const key = (f) => `${f.filepath}#${f.face_index}`;

export default function PersonReview({ person, people, onChanged, onClose }) {
  const [faces, setFaces] = useState(null);
  const [threshold, setThreshold] = useState(0.4);
  const [order, setOrder] = useState("alike");
  const [selected, setSelected] = useState(() => new Set());
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState("");
  const [moveTo, setMoveTo] = useState("");
  const [newName, setNewName] = useState("");
  const [timeline, setTimeline] = useState(false);
  const anchor = useRef(null);
  const name = person.name || "this person";

  const load = useCallback(async () => {
    try {
      const res = await fetch(`${API}/faces/people/${person.id}/faces`);
      if (!res.ok) { setFaces([]); return; }
      const data = await res.json();
      setThreshold(data.threshold ?? 0.4);
      setFaces(data.faces || []);
    } catch { setFaces([]); }
  }, [person.id]);

  useEffect(() => { setSelected(new Set()); setFaces(null); load(); }, [load]);

  useEffect(() => {
    const onKey = (e) => {
      if (e.target.closest?.("input, select, textarea") || timeline) return;
      if (e.key === "Escape") {
        if (selected.size) setSelected(new Set()); else onClose();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selected, onClose, timeline]);

  // A face this close to the bar got in by a whisker; worth a look.
  const doubtfulBelow = threshold + 0.08;
  const isDoubtful = (f) => f.similarity != null && f.similarity < doubtfulBelow;

  const shown = useMemo(() => {
    if (!faces) return [];
    if (order === "alike") return faces;
    return [...faces].sort((a, b) => (a.filepath < b.filepath ? -1 : a.filepath > b.filepath ? 1 : a.face_index - b.face_index));
  }, [faces, order]);
  const doubtful = useMemo(() => shown.filter(isDoubtful), [shown, threshold]); // eslint-disable-line react-hooks/exhaustive-deps

  const click = (face, index, e) => {
    const k = key(face);
    // Read the anchor now: the updater may run after it has moved to this face.
    const from = anchor.current;
    setSelected((prev) => {
      const next = new Set(prev);
      if (e.shiftKey && from != null) {
        const [a, b] = [from, index].sort((x, y) => x - y);
        for (let i = a; i <= b; i++) next.add(key(shown[i]));
      } else if (next.has(k)) next.delete(k);
      else next.add(k);
      return next;
    });
    anchor.current = index;
  };

  const picked = () => shown.filter((f) => selected.has(key(f))).map((f) => ({ filepath: f.filepath, face_index: f.face_index }));

  const act = async (path, body, done) => {
    setBusy(true);
    setNote("");
    try {
      const res = await fetch(`${API}${path}`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
      });
      const out = await res.json();
      if (!res.ok) { setNote(out.detail || "Not saved"); return; }
      setNote(done(out));
      setSelected(new Set());
      setMoveTo("");
      setNewName("");
      await load();
      onChanged();
    } catch { setNote("The backend is not reachable"); } finally { setBusy(false); }
  };

  const removeFaces = (list) => act("/faces/remove", { faces: list },
    (o) => `${o.removed} face${o.removed === 1 ? "" : "s"} taken out of ${name} — they will not be put back`);
  const moveFaces = (list) => {
    const into = Number(moveTo) || 0;
    const nm = moveTo === "new" ? newName.trim() : "";
    if (!into && !nm) return;
    const target = into ? (people.find((p) => p.id === into)?.name || "that person") : nm;
    act("/faces/move", { faces: list, into, name: nm },
      (o) => `${o.moved} face${o.moved === 1 ? "" : "s"} moved to ${target}`);
  };

  const others = useMemo(
    () => people.filter((p) => p.id !== person.id && p.name).sort((a, b) => a.name.localeCompare(b.name)),
    [people, person.id]);

  return (
    <div data-modal="person" className="fixed inset-0 z-[1000] bg-black/80 flex items-stretch sm:items-start justify-center sm:p-4 overflow-auto"
      onClick={onClose}>
      <div className="w-full max-w-6xl bg-neutral-950 sm:border border-neutral-800 sm:rounded-lg flex flex-col min-h-full sm:min-h-0"
        onClick={(e) => e.stopPropagation()}>
        <div className="sticky top-0 z-10 bg-neutral-950/95 backdrop-blur border-b border-neutral-800 px-4 py-3 space-y-2">
          <div className="flex items-center gap-3 flex-wrap">
            <h3 className="text-lg font-medium">{person.name || "Unnamed group"}</h3>
            <span className="text-xs text-neutral-500">
              {faces ? `${faces.length} face${faces.length === 1 ? "" : "s"}` : "Loading…"}
              {doubtful.length > 0 && <span className="text-amber-400"> · {doubtful.length} worth a look</span>}
            </span>
            <div className="ml-auto flex items-center gap-2">
              <div className="flex rounded border border-neutral-800 overflow-hidden text-xs">
                {[["alike", "Least alike first"], ["photo", "By photo"]].map(([k, label]) => (
                  <button key={k} onClick={() => setOrder(k)}
                    className={`px-2 py-1 ${order === k ? "bg-neutral-800 text-white" : "text-neutral-500 hover:text-neutral-300"}`}>
                    {label}
                  </button>
                ))}
              </div>
              <button onClick={() => setTimeline(true)} data-testid="open-timeline"
                title="Their best face for every month they were photographed, played as a flip book"
                className="px-2 py-1 rounded border border-neutral-700 text-xs text-neutral-200 hover:border-amber-600">
                ◷ Timeline
              </button>
              <button onClick={onClose} className="px-2 py-1 text-sm text-neutral-400 hover:text-white">Close</button>
            </div>
          </div>
          <p className="text-xs text-neutral-500">
            Tap faces that are not {name} to select them (Shift-click selects a run). Faces with an amber edge
            only just matched. Taking a face out is remembered: no rescan or regroup puts it back.
          </p>

          <div className={`flex items-center gap-2 flex-wrap text-sm ${busy ? "opacity-50 pointer-events-none" : ""}`}>
            <span className="text-neutral-300">{selected.size} selected</span>
            {doubtful.length > 0 && (
              <button onClick={() => setSelected(new Set(doubtful.map(key)))}
                className="px-2 py-1 rounded border border-amber-800/70 text-amber-300 text-xs hover:border-amber-600">
                Select the {doubtful.length} worth a look
              </button>
            )}
            {selected.size > 0 && (
              <>
                <button onClick={() => setSelected(new Set())} className="text-xs text-neutral-500 hover:text-white">Clear</button>
                <button onClick={() => removeFaces(picked())}
                  className="px-2.5 py-1 rounded bg-red-900/60 text-red-100 hover:bg-red-800">
                  Not {name}
                </button>
                <span className="text-neutral-600">or it is</span>
                <select value={moveTo} onChange={(e) => setMoveTo(e.target.value)}
                  className="bg-neutral-900 border border-neutral-700 rounded px-2 py-1 text-base sm:text-sm text-neutral-200">
                  <option value="">someone else…</option>
                  {others.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
                  <option value="new">a new person…</option>
                </select>
                {moveTo === "new" && (
                  <input value={newName} onChange={(e) => setNewName(e.target.value)} placeholder="Their name"
                    onKeyDown={(e) => { if (e.key === "Enter") moveFaces(picked()); }}
                    className="bg-neutral-900 border border-neutral-700 rounded px-2 py-1 text-base sm:text-sm text-neutral-200 w-36" />
                )}
                {moveTo && (moveTo !== "new" || newName.trim()) && (
                  <button onClick={() => moveFaces(picked())}
                    className="px-2.5 py-1 rounded bg-sky-900/70 text-sky-100 hover:bg-sky-800">
                    Move {selected.size}
                  </button>
                )}
              </>
            )}
            {note && <span className="text-xs text-emerald-400 w-full sm:w-auto">{note}</span>}
          </div>
        </div>

        <div className="p-3 grid gap-2"
          style={{ gridTemplateColumns: "repeat(auto-fill, minmax(min(104px, 30vw), 1fr))" }}>
          {shown.map((f, i) => {
            const on = selected.has(key(f));
            const doubt = isDoubtful(f);
            return (
              <div key={key(f)} className="group relative">
                <button onClick={(e) => click(f, i, e)}
                  title={`${f.filepath.split("/").pop()}${f.similarity != null ? ` · alike ${f.similarity.toFixed(2)}` : " · older face model"}`}
                  className={`block w-full aspect-square rounded overflow-hidden border-2 transition-colors ${
                    on ? "border-sky-400" : doubt ? "border-amber-500/80" : "border-transparent hover:border-neutral-600"}`}>
                  <img src={faceCropUrl(f.filepath, f.face_index)} loading="lazy" alt=""
                    className={`w-full h-full object-cover ${on ? "opacity-70" : ""}`} />
                  {on && <span className="absolute top-1 left-1 w-5 h-5 rounded-full bg-sky-400 text-black text-xs flex items-center justify-center">✓</span>}
                </button>
                <button onClick={() => removeFaces([{ filepath: f.filepath, face_index: f.face_index }])}
                  disabled={busy} title={`Not ${name}`} aria-label={`Not ${name}`}
                  className="absolute top-1 right-1 w-6 h-6 rounded-full bg-black/70 text-neutral-200 text-xs opacity-0 group-hover:opacity-100 [@media(hover:none)]:opacity-100 hover:bg-red-800 transition-opacity">
                  ✕
                </button>
                <p className="mt-0.5 text-[10px] text-neutral-600 truncate">
                  {f.similarity != null ? f.similarity.toFixed(2) : "old model"} · {f.filepath.split("/").slice(-2, -1)[0]}
                </p>
              </div>
            );
          })}
          {faces && faces.length === 0 && <p className="text-sm text-neutral-500">No faces left in this group.</p>}
        </div>
      </div>
      {timeline && <PersonTimeline person={person} onClose={() => setTimeline(false)} />}
    </div>
  );
}
