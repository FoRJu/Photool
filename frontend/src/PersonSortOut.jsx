import { useState, useEffect } from "react";

// ── Sorting out one person ──────────────────────────────────────────
// A named person can turn out to be several: children of one age at one
// event look alike to the recogniser, and once a few of them shared a name
// the group drifted to take in the rest. So the person's faces are split
// into look-alike sets, and the question is "which of these is them?". The
// sets left unticked leave: clear ones remembered as "not them", so no
// regroup brings them back; the too-unclear ones only let go.

const API = "/api";
const STRIP = 12;

function faceCropUrl(filepath, index, size = 96) {
  return `${API}/faces/crop/${filepath}?index=${index}&size=${size}`;
}

const plural = (n, word) => `${n.toLocaleString()} ${word}${n === 1 ? "" : "s"}`;

export default function PersonSortOut({ person, onChanged, onClose }) {
  const [sets, setSets] = useState(null);
  const [keep, setKeep] = useState(() => new Set());
  const [open, setOpen] = useState(() => new Set());
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState("");

  useEffect(() => {
    let live = true;
    fetch(`${API}/faces/people/${person.id}/sort`).then((r) => r.json()).then((d) => {
      if (!live) return;
      const list = d.sets || [];
      setSets(list);
      // Faces drawn by hand were put here by a person: kept unless unticked.
      setKeep(new Set(list.filter((s) => s.kind === "drawn").map((s) => s.key)));
    }).catch(() => setNote("The backend is not reachable"));
    return () => { live = false; };
  }, [person.id]);

  const toggle = (setter, key) => setter((cur) => {
    const next = new Set(cur);
    if (next.has(key)) next.delete(key); else next.add(key);
    return next;
  });

  const kept = (sets || []).filter((s) => keep.has(s.key));
  const leaving = (sets || []).filter((s) => !keep.has(s.key));
  const keptFaces = kept.reduce((n, s) => n + s.count, 0);
  const leavingFaces = leaving.reduce((n, s) => n + s.count, 0);
  const anyClearKept = kept.some((s) => s.kind === "set");

  const apply = async () => {
    if (!window.confirm(`Keep ${plural(keptFaces, "face")} as ${person.name}? `
      + `The other ${plural(leavingFaces, "face")} leave, and the clear ones are remembered as not ${person.name}.`)) return;
    const refs = (list) => list.flatMap((s) => s.faces.map((f) => ({ filepath: f.filepath, face_index: f.face_index })));
    setBusy(true);
    setNote("");
    try {
      const res = await fetch(`${API}/faces/people/${person.id}/sort`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          not_them: refs(leaving.filter((s) => s.kind !== "unclear")),
          unclear: refs(leaving.filter((s) => s.kind === "unclear")),
        }),
      });
      const out = await res.json();
      if (!res.ok) { setNote(out.detail || "Not saved"); return; }
      // The faces that left are grouped among the rest, so they show up as
      // people of their own to name.
      setNote("Saved. Regrouping the faces that left…");
      await fetch(`${API}/faces/people/rebuild`, { method: "POST",
        headers: { "Content-Type": "application/json" }, body: JSON.stringify({}) });
      await onChanged();
      onClose();
    } catch { setNote("The backend is not reachable"); } finally { setBusy(false); }
  };

  const title = (s, i) => (s.kind === "unclear" ? "Too unclear to place"
    : s.kind === "drawn" ? "Drawn by hand" : `Set ${i + 1}`);

  return (
    <div data-modal="sort-out" className="fixed inset-0 z-[1000] bg-black/80 flex items-stretch sm:items-start justify-center sm:p-4 overflow-auto"
      onClick={busy ? undefined : onClose}>
      <div className="w-full max-w-6xl bg-neutral-950 sm:border border-neutral-800 sm:rounded-lg flex flex-col min-h-full sm:min-h-0"
        onClick={(e) => e.stopPropagation()}>
        <div className="sticky top-0 z-10 bg-neutral-950/95 backdrop-blur border-b border-neutral-800 px-4 py-3 space-y-2">
          <div className="flex items-center gap-3 flex-wrap">
            <h3 className="text-lg font-medium">Sort out {person.name}</h3>
            <span className="text-xs text-neutral-500">
              {sets ? `${plural(sets.length, "set")} of look-alike faces` : "Sorting…"}
            </span>
            <div className="ml-auto flex items-center gap-2">
              <button onClick={apply} disabled={busy || !anyClearKept} data-testid="sort-apply"
                title={anyClearKept ? "" : `Tick at least one set that is ${person.name}`}
                className="px-3 py-1 rounded text-sm bg-amber-600/80 hover:bg-amber-600 text-white disabled:opacity-40">
                {busy ? "Saving…" : `Keep ${plural(keptFaces, "face")}`}
              </button>
              <button onClick={onClose} disabled={busy} className="px-2 py-1 text-sm text-neutral-400 hover:text-white">Cancel</button>
            </div>
          </div>
          <p className="text-xs text-neutral-500">
            Tick every set that is {person.name}. The sets left unticked leave {person.name}: the clear ones are
            remembered as someone else, so a regroup does not bring them back; the unclear ones are only let go.
          </p>
          {note && <p className="text-xs text-emerald-400">{note}</p>}
        </div>

        <div className="p-3 space-y-2">
          {(sets || []).map((s, i) => {
            const on = keep.has(s.key);
            const all = open.has(s.key);
            const faces = all ? s.faces : s.faces.slice(0, STRIP);
            return (
              <div key={s.key} data-testid="sort-set"
                className={`rounded-lg border p-2 ${on ? "border-emerald-600/70 bg-emerald-950/20" : "border-neutral-800"}`}>
                <div className="flex items-center gap-3 mb-2">
                  <label className="flex items-center gap-2 cursor-pointer select-none">
                    <input type="checkbox" checked={on} onChange={() => toggle(setKeep, s.key)}
                      className="w-4 h-4 accent-emerald-500" />
                    <span className="text-sm text-neutral-200">{on ? `${person.name}` : `Is ${person.name}?`}</span>
                  </label>
                  <span className="text-xs text-neutral-500">
                    {title(s, i)} · {plural(s.count, "face")}
                    {s.kind === "set" && s.clear < s.count ? ` · ${s.clear} clear` : ""}
                  </span>
                  {s.faces.length > STRIP && (
                    <button onClick={() => toggle(setOpen, s.key)} className="ml-auto text-xs text-neutral-500 hover:text-neutral-200">
                      {all ? "Fewer" : `All ${s.faces.length}`}
                    </button>
                  )}
                </div>
                <div className="grid gap-1" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(72px, 1fr))" }}>
                  {faces.map((f) => (
                    <img key={`${f.filepath}#${f.face_index}`} src={faceCropUrl(f.filepath, f.face_index)} loading="lazy"
                      alt="" title={f.filepath.split("/").slice(-2).join("/")}
                      onClick={() => toggle(setKeep, s.key)}
                      className={`w-full aspect-square object-cover rounded cursor-pointer ${on ? "" : "opacity-80"}`} />
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
