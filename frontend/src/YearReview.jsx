import { useState, useEffect, useMemo, useCallback } from "react";
import Slideshow from "./Slideshow.jsx";
import PersonTimeline from "./PersonTimeline.jsx";
import CollageMaker from "./CollageMaker.jsx";
import { recall, remember } from "./remembered.js";

// ── Year in review ──────────────────────────────────────────────────
// One page per year (backend catalog/review.py): the year in numbers, its best
// frames, the people in it, its biggest shoots and the places away from home,
// and a collage of the best frames to download. The frames are the server's
// pick; which of them go in the collage is the person's.

const API = "/api";
const MONTHS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"];
const MAX_COLLAGE = 30;

function thumbUrl(filepath, size = "grid") {
  const cut = filepath.lastIndexOf("/");
  return `${API}/thumb/${filepath.slice(0, cut)}/${encodeURIComponent(filepath.slice(cut + 1))}?size=${size}`;
}

function faceUrl(p) {
  return `${API}/faces/crop/${p.filepath}?index=${p.face_index}&size=160`;
}

function dayText(day, opts = { day: "numeric", month: "long" }) {
  if (!day) return "";
  const [y, m, d] = day.split("-").map(Number);
  return new Date(y, m - 1, d).toLocaleDateString(undefined, opts);
}

function span(m) {
  if (!m.date_from) return "";
  if (!m.date_to || m.date_to === m.date_from) return dayText(m.date_from);
  return `${dayText(m.date_from)} – ${dayText(m.date_to)}`;
}

const plural = (n, word) => `${n.toLocaleString()} ${word}${n === 1 ? "" : "s"}`;

// A person's name against a perspective's people: exact, or "Prefix*".
function nameMatches(name, patterns) {
  const low = name.toLowerCase();
  return patterns.some((p) => {
    const q = p.trim().toLowerCase();
    return q.endsWith("*") ? low.startsWith(q.slice(0, -1)) : low === q;
  });
}

// ── Perspectives ────────────────────────────────────────────────────
// Family, School, Friends: a name and the people it is about (backend
// catalog/perspectives.py). A year seen from one is the shoots those people are
// most of. Shared by the household; which one a device opens on is its own.
function PerspectiveEditor({ info, onClose, onSaved }) {
  const [editing, setEditing] = useState(null);     // {id?, name, people}
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState("");
  const people = info?.people || [];
  // Name prefixes several people share, offered as one pick ("BAND*").
  const prefixes = useMemo(() => {
    const counts = new Map();
    for (const n of people) {
      const m = /^([A-Za-z0-9#]+)[ -]/.exec(n);
      if (m) counts.set(m[1], (counts.get(m[1]) || 0) + 1);
    }
    return [...counts.entries()].filter(([, c]) => c >= 2).map(([p, c]) => ({ pattern: `${p}*`, count: c }));
  }, [people]);

  const call = async (method, path, body) => {
    setBusy(true);
    setNote("");
    try {
      const res = await fetch(`${API}${path}`, {
        method, headers: body ? { "Content-Type": "application/json" } : undefined,
        body: body ? JSON.stringify(body) : undefined,
      });
      const out = await res.json().catch(() => ({}));
      if (!res.ok) { setNote(out.detail || "Not saved"); return null; }
      await onSaved(out);
      return out;
    } catch { setNote("The backend is not reachable"); return null; } finally { setBusy(false); }
  };

  const save = async () => {
    const body = { name: editing.name.trim(), people: editing.people };
    const out = editing.id ? await call("PUT", `/library/perspectives/${editing.id}`, body)
      : await call("POST", "/library/perspectives", body);
    if (out) setEditing(null);
  };
  const toggle = (pattern) => setEditing((e) => ({
    ...e, people: e.people.includes(pattern) ? e.people.filter((p) => p !== pattern) : [...e.people, pattern],
  }));
  const covered = (name) => editing && nameMatches(name, editing.people.filter((p) => p.endsWith("*")));

  return (
    <div className="fixed inset-0 z-[1100] bg-black/80 flex items-start justify-center p-2 sm:p-6 overflow-auto"
      onClick={(e) => { e.stopPropagation(); onClose(); }} data-testid="perspective-editor">
      <div className="w-full max-w-2xl bg-neutral-950 border border-neutral-800 rounded-lg p-4 space-y-4" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center gap-3">
          <h3 className="text-lg font-medium">Perspectives</h3>
          <span className="text-xs text-neutral-500">who a year is seen through: a shoot belongs where its people are most of it</span>
          <button onClick={onClose} className="ml-auto text-sm text-neutral-400 hover:text-white">Close</button>
        </div>
        {!editing && (
          <>
            <div className="space-y-1.5">
              {(info?.perspectives || []).map((p) => (
                <div key={p.id} className="flex items-center gap-2 text-sm">
                  <span className="text-neutral-200 w-32 truncate">{p.name}</span>
                  <span className="flex-1 text-xs text-neutral-500 truncate">{p.people.join(", ")}</span>
                  <button onClick={() => setEditing({ ...p })} className="text-xs text-sky-300 hover:text-sky-200">edit</button>
                  <button disabled={busy} onClick={() => call("DELETE", `/library/perspectives/${p.id}`)}
                    className="text-xs text-neutral-500 hover:text-red-400">delete</button>
                </div>
              ))}
              {!(info?.perspectives || []).length && <p className="text-sm text-neutral-500">None yet.</p>}
            </div>
            {(info?.suggestions || []).length > 0 && (
              <div className="space-y-1">
                <p className="text-xs text-neutral-500">Suggested from the people you have named — tap to start from one:</p>
                <div className="flex gap-2 flex-wrap">
                  {info.suggestions.map((sg) => (
                    <button key={sg.name} onClick={() => setEditing({ name: sg.name, people: sg.people })} data-testid="perspective-suggestion"
                      title={sg.people.join(", ")}
                      className="px-2 py-1 rounded border border-neutral-700 text-sm text-neutral-200 hover:border-amber-600">＋ {sg.name}</button>
                  ))}
                </div>
              </div>
            )}
            <button onClick={() => setEditing({ name: "", people: [] })}
              className="px-2.5 py-1 rounded bg-neutral-800 hover:bg-neutral-700 text-sm">＋ New perspective</button>
          </>
        )}
        {editing && (
          <div className="space-y-3">
            <input value={editing.name} onChange={(e) => setEditing((x) => ({ ...x, name: e.target.value }))} autoFocus
              placeholder="Family, School, Friends…" data-testid="perspective-name"
              className="w-full bg-neutral-900 border border-neutral-700 rounded px-2 py-1 text-base sm:text-sm text-neutral-200" />
            {prefixes.length > 0 && (
              <div className="space-y-1">
                <p className="text-xs text-neutral-500">Everyone whose name starts with…</p>
                <div className="flex gap-1.5 flex-wrap">
                  {prefixes.map(({ pattern, count }) => (
                    <button key={pattern} onClick={() => toggle(pattern)}
                      className={`px-2 py-0.5 rounded border text-xs ${editing.people.includes(pattern)
                        ? "bg-amber-700/40 border-amber-600 text-amber-100" : "border-neutral-700 text-neutral-400 hover:text-neutral-200"}`}>
                      {pattern} <span className="text-neutral-500">{count}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}
            <div className="space-y-1">
              <p className="text-xs text-neutral-500">People</p>
              <div className="flex gap-1.5 flex-wrap max-h-56 overflow-auto">
                {people.map((n) => {
                  const on = editing.people.includes(n) || covered(n);
                  return (
                    <button key={n} onClick={() => !covered(n) && toggle(n)} disabled={covered(n)}
                      className={`px-2 py-0.5 rounded border text-xs ${on
                        ? "bg-amber-700/40 border-amber-600 text-amber-100" : "border-neutral-700 text-neutral-400 hover:text-neutral-200"} ${covered(n) ? "opacity-60" : ""}`}>
                      {n}
                    </button>
                  );
                })}
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button onClick={save} disabled={busy || !editing.name.trim() || !editing.people.length} data-testid="perspective-save"
                className="px-2.5 py-1 rounded bg-amber-600/80 hover:bg-amber-500 text-black text-sm disabled:opacity-40">Save</button>
              <button onClick={() => setEditing(null)} className="text-sm text-neutral-400 hover:text-white">Cancel</button>
            </div>
          </div>
        )}
        {note && <p className="text-xs text-red-400">{note}</p>}
      </div>
    </div>
  );
}

/** The years to choose from, as a row under "On this day". */
export function YearsStrip({ onOpenYear }) {
  const [years, setYears] = useState(() => recall("years") || null);
  useEffect(() => {
    let live = true;
    fetch(`${API}/library/years`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (!live || !d) return;
        setYears(d.years || []);
        remember("years", d.years || []);
      })
      .catch(() => { if (live) setYears((y) => y || []); });
    return () => { live = false; };
  }, []);
  if (!years || years.length === 0) return null;
  return (
    <section className="flex items-baseline gap-2 flex-wrap" data-testid="years-strip">
      <h2 className="text-sm font-medium text-neutral-200 mr-1">Year in review</h2>
      {years.map((y) => (
        <button key={y.year} onClick={() => onOpenYear(y.year)} title={plural(y.photos, "photo")}
          className="px-2 py-0.5 rounded border border-neutral-800 text-xs text-neutral-300 hover:border-amber-600 hover:text-white">
          {y.year}
        </button>
      ))}
    </section>
  );
}

export default function YearReview({ year, onClose, onOpenFolder }) {
  const [data, setData] = useState(null);
  const [inCollage, setInCollage] = useState(() => new Set());
  const [slides, setSlides] = useState(null);       // { start }
  const [person, setPerson] = useState(null);
  const [making, setMaking] = useState(false);
  const [note, setNote] = useState("");
  // The perspectives, and which one the page is seen through (null until this
  // device's default is known; 0 is everything).
  const [persp, setPersp] = useState(null);
  const [pid, setPid] = useState(null);
  const [managing, setManaging] = useState(false);
  const [printing, setPrinting] = useState(false);

  const loadPerspectives = useCallback(async () => {
    try {
      const res = await fetch(`${API}/library/perspectives`);
      const d = res.ok ? await res.json() : null;
      setPersp(d || { perspectives: [], default: 0, people: [], suggestions: [] });
      return d;
    } catch {
      setPersp({ perspectives: [], default: 0, people: [], suggestions: [] });
      return null;
    }
  }, []);
  useEffect(() => {
    loadPerspectives().then((d) => setPid(d?.default || 0));
  }, [loadPerspectives]);
  const makeDefault = async (id) => {
    try {
      const res = await fetch(`${API}/library/perspectives-default`, {
        method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ id }),
      });
      if (res.ok) {
        const out = await res.json();
        setPersp((p) => ({ ...p, default: out.default }));
      }
    } catch { /* stays as it was */ }
  };
  const seenAs = persp?.perspectives.find((p) => p.id === pid)?.name || "";

  useEffect(() => {
    if (pid == null) return undefined;
    let live = true;
    setData(null);
    setNote("");
    fetch(`${API}/library/year/${year}?perspective=${pid}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (!live) return;
        setData(d || { top: [], people: [], moments: [], places: [], collage: [], summary: null });
        setInCollage(new Set(d?.collage || []));
      })
      .catch(() => { if (live) setData({ top: [], people: [], moments: [], places: [], collage: [], summary: null }); });
    return () => { live = false; };
  }, [year, pid]);

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "Escape" && !slides && !person && !managing && !printing) { e.stopPropagation(); onClose(); }
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [onClose, slides, person, managing, printing]);

  const s = data?.summary;
  const top = data?.top || [];
  const biggestMonth = s ? Math.max(1, ...s.months) : 1;
  // The collage in the order taken, as the page shows them.
  const collageList = useMemo(() => top.map((p) => p.filepath).filter((fp) => inCollage.has(fp)), [top, inCollage]);

  const toggle = (fp) => setInCollage((prev) => {
    const next = new Set(prev);
    if (next.has(fp)) next.delete(fp);
    else if (next.size < MAX_COLLAGE) next.add(fp);
    return next;
  });

  const makeCollage = async () => {
    if (!collageList.length) return;
    setMaking(true);
    setNote("");
    try {
      const subtitle = s ? `${plural(s.photos, "photo")} · ${plural(s.days, "day")}` : "";
      const res = await fetch(`${API}/library/collage`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filepaths: collageList, title: seenAs ? `${year} · ${seenAs}` : String(year), subtitle }),
      });
      if (!res.ok) { setNote((await res.json().catch(() => ({}))).detail || "The collage could not be made"); return; }
      const url = URL.createObjectURL(await res.blob());
      const a = document.createElement("a");
      a.href = url;
      a.download = `Photool ${year}${seenAs ? ` ${seenAs}` : ""}.jpg`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 10000);
      setNote(`Saved Photool ${year}${seenAs ? ` ${seenAs}` : ""}.jpg to your downloads`);
    } catch { setNote("The backend is not reachable"); } finally { setMaking(false); }
  };

  return (
    <div data-modal="year" className="fixed inset-0 z-[1000] bg-black/85 flex items-stretch sm:items-start justify-center sm:p-4 overflow-auto"
      onClick={onClose}>
      <div className="w-full max-w-6xl bg-neutral-950 sm:border border-neutral-800 sm:rounded-lg flex flex-col min-h-full sm:min-h-0"
        onClick={(e) => e.stopPropagation()}>
        <div className="sticky top-0 z-10 bg-neutral-950/95 backdrop-blur border-b border-neutral-800 px-4 py-3 flex items-center gap-3 flex-wrap">
          <h2 className="text-3xl font-light tracking-tight text-white">{year}</h2>
          {persp && (
            <div className="flex items-center gap-1.5">
              <select value={pid ?? 0} data-testid="perspective-select"
                onChange={(e) => { if (e.target.value === "manage") setManaging(true); else setPid(+e.target.value); }}
                className="bg-neutral-900 border border-neutral-700 rounded px-2 py-1 text-base sm:text-sm text-neutral-200">
                <option value={0}>Everything</option>
                {persp.perspectives.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
                <option value="manage">{persp.perspectives.length ? "Edit perspectives…" : "＋ Family, School…"}</option>
              </select>
              {pid != null && (persp.default === pid ? (
                <span className="text-[11px] text-amber-300/80" title="Year in review opens on this here">★ opens here</span>
              ) : (
                <button onClick={() => makeDefault(pid)} data-testid="perspective-default"
                  title="Open Year in review on this, on this device"
                  className="text-[11px] text-neutral-500 hover:text-neutral-300">☆ open on this here</button>
              ))}
            </div>
          )}
          {s && (
            <span className="text-sm text-neutral-400">
              {plural(s.photos, "photo")} · {plural(s.days, "day")} · {plural(s.shoots, "shoot")}
            </span>
          )}
          <div className="ml-auto flex items-center gap-2">
            {top.length > 1 && (
              <button onClick={() => setSlides({ start: 0 })} data-testid="year-play"
                className="px-2.5 py-1 rounded bg-white/10 hover:bg-white/20 text-sm text-neutral-100">▶ Play the year</button>
            )}
            <button onClick={onClose} className="px-2 py-1 text-sm text-neutral-400 hover:text-white">Close</button>
          </div>
        </div>

        {!data && <p className="p-4 text-sm text-neutral-500">Looking back over {year}…</p>}
        {data && !s && <p className="p-4 text-sm text-neutral-500">Nothing to show for {year}.</p>}
        {s && (
          <div className="p-4 space-y-6">
            {/* When: the months, and the busiest day. */}
            <section className="flex items-end gap-6 flex-wrap">
              <div className="flex items-end gap-1 h-20" title="Photos per month" data-testid="year-months">
                {s.months.map((n, i) => (
                  <div key={i} className="flex flex-col items-center gap-1 w-5">
                    <div className={`w-full rounded-sm ${n ? "bg-amber-500/80" : "bg-neutral-800"}`}
                      style={{ height: `${n ? Math.max(4, (n / biggestMonth) * 64) : 2}px` }}
                      title={`${new Date(year, i, 1).toLocaleDateString(undefined, { month: "long" })}: ${plural(n, "photo")}`} />
                    <span className="text-[10px] text-neutral-500">{MONTHS[i]}</span>
                  </div>
                ))}
              </div>
              <div className="text-sm text-neutral-400 space-y-0.5">
                {s.busiest_day && (
                  <p>Busiest day: <button onClick={() => onOpenFolder(s.busiest_day.folder)}
                    className="text-neutral-200 hover:text-white underline decoration-neutral-700">
                    {dayText(s.busiest_day.date, { weekday: "long", day: "numeric", month: "long" })}</button>
                    {" "}({plural(s.busiest_day.photos, "photo")})</p>
                )}
                {s.cameras.length > 0 && (
                  <p>{s.cameras.map((c) => `${c.name} (${c.photos.toLocaleString()})`).join(" · ")}</p>
                )}
                {(s.picked > 0 || s.rated > 0 || s.edited > 0) && (
                  <p>{[s.picked && `${s.picked.toLocaleString()} picked`, s.rated && `${s.rated.toLocaleString()} rated`,
                    s.edited && `${s.edited.toLocaleString()} developed`].filter(Boolean).join(" · ")}</p>
                )}
              </div>
            </section>

            {/* The best frames, and which of them make the collage. */}
            {top.length > 0 && (
              <section className="space-y-2">
                <div className="flex items-baseline gap-3 flex-wrap">
                  <h3 className="text-sm font-medium text-neutral-200">The year's best</h3>
                  <span className="text-xs text-neutral-500">
                    picked from all {plural(s.photos, "photo")} — ✓ marks the {collageList.length} in the collage; tap ✓ to change
                  </span>
                  <button onClick={() => setPrinting(true)} disabled={collageList.length < 2}
                    title="The ✓ frames (up to 9) on a fixed page: a phone story, a square or a print"
                    className="ml-auto px-2.5 py-1 rounded border border-neutral-700 text-sm text-neutral-200 hover:border-amber-600 disabled:opacity-40">
                    ▦ Print layout…
                  </button>
                  <button onClick={makeCollage} disabled={making || !collageList.length} data-testid="year-collage"
                    className="px-2.5 py-1 rounded bg-amber-600/80 hover:bg-amber-500 text-black text-sm disabled:opacity-40">
                    {making ? "Making the collage…" : `⤓ Collage of ${collageList.length}`}
                  </button>
                </div>
                {note && <p className="text-xs text-emerald-400">{note}</p>}
                <div className="grid gap-1.5" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(min(150px, 30vw), 1fr))" }}>
                  {top.map((p, i) => {
                    const on = inCollage.has(p.filepath);
                    return (
                      <div key={p.filepath} className="relative group" data-testid="year-top">
                        <button onClick={() => setSlides({ start: i })} title={`${p.folder} · ${p.filename}`}
                          className="block w-full aspect-[4/3] rounded overflow-hidden bg-neutral-900">
                          <img src={thumbUrl(p.filepath)} loading="lazy" alt="" className="w-full h-full object-cover" />
                        </button>
                        <button onClick={() => toggle(p.filepath)} aria-pressed={on} data-testid="collage-toggle"
                          title={on ? "In the collage — tap to leave it out" : "Put it in the collage"}
                          className={`absolute top-1 right-1 w-6 h-6 rounded-full text-xs flex items-center justify-center ${on
                            ? "bg-amber-400 text-black" : "bg-black/60 text-neutral-300 opacity-0 group-hover:opacity-100 [@media(hover:none)]:opacity-100"}`}>✓</button>
                      </div>
                    );
                  })}
                </div>
              </section>
            )}

            {data.people.length > 0 && (
              <section className="space-y-2">
                <h3 className="text-sm font-medium text-neutral-200">Who was in it</h3>
                <div className="flex gap-3 overflow-x-auto pb-1">
                  {data.people.map((p) => (
                    <button key={p.id} onClick={() => setPerson({ id: p.id, name: p.name })}
                      title={`${p.name}'s timeline`} className="shrink-0 w-20 text-center group">
                      <img src={faceUrl(p)} loading="lazy" alt="" className="w-20 h-20 rounded-full object-cover bg-neutral-900 border border-neutral-800 group-hover:border-amber-600" />
                      <p className="mt-1 text-xs text-neutral-200 truncate">{p.name}</p>
                      <p className="text-[10px] text-neutral-500">{plural(p.photos, "photo")}</p>
                    </button>
                  ))}
                </div>
              </section>
            )}

            {data.moments.length > 0 && (
              <section className="space-y-2">
                <h3 className="text-sm font-medium text-neutral-200">The biggest shoots</h3>
                <div className="grid gap-2" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(min(220px, 44vw), 1fr))" }}>
                  {data.moments.map((m) => (
                    <button key={m.folder} onClick={() => onOpenFolder(m.folder)} className="text-left group" title={`Open ${m.folder}`}>
                      {m.cover && <img src={thumbUrl(m.cover)} loading="lazy" alt=""
                        className="w-full aspect-[3/2] object-cover rounded bg-neutral-900 border border-transparent group-hover:border-neutral-600" />}
                      <p className="mt-1 text-sm text-neutral-200 truncate">{m.title || span(m) || m.folder}</p>
                      <p className="text-xs text-neutral-500 truncate">
                        {[m.title && span(m), plural(m.photos, "photo"), m.place && `📍 ${m.place}`, m.people.join(", ")].filter(Boolean).join(" · ")}
                      </p>
                    </button>
                  ))}
                </div>
              </section>
            )}

            {data.places.length > 0 && (
              <section className="space-y-2">
                <h3 className="text-sm font-medium text-neutral-200">Away from home</h3>
                <div className="flex gap-2 flex-wrap">
                  {data.places.map((p) => (
                    <button key={p.label} onClick={() => onOpenFolder(p.folders[0])}
                      className="flex items-center gap-2 rounded border border-neutral-800 hover:border-neutral-600 pr-3">
                      {p.cover && <img src={thumbUrl(p.cover)} loading="lazy" alt="" className="w-14 h-10 object-cover rounded-l" />}
                      <span className="text-sm text-neutral-200">📍 {p.label}</span>
                      <span className="text-xs text-neutral-500">{plural(p.photos, "photo")}</span>
                    </button>
                  ))}
                </div>
              </section>
            )}
          </div>
        )}
      </div>
      {slides && top.length > 0 && (
        <div onClick={(e) => e.stopPropagation()}>
          <Slideshow photos={top} start={slides.start} title={`${year} · the year's best`} onClose={() => setSlides(null)} />
        </div>
      )}
      {person && (
        <div onClick={(e) => e.stopPropagation()}>
          <PersonTimeline person={person} onClose={() => setPerson(null)} />
        </div>
      )}
      {printing && (
        <div onClick={(e) => e.stopPropagation()}>
          <CollageMaker filepaths={collageList.slice(0, 9)} onClose={() => setPrinting(false)} />
        </div>
      )}
      {managing && (
        <PerspectiveEditor info={persp} onClose={() => setManaging(false)}
          onSaved={async (out) => {
            const d = await loadPerspectives();
            // A new perspective is shown at once; a deleted one falls back to everything.
            if (out?.id) setPid(out.id);
            else if (d && !d.perspectives.some((p) => p.id === pid)) setPid(0);
          }} />
      )}
    </div>
  );
}
