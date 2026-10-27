import { useState, useEffect, useCallback, useMemo, useRef } from "react";
import YearReview, { YearsStrip } from "./YearReview.jsx";
import { recall, remember } from "./remembered.js";
import ClockPanel from "./ClockPanel.jsx";

// ── The Library's front page ─────────────────────────────────────────
// Folders are filed by date (2026/2026-07-04), which says when and nothing
// else. A card says the rest: the best frame in it, the day spelled out, where
// it was, who is in it and what it was. Anything typed on a card wins over
// what was worked out. The map puts the same cards where they were taken.

const API = "/api";

// The country to name places from before the library knows where home is: the
// browser's own region (en-US → US).
function browserCountry() {
  try {
    const region = new Intl.Locale(navigator.language || "").maximize().region;
    return region || "";
  } catch { return ""; }
}
const VIEW_KEY = "photool.library.view";

function coverUrl(filepath, size = "grid") {
  if (!filepath) return "";
  const cut = filepath.lastIndexOf("/");
  return `${API}/thumb/${filepath.slice(0, cut)}/${encodeURIComponent(filepath.slice(cut + 1))}?size=${size}`;
}

function parseDay(day) {
  if (!day) return null;
  const [y, m, d] = day.split("-").map(Number);
  return new Date(y, m - 1, d);
}

// "Saturday, July 4, 2026", or "July 4 – 6, 2026" for a folder that spans days.
export function dateLabel(card) {
  const from = parseDay(card?.date_from);
  if (!from) return "";
  const to = parseDay(card.date_to) || from;
  if (to.getTime() === from.getTime()) {
    return from.toLocaleDateString(undefined, { weekday: "long", month: "long", day: "numeric", year: "numeric" });
  }
  const sameYear = from.getFullYear() === to.getFullYear();
  const sameMonth = sameYear && from.getMonth() === to.getMonth();
  const left = from.toLocaleDateString(undefined, { month: "long", day: "numeric", ...(sameYear ? {} : { year: "numeric" }) });
  const right = sameMonth
    ? `${to.getDate()}, ${to.getFullYear()}`
    : to.toLocaleDateString(undefined, { month: "long", day: "numeric", year: "numeric" });
  return `${left} – ${right}`;
}

// "graduation_2026_UW_4star" → "graduation 2026 UW 4star"
function humanize(folder) {
  const last = (folder || "").split("/").pop() || folder || "";
  return last.replace(/[_]+/g, " ").replace(/\s+/g, " ").trim();
}

function folderIsADate(folder) {
  return /(?:19|20)\d\d[-_.]?(?:0[1-9]|1[0-2])[-_.]?(?:0[1-9]|[12]\d|3[01])$/.test(folder || "");
}

// What a folder is called on its card and above its grid.
export function cardTitle(card, folder) {
  if (card?.note?.title) return card.note.title;
  if (card && folderIsADate(folder) && card.date_from) return dateLabel(card);
  if (!folderIsADate(folder)) return humanize(folder);
  return dateLabel(card) || folder;
}

function listNames(names, max = 3) {
  if (names.length <= max) {
    return names.length > 1 ? `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}` : names[0] || "";
  }
  const rest = names.length - max;
  return `${names.slice(0, max).join(", ")} and ${rest} other${rest === 1 ? "" : "s"}`;
}

// The line under the title, worked out when nobody wrote one.
export function cardSubtitle(card) {
  if (card?.note?.subtitle) return card.note.subtitle;
  if (!card) return "";
  const parts = [];
  if (card.place?.label && !card.at_home) parts.push(card.place.label);
  if (card.people?.length) parts.push(`with ${listNames(card.people.map((p) => p.name))}`);
  if (card.tags?.length) parts.push(card.tags.slice(0, 3).map((t) => t.name).join(", "));
  return parts.join(" · ");
}

function yearOf(card, folder) {
  if (card?.date_from) return card.date_from.slice(0, 4);
  const m = /^(?:19|20)\d\d/.exec(folder || "");
  return m ? m[0] : "Other";
}

function searchText(entry) {
  const { card, folder } = entry;
  return [
    folder, cardTitle(card, folder), cardSubtitle(card), dateLabel(card),
    card?.place?.label, ...(card?.place?.towns || []),
    ...(card?.people || []).map((p) => p.name), ...(card?.tags || []).map((t) => t.name),
  ].filter(Boolean).join(" ").toLowerCase();
}

// ── Reading locations ───────────────────────────────────────────────

function GpsLine({ status, onScan, onStop }) {
  if (!status) return null;
  const { gps, job } = status;
  if (job?.status === "running") {
    return (
      <p className="text-xs text-neutral-300">
        {job.phase === "names" ? "Getting the list of place names (once)…"
          : `Reading GPS · ${job.done.toLocaleString()} / ${job.total.toLocaleString()}`}
        {job.found > 0 && <span className="text-emerald-500"> · {job.found} found</span>}
        <button onClick={onStop} className="ml-2 text-neutral-500 hover:text-white">Stop</button>
      </p>
    );
  }
  return (
    <div className="text-xs text-neutral-500 space-y-1">
      <p>
        {gps.found > 0 ? `${gps.found.toLocaleString()} photos have GPS.` : "No photo read so far has GPS."}
        {" "}New photos are read as they are added.
      </p>
      {gps.unread > 0 && (
        <button onClick={onScan}
          title="Reads the start of every older photo for a GPS position. Takes a while on a large library."
          className="text-neutral-400 underline decoration-neutral-700 hover:text-white">
          Read GPS from {gps.unread.toLocaleString()} older photos
        </button>
      )}
      {job?.status === "failed" && <p className="text-red-400">{job.error}</p>}
    </div>
  );
}

function PlacesPanel({ home, homeCountry, status, onHome, onScan, onStop, onClose }) {
  return (
    <div className="absolute right-0 top-full mt-1 z-[1050] w-[min(360px,calc(100vw-32px))] bg-neutral-950 border border-neutral-800 rounded-lg shadow-2xl p-3 space-y-3">
      <div className="space-y-1">
        <div className="flex items-baseline justify-between">
          <span className="text-sm text-neutral-200">Home</span>
          <button onClick={onClose} className="text-xs text-neutral-500 hover:text-white">Close</button>
        </div>
        <p className="text-xs text-neutral-500">
          Folders with no place count as home. Cards only name a place when it is somewhere else.
        </p>
        <PlacePicker value={home} onChange={onHome} home={homeCountry} />
      </div>
      <div className="border-t border-neutral-800 pt-2">
        <GpsLine status={status} onScan={onScan} onStop={onStop} />
      </div>
    </div>
  );
}

// ── Editing a card ──────────────────────────────────────────────────

function PlacePicker({ value, onChange, home, suggestions = [] }) {
  const [text, setText] = useState("");
  const [hits, setHits] = useState([]);
  const [error, setError] = useState("");
  const seq = useRef(0);

  useEffect(() => {
    const q = text.trim();
    if (q.length < 2) { setHits([]); return; }
    const mine = ++seq.current;
    const timer = setTimeout(async () => {
      try {
        const res = await fetch(`${API}/places/search?q=${encodeURIComponent(q)}&home=${home || browserCountry()}`);
        const data = await res.json();
        if (mine !== seq.current) return;
        if (!res.ok) { setError(data.detail || "Place names are not available"); setHits([]); return; }
        setError("");
        setHits(data.places || []);
      } catch { if (mine === seq.current) setError("The backend is not reachable"); }
    }, 200);
    return () => clearTimeout(timer);
  }, [text, home]);

  if (value?.lat != null) {
    return (
      <div className="flex items-center gap-2 text-sm">
        <span className="text-neutral-200">📍 {value.place || `${value.lat.toFixed(3)}, ${value.lon.toFixed(3)}`}</span>
        <button onClick={() => onChange(null)} className="text-xs text-neutral-500 hover:text-red-400">clear</button>
      </div>
    );
  }
  return (
    <div className="relative">
      <input value={text} onChange={(e) => setText(e.target.value)}
        placeholder="Type a town: Seattle, or Portland, Maine"
        className="w-full bg-neutral-900 border border-neutral-700 rounded px-2 py-1 text-base sm:text-sm text-neutral-200 placeholder-neutral-600" />
      {error && <p className="text-xs text-amber-400 mt-1">{error}</p>}
      {suggestions.length > 0 && hits.length === 0 && (
        <div className="flex flex-wrap gap-1 mt-1.5">
          {suggestions.map((p) => (
            <button key={`${p.place}|${p.lat}`} onClick={() => onChange({ place: p.place, lat: p.lat, lon: p.lon })}
              className="text-xs px-2 py-1 rounded-full border border-neutral-700 text-neutral-300 hover:text-white hover:border-neutral-500">
              {p.home ? "⌂ " : ""}{p.place}
            </button>
          ))}
        </div>
      )}
      {hits.length > 0 && (
        <ul className="absolute z-[1100] left-0 right-0 mt-1 bg-neutral-900 border border-neutral-700 rounded shadow-xl max-h-56 overflow-auto">
          {hits.map((p, i) => (
            <li key={`${p.lat},${p.lon},${i}`}>
              <button onClick={() => { onChange({ place: p.label, lat: p.lat, lon: p.lon }); setText(""); setHits([]); }}
                className="w-full text-left px-2 py-1 text-sm hover:bg-neutral-800">
                <span className="text-neutral-200">{p.name}</span>
                <span className="text-neutral-500">{[p.region, p.country].filter((x) => x && x !== p.name).join(", ") ? `, ${[p.region, p.country].filter((x) => x && x !== p.name).join(", ")}` : ""}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function CardEditor({ entry, home, suggestions, onClose, onSaved }) {
  const { card, folder } = entry;
  const note = card?.note || {};
  const [title, setTitle] = useState(note.title || "");
  const [subtitle, setSubtitle] = useState(note.subtitle || "");
  const [cover, setCover] = useState(note.cover || "");
  const [takenOn, setTakenOn] = useState(note.taken_on || "");
  const [kind, setKind] = useState(note.kind || "");
  const [place, setPlace] = useState(note.lat != null ? { place: note.place, lat: note.lat, lon: note.lon } : null);
  const [covers, setCovers] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch(`${API}/library/cards/${folder}/covers`).then((r) => r.json())
      .then((d) => setCovers(d.filepaths || [])).catch(() => {});
  }, [folder]);

  useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const autoCard = card ? { ...card, note: null } : null;
  const save = async () => {
    setBusy(true);
    setError("");
    try {
      const res = await fetch(`${API}/library/cards/${folder}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title, subtitle, cover, place: place?.place || "", lat: place?.lat ?? null, lon: place?.lon ?? null,
          taken_on: takenOn, kind }),
      });
      if (!res.ok) { setError((await res.json()).detail || "Not saved"); return; }
      onSaved();
      onClose();
    } catch { setError("The backend is not reachable"); } finally { setBusy(false); }
  };

  return (
    <div data-modal="folder" className="fixed inset-0 z-[1000] bg-black/70 flex items-start justify-center p-4 overflow-auto" onClick={onClose}>
      <div className="w-full max-w-2xl bg-neutral-950 border border-neutral-800 rounded-lg p-4 space-y-3 mt-10"
        onClick={(e) => e.stopPropagation()}>
        <div className="flex items-baseline justify-between gap-2">
          <h3 className="text-lg font-medium">About this folder</h3>
          <span className="font-mono text-xs text-neutral-500 truncate">{folder}</span>
        </div>

        <label className="block space-y-1">
          <span className="text-xs text-neutral-500">Title</span>
          <input value={title} onChange={(e) => setTitle(e.target.value)} autoFocus
            placeholder={cardTitle(autoCard, folder)}
            className="w-full bg-neutral-900 border border-neutral-700 rounded px-2 py-1 text-base sm:text-sm text-neutral-200 placeholder-neutral-600" />
        </label>
        <label className="block space-y-1">
          <span className="text-xs text-neutral-500">What it was</span>
          <input value={subtitle} onChange={(e) => setSubtitle(e.target.value)}
            placeholder={cardSubtitle(autoCard) || "Grandma's 80th, first day of school, a weekend at the coast…"}
            className="w-full bg-neutral-900 border border-neutral-700 rounded px-2 py-1 text-base sm:text-sm text-neutral-200 placeholder-neutral-600" />
        </label>
        <div className="space-y-1">
          <span className="text-xs text-neutral-500">
            Where{card?.place?.source === "gps" && !place ? ` · from GPS: ${card.place.label || "a point on the map"}` : ""}
          </span>
          <PlacePicker value={place} onChange={setPlace} home={home} suggestions={suggestions} />
        </div>
        {/* A camera that lost its battery dates everything 0000 or 2009-01-01:
            the day it really was, for the photos whose clock gave none. */}
        <label className="flex items-center gap-2 flex-wrap">
          <span className="text-xs text-neutral-500">Taken on</span>
          <input type="date" value={takenOn} onChange={(e) => setTakenOn(e.target.value)}
            className="bg-neutral-900 border border-neutral-700 rounded px-2 py-1 text-sm text-neutral-200" />
          {takenOn && <button type="button" onClick={() => setTakenOn("")} className="text-xs text-neutral-500 hover:text-white">clear</button>}
          <span className="text-xs text-neutral-600">only for photos whose camera clock gave no date</span>
        </label>

        {/* What was shot decides how it is culled, by you or by an agent: a school
            event is culled to share, a family day keeps most frames of people. */}
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-xs text-neutral-500">Kind of shoot</span>
          {[["", "not said"], ["school", "School event — to share"], ["family", "Family"]].map(([value, label]) => (
            <button key={value || "none"} type="button" onClick={() => setKind(value)}
              className={`px-2 py-0.5 rounded border text-xs ${kind === value
                ? "border-amber-500 text-amber-300" : "border-neutral-800 text-neutral-500 hover:text-neutral-300"}`}>
              {label}
            </button>
          ))}
        </div>

        <div className="space-y-1">
          <span className="text-xs text-neutral-500">Cover</span>
          <div className="grid grid-cols-6 sm:grid-cols-8 gap-1">
            <button onClick={() => setCover("")} title="Let the best frame stand for the folder"
              className={`aspect-square rounded border text-[10px] ${cover === "" ? "border-amber-500 text-amber-300" : "border-neutral-800 text-neutral-500 hover:text-neutral-300"}`}>
              best frame
            </button>
            {covers.map((fp) => (
              <button key={fp} onClick={() => setCover(fp)} title={fp.split("/").pop()}
                className={`aspect-square rounded overflow-hidden border-2 ${cover === fp ? "border-amber-500" : "border-transparent hover:border-neutral-600"}`}>
                <img src={coverUrl(fp)} loading="lazy" alt="" className="w-full h-full object-cover" />
              </button>
            ))}
          </div>
        </div>

        {error && <p className="text-sm text-red-400">{error}</p>}
        <div className="flex justify-end gap-2 pt-1">
          <button onClick={onClose} className="px-3 py-1.5 rounded text-sm text-neutral-400 hover:text-white">Cancel</button>
          <button onClick={save} disabled={busy}
            className="px-3 py-1.5 rounded text-sm bg-amber-600/80 hover:bg-amber-600 text-white disabled:opacity-40">
            Save
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Cards ───────────────────────────────────────────────────────────

function FolderCard({ entry, onOpen, onEdit, selecting, selected, onToggle, demotion = null }) {
  const { card, folder, disk } = entry;
  const whole = demotion?.includes("*");
  const title = cardTitle(card, folder);
  const subtitle = cardSubtitle(card);
  const count = card?.count ?? disk?.unique_images ?? disk?.file_count ?? 0;
  const decided = card?.decided ?? disk?.decided ?? 0;
  const picks = card?.picks ?? disk?.picks ?? 0;
  const showDate = card?.date_from && title !== dateLabel(card);
  return (
    <div data-folder={folder} data-demoted={whole ? "folder" : demotion ? "camera" : undefined}
      className={`group relative rounded-lg overflow-hidden border bg-neutral-900/40 transition ${
      selected ? "border-amber-500 ring-2 ring-amber-500/60" : "border-neutral-800 hover:border-neutral-600"} ${
      whole && !selected ? "opacity-45 hover:opacity-100" : ""}`}>
      <button onClick={() => (selecting ? onToggle(folder) : onOpen(folder))} className="block w-full text-left">
        <div className="relative aspect-[4/3] bg-neutral-900">
          {card?.cover ? (
            <img src={coverUrl(card.cover)} loading="lazy" alt=""
              className="absolute inset-0 w-full h-full object-cover transition-transform duration-300 group-hover:scale-[1.03]" />
          ) : (
            <div className="absolute inset-0 flex items-center justify-center text-neutral-700 text-3xl">▦</div>
          )}
          <div className="absolute inset-x-0 bottom-0 h-2/3 bg-gradient-to-t from-black/85 via-black/40 to-transparent" />
          {card?.place?.label && !card.at_home && !selecting && (
            <span className="absolute top-2 left-2 max-w-[80%] truncate text-[11px] px-1.5 py-0.5 rounded bg-black/55 text-neutral-100 backdrop-blur-sm"
              title={card.place.source === "gps" ? `From GPS${card.place.towns?.length > 1 ? `: ${card.place.towns.join(", ")}` : ""}` : "Set by you"}>
              📍 {card.place.label}
            </span>
          )}
          <div className="absolute inset-x-0 bottom-0 p-2.5">
            {showDate && <p className="text-[11px] text-neutral-300/90">{dateLabel(card)}</p>}
            <p className="text-[15px] font-medium text-white leading-snug line-clamp-2 drop-shadow">{title}</p>
          </div>
        </div>
        <div className="px-2.5 py-2 space-y-1">
          <p className={`text-xs leading-snug line-clamp-2 min-h-[2lh] ${subtitle ? "text-neutral-300" : "text-neutral-600 italic"}`}>
            {subtitle || "Nothing known yet — add a line about it"}
          </p>
          <div className="flex items-center gap-2 text-[11px] text-neutral-500 whitespace-nowrap">
            <span>{count.toLocaleString()} photo{count === 1 ? "" : "s"}</span>
            {picks > 0 && <span className="text-green-600">{picks} picks</span>}
            {count > 0 && decided > 0 && (
              <span className="text-emerald-700">{Math.round((100 * decided) / count)}% reviewed</span>
            )}
            {demotion && (
              <span className="text-amber-500/70" title="Less important on this device">
                ↓ {whole ? "less important" : demotion.map((c) => c || "Unknown camera").join(", ")}
              </span>
            )}
            <span className="ml-auto min-w-0 font-mono text-neutral-600 truncate" title={folder}>{folder}</span>
          </div>
        </div>
      </button>
      {selecting ? (
        <span className={`pointer-events-none absolute top-2 left-2 w-7 h-7 rounded-full border-2 flex items-center justify-center text-sm ${
          selected ? "bg-amber-500 border-amber-500 text-black" : "bg-black/50 border-white/70 text-transparent"}`}>✓</span>
      ) : card && (
        // Always shown where there is no hover to reveal it: a phone.
        <button onClick={() => onEdit(entry)} title="Title, description, place and cover" aria-label="Edit folder"
          className="absolute top-2 right-2 w-9 h-9 sm:w-7 sm:h-7 rounded-full bg-black/60 text-neutral-200 opacity-0 group-hover:opacity-100 focus:opacity-100 [@media(hover:none)]:opacity-100 hover:bg-black/80 transition-opacity">
          ✎
        </button>
      )}
    </div>
  );
}

// ── Map ─────────────────────────────────────────────────────────────

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function pinHtml(cover, n, extra = "") {
  const bg = cover ? ` style="background-image:url('${escapeHtml(coverUrl(cover))}')"` : "";
  return `<div class="photool-pin${extra}"${bg}>${n > 1 ? `<span>${n}</span>` : ""}</div>`;
}

function FolderMap({ entries, home, onOpen, onEdit }) {
  const box = useRef(null);
  const mapRef = useRef(null);
  const layerRef = useRef(null);
  const [failed, setFailed] = useState("");
  const [ready, setReady] = useState(false);
  const handlers = useRef({ onOpen, onEdit });
  handlers.current = { onOpen, onEdit };

  // Folders taken at the same spot share a pin: twenty birthdays at home are
  // one place with twenty folders, not twenty pins on top of each other.
  const spots = useMemo(() => {
    const groups = new Map();
    for (const e of entries) {
      const p = e.card?.place;
      if (e.card?.at_home || !p || p.lat == null) continue;
      const key = `${p.lat.toFixed(2)},${p.lon.toFixed(2)}`;
      if (!groups.has(key)) groups.set(key, { lat: p.lat, lon: p.lon, label: p.label, entries: [] });
      groups.get(key).entries.push(e);
    }
    return [...groups.values()];
  }, [entries]);
  const homeEntries = useMemo(() => entries.filter((e) => e.card?.at_home), [entries]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const L = (await import("leaflet")).default;
        await import("leaflet/dist/leaflet.css");
        if (cancelled || !box.current) return;
        // Never zoomed out past a world that fills the box: below that, a phone
        // shows grey above the Arctic and below the Antarctic.
        const fill = Math.ceil(Math.log2(Math.max(box.current.clientWidth, box.current.clientHeight) / 256));
        const map = L.map(box.current, {
          worldCopyJump: true, zoomControl: true, minZoom: Math.max(1, fill),
          maxBounds: [[-85, -720], [85, 720]], maxBoundsViscosity: 1,
        }).setView([30, 0], Math.max(2, fill));
        L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
          maxZoom: 19,
          attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        }).addTo(map);
        map.attributionControl.setPrefix('<a href="https://leafletjs.com">Leaflet</a>');
        mapRef.current = { L, map };
        layerRef.current = L.layerGroup().addTo(map);
        setReady(true);
      } catch (e) {
        if (!cancelled) setFailed(String(e?.message || e));
      }
    })();
    return () => {
      cancelled = true;
      if (mapRef.current) { mapRef.current.map.remove(); mapRef.current = null; }
    };
  }, []);

  useEffect(() => {
    if (!ready || !mapRef.current) return;
    const { L, map } = mapRef.current;
    const layer = layerRef.current;

    const popupFor = (group) => {
      const rows = group.flatMap((spot) => spot.entries.map((e) => ({ e, label: spot.label })));
      const labels = [...new Set(group.map((g) => g.label).filter(Boolean))];
      return `<div class="photool-pop"><p class="photool-pop-place">${escapeHtml(labels.slice(0, 3).join(" · "))}</p>${
        rows.map(({ e }, i) => `<button data-i="${i}" class="photool-pop-row">${
          e.card.cover ? `<img src="${escapeHtml(coverUrl(e.card.cover))}" alt="">` : "<i></i>"
        }<span><b>${escapeHtml(cardTitle(e.card, e.folder))}</b><small>${escapeHtml(cardSubtitle(e.card) || e.folder)}</small></span></button>`).join("")
      }</div>`;
    };

    // Spots closer than a pin's width on screen share one pin, so a zoomed-out
    // world shows "Seattle area, 12 folders" rather than a pile of pins on top
    // of each other. Zooming in pulls them apart again.
    const draw = () => {
      layer.clearLayers();
      const groups = [];
      for (const spot of spots) {
        const pt = map.latLngToLayerPoint([spot.lat, spot.lon]);
        const near = groups.find((g) => g.pt.distanceTo(pt) < 48);
        if (near) near.spots.push(spot);
        else groups.push({ pt, spots: [spot] });
      }
      for (const { spots: group } of groups) {
        const lead = group.reduce((a, b) => (b.entries.length > a.entries.length ? b : a)).entries[0];
        const n = group.reduce((k, g) => k + g.entries.length, 0);
        const lat = group.reduce((k, g) => k + g.lat, 0) / group.length;
        const lon = group.reduce((k, g) => k + g.lon, 0) / group.length;
        const icon = L.divIcon({
          className: "",
          iconSize: [52, 52],
          iconAnchor: [26, 26],
          html: pinHtml(lead.card.cover, n),
        });
        const title = [...new Set(group.map((g) => g.label).filter(Boolean))].join(", ");
        const marker = L.marker([lat, lon], { icon, title, riseOnHover: true });
        if (group.length > 1 && map.getZoom() < map.getMaxZoom()) {
          // Several places under one pin: open them up.
          marker.on("click", () => map.fitBounds(group.map((g) => [g.lat, g.lon]), { padding: [80, 80], maxZoom: 15 }));
        } else {
          const rows = group.flatMap((spot) => spot.entries);
          marker.bindPopup(popupFor(group), { maxWidth: 320, minWidth: 240 });
          marker.on("popupopen", (ev) => {
            ev.popup.getElement()?.querySelectorAll("button[data-i]").forEach((btn) => {
              btn.onclick = () => handlers.current.onOpen(rows[Number(btn.dataset.i)].folder);
            });
          });
        }
        layer.addLayer(marker);
      }
    };

    // Home is one pin, whatever the zoom: everything that happened at home.
    let homeMarker = null;
    if (home && homeEntries.length) {
      const lead = homeEntries.find((e) => e.card?.cover) || homeEntries[0];
      homeMarker = L.marker([home.lat, home.lon], {
        icon: L.divIcon({ className: "", iconSize: [60, 60], iconAnchor: [30, 30],
          html: pinHtml(lead.card?.cover, homeEntries.length, " photool-pin-home") }),
        title: "Home", zIndexOffset: -100,
      }).bindPopup(`<div class="photool-pop"><p class="photool-pop-place">⌂ Home · ${escapeHtml(home.place || "")}</p>
        <p class="photool-pop-note">${homeEntries.length} folder${homeEntries.length === 1 ? "" : "s"} taken at home or with no place of their own.</p></div>`,
        { maxWidth: 280 });
    }

    const bounds = spots.map((s) => [s.lat, s.lon]);
    if (homeMarker) bounds.push([home.lat, home.lon]);
    if (bounds.length === 1) map.setView(bounds[0], 10);
    else if (bounds.length > 1) map.fitBounds(bounds, { padding: [60, 60], maxZoom: 12 });
    const drawAll = () => { draw(); if (homeMarker) layer.addLayer(homeMarker); };
    drawAll();
    map.on("zoomend", drawAll);
    return () => { map.off("zoomend", drawAll); };
  }, [ready, spots, home, homeEntries]);

  const unplaced = home ? [] : entries.filter((e) => e.card && !(e.card.place && e.card.place.lat != null));
  return (
    <div className="space-y-3">
      <div ref={box} className="photool-map w-full h-[min(70vh,720px)] rounded-lg border border-neutral-800 overflow-hidden bg-neutral-900" />
      {failed && <p className="text-sm text-red-400">The map could not be drawn: {failed}</p>}
      {spots.length === 0 && (
        <p className="text-sm text-neutral-500">
          {home ? "Every folder is at home so far. " : "No folder has a place yet. "}
          Give a trip a place from its card, or choose Select to place several folders at once.
        </p>
      )}
      {!home && (
        <p className="text-xs text-neutral-500">Set home under Places, and folders with no place will count as home.</p>
      )}
      {unplaced.length > 0 && (
        <div className="space-y-1">
          <p className="text-xs text-neutral-500">
            {unplaced.length} folder{unplaced.length === 1 ? " has" : "s have"} no place yet — click one to say where it was:
          </p>
          <div className="flex flex-wrap gap-1">
            {unplaced.slice(0, 60).map((e) => (
              <button key={e.folder} onClick={() => onEdit(e)}
                className="text-xs px-2 py-0.5 rounded border border-neutral-800 text-neutral-400 hover:text-white hover:border-neutral-600">
                {cardTitle(e.card, e.folder)}
              </button>
            ))}
            {unplaced.length > 60 && <span className="text-xs text-neutral-600">and {unplaced.length - 60} more</span>}
          </div>
        </div>
      )}
    </div>
  );
}

// ── The overview ────────────────────────────────────────────────────

// ── The same words, above an open folder ────────────────────────────
// Memory comes back while looking at the photographs, not at the card, so the
// title, the place and the line about it can be written from inside the folder.

// ── On this day ──────────────────────────────────────────────────────
//
// Earlier years' best frames from today's date (backend catalog/memories.py), at
// the top of the library. A year with nothing on the day itself may show the
// nearest day a few either side, and says so.

function agoText(year) {
  const n = year.ago;
  const when = n === 1 ? "A year ago" : `${n} years ago`;
  const day = parseDay(year.date);
  const shown = day ? day.toLocaleDateString(undefined, { weekday: "short", day: "numeric", month: "short", year: "numeric" }) : year.date;
  return { when, shown };
}

export function Memories({ onPlay, onOpen }) {
  // Today's, as last seen here, while the server works them out again.
  const [years, setYears] = useState(() => {
    const kept = recall("memories");
    return kept?.day === new Date().toLocaleDateString("en-CA") ? kept.years : null;
  });
  useEffect(() => {
    let live = true;
    fetch(`${API}/library/memories`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (!live || !d) return;
        setYears(d.years || []);
        remember("memories", { day: d.day, years: d.years || [] });
      })
      .catch(() => { if (live) setYears((y) => y || []); });
    return () => { live = false; };
  }, []);
  if (!years || years.length === 0) return null;
  const all = years.flatMap((y) => y.photos);
  return (
    <section className="space-y-2" data-testid="memories">
      <div className="flex items-baseline gap-3">
        <h2 className="text-sm font-medium text-neutral-200">On this day</h2>
        {all.length > 1 && (
          <button onClick={() => onPlay(all, 0, "On this day")}
            className="text-xs text-sky-300 hover:text-sky-200">▶ Play all {all.length}</button>
        )}
      </div>
      <div className="space-y-3">
        {years.map((y) => {
          const { when, shown } = agoText(y);
          return (
            <div key={y.year} className="space-y-1">
              <div className="flex items-baseline gap-2 text-xs flex-wrap">
                <span className="text-neutral-200">{when}</span>
                <button onClick={() => onOpen(y.folder)} className="text-neutral-500 hover:text-neutral-300 underline decoration-neutral-700"
                  title={`Open ${y.folder}`}>{shown}</button>
                {!y.exact && <span className="text-neutral-600">(the nearest day with photos)</span>}
                <span className="text-neutral-600">· {y.total.toLocaleString()} photo{y.total === 1 ? "" : "s"}</span>
                <button onClick={() => onPlay(y.photos, 0, `${when} · ${shown}`)} className="text-sky-300 hover:text-sky-200">▶ Play</button>
              </div>
              <div className="flex gap-1.5 overflow-x-auto pb-1">
                {y.photos.map((p, i) => (
                  <button key={p.filepath} onClick={() => onPlay(y.photos, i, `${when} · ${shown}`)}
                    className="shrink-0 rounded overflow-hidden border border-neutral-800 hover:border-neutral-500">
                    <img src={coverUrl(p.filepath)} alt="" loading="lazy" className="h-24 sm:h-28 w-auto object-cover bg-neutral-900" />
                  </button>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

export function FolderAbout({ folder, card, onCards, before = null, after = null }) {
  const [editing, setEditing] = useState(null);

  const open = async () => {
    // The latest card, home and recent places, so the editor offers the same
    // one-tap places as it does on the overview.
    try {
      const data = await fetch(`${API}/library/cards`).then((r) => r.json());
      onCards?.(Object.fromEntries((data.cards || []).map((c) => [c.folder, c])));
      const fresh = (data.cards || []).find((c) => c.folder === folder) || card;
      const suggestions = [];
      if (data.home) suggestions.push({ ...data.home, home: true });
      for (const p of data.recent_places || []) {
        if (!suggestions.some((o) => o.place === p.place)) suggestions.push(p);
      }
      setEditing({ card: fresh, home: data.home_country || "", suggestions: suggestions.slice(0, 8) });
    } catch {
      setEditing({ card, home: "", suggestions: [] });
    }
  };
  const saved = async () => {
    try {
      const data = await fetch(`${API}/library/cards`).then((r) => r.json());
      onCards?.(Object.fromEntries((data.cards || []).map((c) => [c.folder, c])));
    } catch {}
  };

  const title = cardTitle(card, folder);
  const subtitle = cardSubtitle(card);
  const date = card?.date_from && title !== dateLabel(card) ? dateLabel(card) : "";
  const place = card?.place?.label && !card.at_home ? card.place.label : "";
  return (
    <div className="flex items-start gap-3">
      {before}
      <div className="min-w-0 flex-1">
        <div className="flex items-baseline gap-x-2 flex-wrap">
          <h3 className="text-lg font-medium text-neutral-100" title={folder}>{title}</h3>
          {date && <span className="text-sm text-neutral-400">{date}</span>}
          {place && !subtitle.includes(place) && <span className="text-sm text-neutral-400">📍 {place}</span>}
        </div>
        {subtitle ? (
          <p className="text-sm text-neutral-400">{subtitle}</p>
        ) : card && !card.note?.title && (
          <button onClick={open} className="text-left text-sm text-neutral-600 italic hover:text-neutral-300">
            What was this? Add a title, a place or a line about it
          </button>
        )}
      </div>
      {after}
      {card && (
        <button onClick={open} title="Title, description, place and cover (E)" aria-label="Edit folder"
          className="shrink-0 px-2.5 py-1 rounded border border-neutral-700 text-sm text-neutral-300 hover:text-white hover:border-neutral-500">
          ✎ About
        </button>
      )}
      {editing && (
        <CardEditor entry={{ card: editing.card, folder }} home={editing.home} suggestions={editing.suggestions}
          onClose={() => setEditing(null)} onSaved={saved} />
      )}
    </div>
  );
}

export default function FolderOverview({ folders, onOpen, onCards, onPlay, returnTo = null, restoreScroll = null, demoted = [] }) {
  const [data, setData] = useState(() => recall("cards") || null);
  const [status, setStatus] = useState(null);
  const [query, setQuery] = useState("");
  const [editing, setEditing] = useState(null);
  const [placesOpen, setPlacesOpen] = useState(false);
  const [clocksOpen, setClocksOpen] = useState(false);
  const [openYear, setOpenYear] = useState(null);
  // Choosing several folders to give one place: a trip is often a run of days.
  const [selecting, setSelecting] = useState(false);
  const [selected, setSelected] = useState(() => new Set());
  const [bulkBusy, setBulkBusy] = useState(false);
  const [bulkNote, setBulkNote] = useState("");
  const [view, setView] = useState(() => {
    try { return localStorage.getItem(VIEW_KEY) || "cards"; } catch { return "cards"; }
  });
  const chooseView = (v) => {
    setView(v);
    try { localStorage.setItem(VIEW_KEY, v); } catch {}
  };

  const load = useCallback(async () => {
    try {
      const res = await fetch(`${API}/library/cards`);
      if (!res.ok) return;
      const next = await res.json();
      setData(next);
      remember("cards", next);
      onCards?.(Object.fromEntries((next.cards || []).map((c) => [c.folder, c])));
    } catch {}
  }, [onCards]);

  const loadStatus = useCallback(async () => {
    try {
      const res = await fetch(`${API}/places/status`);
      if (res.ok) setStatus(await res.json());
    } catch {}
  }, []);

  useEffect(() => { load(); loadStatus(); }, [load, loadStatus]);

  const running = status?.job?.status === "running";
  useEffect(() => {
    if (!running) return;
    let last = status?.job?.found || 0;
    const timer = setInterval(async () => {
      try {
        const next = await fetch(`${API}/places/status`).then((r) => r.json());
        setStatus(next);
        if (next.job.status !== "running" || next.job.found !== last) {
          last = next.job.found;
          load();
        }
      } catch {}
    }, 1500);
    return () => clearInterval(timer);
  }, [running, load]); // eslint-disable-line react-hooks/exhaustive-deps

  const scan = async () => {
    try {
      await fetch(`${API}/places/scan`, { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
    } catch {}
    loadStatus();
  };
  const stop = async () => {
    try { await fetch(`${API}/places/scan/stop`, { method: "POST" }); } catch {}
    loadStatus();
  };
  const setHome = async (place) => {
    try {
      await fetch(`${API}/places/home`, {
        method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(place ? { place: place.place, lat: place.lat, lon: place.lon } : {}),
      });
    } catch {}
    load();
  };

  const home = data?.home || null;
  // One tap for the places already in use: home first, then the latest typed.
  const suggestions = useMemo(() => {
    const out = [];
    if (home) out.push({ ...home, home: true });
    for (const p of data?.recent_places || []) {
      if (!out.some((o) => o.place === p.place)) out.push(p);
    }
    return out.slice(0, 8);
  }, [home, data]);

  // Every folder on disk, with its card when the catalog has one.
  const entries = useMemo(() => {
    const byFolder = new Map((data?.cards || []).map((c) => [c.folder, c]));
    const seen = new Set();
    const out = [];
    for (const f of folders) {
      seen.add(f.path);
      out.push({ folder: f.path, disk: f, card: byFolder.get(f.path) || null });
    }
    for (const c of data?.cards || []) {
      if (!seen.has(c.folder)) out.push({ folder: c.folder, disk: null, card: c });
    }
    out.sort((a, b) => {
      const da = a.card?.date_from || "", db = b.card?.date_from || "";
      if (da !== db) return da < db ? 1 : -1;
      return a.folder < b.folder ? 1 : -1;
    });
    return out;
  }, [folders, data]);

  const shown = useMemo(() => {
    const words = query.toLowerCase().split(/\s+/).filter(Boolean);
    if (!words.length) return entries;
    return entries.filter((e) => { const t = searchText(e); return words.every((w) => t.includes(w)); });
  }, [entries, query]);

  // Back from a folder: the page where it was left, and the card of the
  // folder just closed picked out — or, if the window changed size meanwhile
  // and that place no longer shows it, scrolled to.
  const returned = useRef(false);
  useEffect(() => {
    if (returned.current || !data || !folders.length || view !== "cards") return;
    if (!returnTo && restoreScroll == null) return;
    returned.current = true;
    requestAnimationFrame(() => {
      if (restoreScroll != null) window.scrollTo(0, restoreScroll);
      const el = returnTo && document.querySelector(`[data-folder="${CSS.escape(returnTo)}"]`);
      if (el) {
        const box = el.getBoundingClientRect();
        if (box.bottom < 0 || box.top > window.innerHeight) el.scrollIntoView({ block: "center" });
        el.classList.add("photool-returned");
        setTimeout(() => el.classList.remove("photool-returned"), 1600);
      }
    });
  }, [returnTo, restoreScroll, data, folders, view]);

  // What this device demoted, by folder: ["*"] for the whole of it, or cameras.
  const demotions = useMemo(() => {
    const out = new Map();
    for (const d of demoted) out.set(d.folder, [...(out.get(d.folder) || []), d.camera]);
    return out;
  }, [demoted]);

  const years = useMemo(() => {
    // By year wherever a folder sorts: a folder filed under one year but dated
    // in another would otherwise split its year into two sections.
    const groups = new Map();
    for (const e of shown) {
      const y = yearOf(e.card, e.folder);
      if (!groups.has(y)) groups.set(y, { year: y, entries: [] });
      groups.get(y).entries.push(e);
    }
    // A folder demoted whole sinks to the end of its year, still in date order.
    const down = (e) => (demotions.get(e.folder) || []).includes("*");
    for (const g of groups.values()) g.entries = [...g.entries.filter((e) => !down(e)), ...g.entries.filter(down)];
    return [...groups.values()];
  }, [shown, demotions]);

  const toggle = useCallback((folder) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(folder)) next.delete(folder); else next.add(folder);
      return next;
    });
  }, []);
  const toggleYear = (group) => {
    const ids = group.filter((e) => e.card).map((e) => e.folder);
    setSelected((prev) => {
      const next = new Set(prev);
      const all = ids.every((f) => next.has(f));
      for (const f of ids) { if (all) next.delete(f); else next.add(f); }
      return next;
    });
  };
  const endSelecting = () => { setSelecting(false); setSelected(new Set()); setBulkNote(""); };

  const placeSelected = async (place) => {
    if (!selected.size) return;
    setBulkBusy(true);
    setBulkNote("");
    try {
      const res = await fetch(`${API}/library/places`, {
        method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ folders: [...selected], place: place?.place || "", lat: place?.lat ?? null, lon: place?.lon ?? null }),
      });
      const out = await res.json();
      if (!res.ok) { setBulkNote(out.detail || "Not saved"); return; }
      await load();
      setBulkNote(place
        ? `${out.changed} folder${out.changed === 1 ? "" : "s"} placed in ${place.place}`
        : `Place cleared on ${out.changed} folder${out.changed === 1 ? "" : "s"}`);
      setSelected(new Set());
    } catch { setBulkNote("The backend is not reachable"); } finally { setBulkBusy(false); }
  };

  return (
    <div className={`space-y-4 ${selecting ? "pb-48" : ""}`}>
      {onPlay && !query && <Memories onPlay={onPlay} onOpen={onOpen} />}
      {!query && <YearsStrip onOpenYear={setOpenYear} />}
      {openYear && (
        <YearReview year={openYear} onClose={() => setOpenYear(null)}
          onOpenFolder={(folder) => { setOpenYear(null); onOpen(folder); }} />
      )}
      <div className="flex items-center gap-2 flex-wrap">
        <div className="flex rounded border border-neutral-800 overflow-hidden text-sm">
          {[["cards", "Folders"], ["map", "Map"]].map(([k, label]) => (
            <button key={k} onClick={() => chooseView(k)}
              className={`px-3 py-1 ${view === k ? "bg-neutral-800 text-white" : "text-neutral-500 hover:text-neutral-300"}`}>
              {label}
            </button>
          ))}
        </div>
        <input value={query} onChange={(e) => setQuery(e.target.value)}
          placeholder="Find a folder by title, place, person or tag"
          className="flex-1 min-w-[200px] max-w-md bg-neutral-900 border border-neutral-800 rounded px-2 py-1 text-base sm:text-sm text-neutral-200 placeholder-neutral-600" />
        {query && <span className="text-xs text-neutral-500">{shown.length} of {entries.length}</span>}
        <div className="ml-auto flex items-center gap-2">
          {running && (
            <span className="text-xs text-neutral-400">
              Reading GPS {status.job.done.toLocaleString()} / {status.job.total.toLocaleString()}
            </span>
          )}
          {view === "cards" && (
            <button onClick={() => (selecting ? endSelecting() : setSelecting(true))}
              className={`px-3 py-1 rounded border text-sm ${selecting
                ? "border-amber-500 text-amber-300" : "border-neutral-800 text-neutral-400 hover:text-white"}`}>
              {selecting ? "Done" : "Select"}
            </button>
          )}
          <button onClick={() => setClocksOpen(true)} data-testid="clocks-button"
            title="Camera clocks: photos dated by the time where they were taken"
            className="px-3 py-1 rounded border text-sm border-neutral-800 text-neutral-400 hover:text-white">
            🕒 Clocks
          </button>
          <div className="relative">
            <button onClick={() => setPlacesOpen((v) => !v)}
              className={`px-3 py-1 rounded border text-sm ${placesOpen
                ? "border-neutral-600 text-white" : "border-neutral-800 text-neutral-400 hover:text-white"}`}>
              {home ? `⌂ ${home.place.split(",")[0]}` : "Places"}
            </button>
            {placesOpen && (
              <PlacesPanel home={home} homeCountry={data?.home_country || ""} status={status} onHome={setHome}
                onScan={scan} onStop={stop} onClose={() => setPlacesOpen(false)} />
            )}
          </div>
        </div>
      </div>

      {view === "map" ? (
        <FolderMap entries={shown} home={home} onOpen={onOpen} onEdit={setEditing} />
      ) : (
        years.map(({ year, entries: group }) => (
          <section key={year} className="space-y-2">
            <h3 className="sticky top-0 z-10 bg-neutral-950/90 backdrop-blur py-1 text-sm font-medium text-neutral-400 flex items-baseline gap-2">
              <span>{year} <span className="text-neutral-600 font-normal">· {group.length} folder{group.length === 1 ? "" : "s"}</span></span>
              {selecting && (
                <button onClick={() => toggleYear(group)} className="text-xs font-normal text-amber-400/90 hover:text-amber-300">
                  {group.filter((e) => e.card).every((e) => selected.has(e.folder)) ? "Unselect" : "Select"} all of {year}
                </button>
              )}
            </h3>
            <div className="grid gap-3"
              style={{ gridTemplateColumns: "repeat(auto-fill, minmax(min(260px, 100%), 1fr))" }}>
              {group.map((e) => (
                <FolderCard key={e.folder} entry={e} onOpen={onOpen} onEdit={setEditing}
                  selecting={selecting && !!e.card} selected={selected.has(e.folder)} onToggle={toggle}
                  demotion={demotions.get(e.folder) || null} />
              ))}
            </div>
          </section>
        ))
      )}

      {selecting && view === "cards" && (
        <div className="fixed inset-x-0 bottom-0 z-[900] bg-neutral-950/95 backdrop-blur border-t border-neutral-800 px-4 py-3"
          style={{ paddingBottom: "max(0.75rem, env(safe-area-inset-bottom))" }}>
          <div className="max-w-3xl mx-auto space-y-2">
            <div className="flex items-center gap-3 text-sm">
              <span className="text-neutral-200">
                {selected.size ? `${selected.size} folder${selected.size === 1 ? "" : "s"} selected` : "Tap folders to select them"}
              </span>
              {bulkNote && <span className="text-xs text-emerald-400">{bulkNote}</span>}
              <button onClick={endSelecting} className="ml-auto text-xs text-neutral-500 hover:text-white">Done</button>
            </div>
            {selected.size > 0 && (
              <div className={`flex flex-wrap items-start gap-2 ${bulkBusy ? "opacity-50 pointer-events-none" : ""}`}>
                <div className="flex-1 min-w-[220px]">
                  <PlacePicker value={null} onChange={placeSelected} home={data?.home_country || ""} suggestions={suggestions} />
                </div>
                <button onClick={() => placeSelected(null)}
                  className="px-2 py-1 text-xs rounded border border-neutral-800 text-neutral-400 hover:text-red-400">
                  Clear place
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {editing && (
        <CardEditor entry={editing} home={data?.home_country || ""} suggestions={suggestions}
          onClose={() => setEditing(null)} onSaved={load} />
      )}
      {clocksOpen && (
        <ClockPanel onClose={() => setClocksOpen(false)} onChanged={load}
          onOpenFolder={(folder) => { setClocksOpen(false); onOpen(folder); }} />
      )}
    </div>
  );
}
