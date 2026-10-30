import { Component, Fragment, useState, useEffect, useCallback, useRef, useMemo } from "react";

import DevelopView from "./DevelopView.jsx";
import FolderOverview, { FolderAbout } from "./FolderOverview.jsx";
import PersonReview from "./PersonReview.jsx";
import Slideshow from "./Slideshow.jsx";
import CollageMaker from "./CollageMaker.jsx";
import { recall, remember, forgetAll } from "./remembered.js";
import { copiedEdit, useCopiedEdit, PASTE_AS, pasteAs, keepPasteAs, usePasteAs, pasteSummary, lightChanged } from "./editClipboard.js";

// Relative, so the page works from another computer on the network as well as
// through the dev server's proxy (vite.config.js) on this one.
const API = "/api";

// A render error anywhere in the tree unmounts the whole app and leaves a blank
// page, with the reason only in the console. That is a bad way to learn about a
// mistake in one panel, so the develop view — the one screen that replaces
// everything else — reports its own failure instead of taking the library with it.
class PanelBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    console.error("Develop view failed to render:", error, info);
  }

  render() {
    if (this.state.error) {
      return (
        <div className="fixed inset-0 z-[60] bg-neutral-950 text-neutral-100 flex items-center justify-center p-8">
          <div className="max-w-lg space-y-3">
            <h2 className="text-lg font-medium">The develop view could not open this frame</h2>
            <p className="text-sm text-neutral-400">
              The rest of the app is unaffected — close this and carry on. The
              reason is in the browser console.
            </p>
            <pre className="text-xs text-red-400 whitespace-pre-wrap break-words">
              {String(this.state.error?.message || this.state.error)}
            </pre>
            <button onClick={() => { this.setState({ error: null }); this.props.onClose?.(); }}
              className="px-3 py-1.5 rounded border border-neutral-700 text-sm hover:text-white">
              Close
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

// ── Client-side pair filtering (for instant display before catalog loads) ──

function _clientFilter(photos, filter) {
  let result = photos;
  const rawExts = new Set([".cr3", ".cr2", ".arw", ".nef", ".dng", ".raf", ".orf", ".rw2"]);
  const jpgExts = new Set([".jpg", ".jpeg"]);

  if (filter.pairMode === "paired") {
    const byBase = {};
    for (const p of result) {
      if (!byBase[p.base_name]) byBase[p.base_name] = [];
      byBase[p.base_name].push(p);
    }
    result = Object.values(byBase).map((group) => {
      if (group.length === 1) return group[0];
      const jpg = group.find((p) => jpgExts.has(p.extension));
      return jpg || group[0];
    });
  } else if (filter.pairMode === "jpg") {
    result = result.filter((p) => jpgExts.has(p.extension));
  } else if (filter.pairMode === "raw") {
    result = result.filter((p) => rawExts.has(p.extension));
  }

  if (filter.flag !== null) result = result.filter((p) => p.flag === filter.flag);
  if (filter.minRating > 0) result = result.filter((p) => p.rating >= filter.minRating);
  if (filter.color) result = result.filter((p) => p.color_label === filter.color);
  if (filter.edited) result = result.filter((p) => p.edited);

  // Undecided means no cull state anywhere on the image. An event tag is not a
  // decision — it says which event a photo belongs to — and a frame whose twin
  // has already been decided is not asking to be decided again. Both rules match
  // the server query; this is only the pre-index flash of the same folder.
  if (filter.unreviewed) {
    const decided = (p) => p.rating > 0 || p.flag !== 0 || (p.color_label || 0) !== 0;
    const decidedBases = new Set();
    for (const p of photos) if (decided(p)) decidedBases.add(p.base_name);
    result = result.filter((p) => !decided(p) && !decidedBases.has(p.base_name));
  }

  // Tag matching is pair-aware, like the server query: if either half of a
  // CR3/JPG pair carries the tag, the photo the user actually sees stays up.
  if (filter.tag) {
    const wanted = filter.tag.toLowerCase();
    const has = (p) => (p.tags || []).some((t) => t.toLowerCase() === wanted);
    const taggedBases = new Set();
    for (const p of result) if (has(p)) taggedBases.add(p.base_name);
    result = result.filter((p) => has(p) || taggedBases.has(p.base_name));
  }

  // One camera's frames, when a day shot on two was imported into one folder.
  // A pair goes by whichever half knows its camera ("" is none known).
  if (filter.camera != null) {
    const cams = pairCameras(photos);
    result = result.filter((p) => (cams.get(p.base_name) || "") === filter.camera);
  }
  // The cameras demoted in this folder are not on the grid (see hiddenCameras).
  if (filter.hidden?.length) {
    const cams = pairCameras(photos);
    const hidden = new Set(filter.hidden);
    result = result.filter((p) => !hidden.has(cams.get(p.base_name) || ""));
  }

  // A search typed while in a folder narrows the folder (the pairs it found).
  if (filter.matches) result = result.filter((p) => filter.matches.bases.has(p.base_name));

  return result;
}

// The camera of each pair in a folder, by base name: whichever file of it
// carries one (a camera's JPEG may have lost its EXIF while the raw kept it).
function pairCameras(photos) {
  const cams = new Map();
  for (const p of photos) {
    const c = (p.camera || "").trim();
    if (c && !cams.get(p.base_name)) cams.set(p.base_name, c);
  }
  return cams;
}

// The cameras in a folder with how many photos each, the most first; "" for
// the photos no camera is known for.
function folderCameras(photos) {
  const cams = pairCameras(photos);
  const counts = new Map();
  for (const base of new Set(photos.map((p) => p.base_name))) {
    const c = cams.get(base) || "";
    counts.set(c, (counts.get(c) || 0) + 1);
  }
  return [...counts.entries()].map(([name, count]) => ({ name, count }))
    .sort((a, b) => (a.name === "") - (b.name === "") || b.count - a.count);
}

// A folder's pairs with the cameras in ``hidden`` left out: how many went, and
// the counts the review progress shows for the rest.
function shownCounts(photos, hidden) {
  const cams = pairCameras(photos);
  const out = new Set(hidden);
  const gone = new Set(), kept = new Set(), decided = new Set(), picks = new Set(), rejects = new Set();
  for (const p of photos) {
    if (out.has(cams.get(p.base_name) || "")) { gone.add(p.base_name); continue; }
    kept.add(p.base_name);
    if (p.rating > 0 || p.flag !== 0 || (p.color_label || 0) !== 0) decided.add(p.base_name);
    if (p.flag === 1) picks.add(p.base_name);
    if (p.flag === -1) rejects.add(p.base_name);
  }
  return { hidden: gone.size, unique_images: kept.size, decided: decided.size,
    undecided: kept.size - decided.size, picks: picks.size, rejects: rejects.size };
}

// The cameras this device demoted in a folder, which its grid leaves out — and
// with them the cull view, the slideshow, the bursts and every action on "these".
// Not when the whole folder is demoted (it is less important as a whole; inside
// it everything still counts), not the camera picked by its chip, and not once
// "show" has been pressed.
function hiddenCameras(demoted, folder, camera, showDemoted) {
  if (!folder || showDemoted) return [];
  const mine = demoted.filter((d) => d.folder === folder).map((d) => d.camera);
  if (mine.includes("*")) return [];
  return mine.filter((c) => c !== camera);
}

// The grid's filter state in the shape the bulk endpoints expect, so "add to
// basket" and "tag these" act on exactly the selection the user is looking at.
function filterPayload(filter) {
  return {
    min_rating: filter.minRating || 0,
    flag: filter.flag ?? null,
    color: filter.color ?? null,
    tag: filter.tag || null,
    unreviewed: !!filter.unreviewed,
    pair_mode: filter.pairMode || "paired",
    camera: filter.camera ?? null,
    hide_cameras: filter.hidden?.length ? filter.hidden : null,
    bases: filter.matches ? [...filter.matches.bases] : null,
  };
}

// The same filter, in the shape the per-folder read endpoints take as query
// parameters. Bursts and quality hints both have to see exactly what the grid is
// showing, or they answer for frames the user cannot see.
function photoQuery(filter) {
  const params = new URLSearchParams();
  if (filter.minRating > 0) params.set("min_rating", filter.minRating);
  if (filter.flag !== null) params.set("flag", filter.flag);
  if (filter.color) params.set("color", filter.color);
  if (filter.tag) params.set("tag", filter.tag);
  if (filter.unreviewed) params.set("unreviewed", "true");
  if (filter.pairMode) params.set("pair_mode", filter.pairMode);
  if (filter.camera != null) params.set("camera", filter.camera);
  for (const c of filter.hidden || []) params.append("hide_camera", c);
  return params;
}

// Opening a folder refetches the list once background indexing finishes, and that
// response can predate a rating the user just made — the star would light up and
// then go dark again. Remember the user's own edits and re-apply them over any
// server list so a rating never visually reverts.
function _applyPendingEdits(list, pendingEdits) {
  if (!pendingEdits || pendingEdits.size === 0) return list;
  return list.map((p) => {
    const edits = pendingEdits.get(p.filepath);
    return edits ? { ...p, ...edits } : p;
  });
}

// ── URL builders ────────────────────────────────────────────────────

// A short stand-in for a photo's edit, used only to give an edited thumbnail its
// own URL. The server draws the edit whether or not this is sent; what this does
// is make the browser ask again when the edit changes, instead of showing the
// copy it kept from before. An unedited photo has none, so its URL is unchanged.
function editKey(photo) {
  const edit = photo && photo.edit;
  if (!edit) return "";
  const keys = Object.keys(edit).filter((k) => Number(edit[k]) !== 0).sort();
  if (!keys.length) return "";
  let h = 5381;
  for (const ch of keys.map((k) => `${k}=${edit[k]}`).join(";")) {
    h = ((h << 5) + h + ch.charCodeAt(0)) | 0;
  }
  return `e${(h >>> 0).toString(36)}`;
}

// A per-viewer convenience kept in the browser (thumbnail size and the like).
// Storage can be missing or refuse writes (a private window, a blocked site), so
// every read and write is guarded and the default stands in.
function useStoredState(key, initial) {
  const [value, setValue] = useState(() => {
    try {
      const raw = window.localStorage.getItem(key);
      return raw === null ? initial : JSON.parse(raw);
    } catch {
      return initial;
    }
  });
  useEffect(() => {
    try { window.localStorage.setItem(key, JSON.stringify(value)); } catch { /* ignore */ }
  }, [key, value]);
  return [value, setValue];
}

// Grid cell widths in CSS pixels. The grid fills the window with as many
// columns of this width as fit, so a 34" ultrawide gets fourteen columns and a
// laptop five, instead of both getting six.
const THUMB_SIZES = { S: 140, M: 190, L: 260, XL: 360 };

// A phone: a narrow screen worked with a finger. There the layout gets out of
// the photos' way (a tab bar at the bottom, the filters folded, a viewer swiped
// through); a wide touch screen, an iPad, keeps the desktop layout.
const PHONE_QUERY = "(max-width: 640px) and (pointer: coarse)";
function usePhone() {
  const [phone, setPhone] = useState(() => !!window.matchMedia?.(PHONE_QUERY).matches);
  useEffect(() => {
    const m = window.matchMedia?.(PHONE_QUERY);
    if (!m) return undefined;
    const on = () => setPhone(m.matches);
    m.addEventListener("change", on);
    return () => m.removeEventListener("change", on);
  }, []);
  return phone;
}
// Thumbnails on a phone: three across.
const PHONE_CELL = 100;

function thumbUrl(folderPath, filename, size = "grid", edit = "") {
  const version = edit ? `&v=${edit}` : "";
  return `${API}/thumb/${folderPath}/${encodeURIComponent(filename)}?size=${size}${version}`;
}

// ── Where the app is, as an address ─────────────────────────────────

const TABS = ["import", "library", "people", "activity", "export", "devices"];

function routeHash(r) {
  const folder = r.folder ? "/" + r.folder.split("/").map(encodeURIComponent).join("/") : "";
  return `#/${r.tab}${folder}${r.viewer ? "?photo" : ""}`;
}

function parseRoute(hash) {
  const m = /^#\/([a-z]+)(\/[^?]*)?(\?photo)?$/.exec(hash || "");
  if (m && m[1] === "queue") m[1] = "activity";   // the tab's old name, in old links
  if (!m || !TABS.includes(m[1])) return null;
  let folder = null;
  if (m[2] && m[2].length > 1) {
    try { folder = m[2].slice(1).split("/").map(decodeURIComponent).join("/"); } catch { folder = null; }
  }
  return { tab: m[1], folder: m[1] === "library" ? folder : null, viewer: !!m[3] && !!folder };
}

function sameRoute(a, b) {
  return !!a && !!b && a.tab === b.tab && (a.folder || null) === (b.folder || null) && !!a.viewer === !!b.viewer;
}

// ── Prefetch helper ─────────────────────────────────────────────────

function prefetch(filepaths, size = "grid") {
  if (!filepaths.length) return;
  fetch(`${API}/thumbnails/prefetch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filepaths, size }),
  }).catch(() => {});
}

// ── Lazy image (IntersectionObserver) ───────────────────────────────

function LazyImage({ src, alt, className }) {
  const ref = useRef(null);
  const [loaded, setLoaded] = useState(false);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) { setVisible(true); obs.disconnect(); } },
      { rootMargin: "200px" }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  return (
    <div ref={ref} className={`${className} bg-neutral-900`}>
      {visible && (
        <img
          src={src}
          alt={alt}
          loading="lazy"
          onLoad={() => setLoaded(true)}
          className={`w-full h-full object-cover transition-opacity duration-200 ${
            loaded ? "opacity-100" : "opacity-0"
          }`}
        />
      )}
    </div>
  );
}

// ── Stars / Flag / Color components ─────────────────────────────────

function Stars({ rating, onRate, size = "text-sm" }) {
  return (
    <span className={`inline-flex gap-0.5 ${size}`}>
      {[1, 2, 3, 4, 5].map((n) => (
        <button
          key={n}
          onClick={(e) => { e.stopPropagation(); onRate(rating === n ? 0 : n); }}
          className={`transition-colors ${
            n <= rating ? "text-yellow-400" : "text-neutral-700 hover:text-neutral-500"
          }`}
        >★</button>
      ))}
    </span>
  );
}

function FlagBadge({ flag, size = "text-xs" }) {
  if (flag === 1) return <span className={`${size} text-green-400 font-bold`}>P</span>;
  if (flag === -1) return <span className={`${size} text-red-400 font-bold`}>✕</span>;
  return null;
}

const COLOR_MAP = { 0: "", 1: "bg-red-500", 2: "bg-yellow-400", 3: "bg-green-500", 4: "bg-blue-500", 5: "bg-purple-500" };
const COLOR_NAMES = { 1: "Red", 2: "Yellow", 3: "Green", 4: "Blue", 5: "Purple" };

function ColorDot({ color, size = "w-2.5 h-2.5" }) {
  if (!color) return null;
  return <span className={`${size} rounded-full ${COLOR_MAP[color]} inline-block`} />;
}

// A frame whose edit takes noise out (adjust/noise.py), for its NR badge.
function denoised(photo) {
  const edit = photo && photo.edit;
  return !!edit && (Number(edit.nr_luma) > 0 || Number(edit.nr_color) > 0);
}

// ── Extension badge ─────────────────────────────────────────────────

// Videos sit in the grid beside the photographs, to be watched and culled like
// them (stars, flags, tags, basket), but are never developed.
const VIDEO_EXTS = [".mp4", ".mov", ".avi", ".mts", ".m4v"];

function isVideo(photoOrPath) {
  if (!photoOrPath) return false;
  if (typeof photoOrPath === "object" && photoOrPath.video) return true;
  const path = typeof photoOrPath === "string" ? photoOrPath : photoOrPath.filepath || "";
  return VIDEO_EXTS.some((e) => path.toLowerCase().endsWith(e));
}

function clipLength(seconds) {
  const s = Math.round(Number(seconds) || 0);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = s % 60;
  const pad = (n) => String(n).padStart(2, "0");
  return h ? `${h}:${pad(m)}:${pad(r)}` : `${m}:${pad(r)}`;
}

// An EXIF date ("2026:04:12 08:25:23") as a person reads it: "Sun, Apr 12, 2026, 8:25:23 AM".
// It is the time where the photo was taken, so it is shown as is, never moved to this
// computer's time zone.
function takenAt(exifDate) {
  const m = /^(\d{4}):(\d{2}):(\d{2})[ T](\d{2}):(\d{2}):(\d{2})/.exec(exifDate || "");
  if (!m) return exifDate || "";
  const d = new Date(Date.UTC(+m[1], m[2] - 1, +m[3], +m[4], +m[5], +m[6]));
  return d.toLocaleString(undefined, {
    timeZone: "UTC", weekday: "short", year: "numeric", month: "short", day: "numeric",
    hour: "numeric", minute: "2-digit", second: "2-digit",
  });
}

function VideoBadge({ photo }) {
  if (!isVideo(photo)) return null;
  return (
    <span className="px-1 py-0.5 rounded bg-black/75 border border-neutral-600 text-white text-[9px] leading-none font-semibold"
      title="A video: open it to play">▶ {photo.duration ? clipLength(photo.duration) : ""}</span>
  );
}

function ExtBadge({ ext }) {
  const isRaw = [".cr3", ".cr2", ".arw", ".nef", ".dng"].includes(ext);
  if (!isRaw) return null;
  return (
    <span className="text-[9px] font-bold px-1 py-0.5 rounded bg-amber-900/80 text-amber-300 uppercase">
      {ext.replace(".", "")}
    </span>
  );
}

// ── Quality warnings ────────────────────────────────────────────────
//
// Read from the measurements the backend already collects, and deliberately
// only ever a mark on the frame: a soft frame may be the right photo, and
// clipped highlights may be the sky. Nothing here rejects anything.

function HintBadges({ hint }) {
  if (!hint || (!hint.soft_frame && !hint.blown_highlights && !hint.eyes_closed)) return null;
  const chip = "px-1 py-0.5 rounded bg-neutral-900/85 border border-neutral-700 text-[9px] leading-none";
  return (
    <>
      {hint.soft_frame && (
        <span className={`${chip} text-amber-300/90`}
          title="Measurably softer than the other frames in this folder — often back-button focus or a moving subject. A warning, not a verdict.">
          soft
        </span>
      )}
      {hint.blown_highlights && (
        <span className={`${chip} text-red-300`}
          title="Highlights clipped to white — detail the exposure cannot give back.">
          ☀
        </span>
      )}
      {hint.eyes_closed && (
        <span className={`${chip} text-sky-300`}
          title="Someone's eyes look closed here, against the same person in the other frames of this burst. A hint from the faces, not a verdict.">
          eyes?
        </span>
      )}
    </>
  );
}

// ── An agent's proposals ────────────────────────────────────────────
//
// What an agent culling over MCP suggests for a frame (backend
// catalog/proposals.py). Kept apart from the person's own marks, and drawn in
// violet so the two are never confused, until the person accepts it.

const LABEL_NAMES = { 1: "red", 2: "yellow", 3: "green", 4: "blue", 5: "purple" };

function proposalText(p) {
  const parts = [];
  if (p.rating != null) parts.push(p.rating ? "★".repeat(p.rating) : "no stars");
  if (p.flag === 1) parts.push("pick");
  if (p.flag === -1) parts.push("reject");
  if (p.flag === 0) parts.push("unflag");
  if (p.color_label != null) parts.push(p.color_label ? `${LABEL_NAMES[p.color_label]} label` : "no label");
  return parts.join(" · ");
}

function ProposalChip({ proposal }) {
  if (!proposal) return null;
  const mark = proposal.flag === -1 ? "✗" : proposal.flag === 1 ? "✓" : proposal.rating ? `★${proposal.rating}` : "·";
  return (
    <span data-testid="proposal-chip"
      className={`px-1 py-0.5 rounded border text-[9px] leading-none font-semibold ${proposal.flag === -1
        ? "bg-violet-950/90 border-violet-500/70 text-red-300" : "bg-violet-900/85 border-violet-500 text-violet-100"}`}
      title={`Suggested by ${proposal.agent || "an agent"}: ${proposalText(proposal)}${proposal.reason ? ` — ${proposal.reason}` : ""}. Open the photo to accept or dismiss it.`}>
      ✦{mark}
    </span>
  );
}

// The folder's waiting proposals, all at once. Accepting them all keeps every
// mark the person has already set; one accepted in the photo view replaces it.
function ProposalBar({ proposals, busy, message, canUndo, onAccept, onDismiss, onSelect, onUndo }) {
  const list = Object.values(proposals || {});
  if (!list.length && !canUndo && !message) return null;
  const agents = [...new Set(list.map((p) => p.agent).filter(Boolean))];
  const rejects = list.filter((p) => p.flag === -1).length;
  return (
    <div data-testid="proposal-bar"
      className="mb-2 px-3 py-2 rounded border border-violet-800/70 bg-violet-950/40 text-sm flex items-center gap-x-3 gap-y-1.5 flex-wrap">
      {list.length > 0 && (
        <>
          <span className="text-violet-200">
            ✦ {agents.length ? agents.join(", ") : "An agent"} suggests marks for {list.length} photo{list.length === 1 ? "" : "s"}
            {rejects ? <span className="text-violet-300/70"> ({rejects} reject{rejects === 1 ? "" : "s"})</span> : null}
          </span>
          <button onClick={onSelect} disabled={busy}
            className="px-2 py-0.5 rounded border border-violet-700 text-violet-200 hover:bg-violet-900/60 text-xs disabled:opacity-40"
            title="Select them, to look through or act on them from the selection bar">Select them</button>
          <button onClick={onAccept} disabled={busy} data-testid="proposals-accept"
            className="px-2 py-0.5 rounded bg-violet-700 text-white hover:bg-violet-600 text-xs disabled:opacity-40"
            title="Make them your marks. A star, flag or label you have already set is kept">Accept all</button>
          <button onClick={onDismiss} disabled={busy} data-testid="proposals-dismiss"
            className="px-2 py-0.5 rounded border border-neutral-700 text-neutral-300 hover:text-white text-xs disabled:opacity-40">Dismiss all</button>
        </>
      )}
      {canUndo && (
        <button onClick={onUndo} disabled={busy} data-testid="proposals-undo"
          className="px-2 py-0.5 rounded border border-neutral-700 text-neutral-300 hover:text-white text-xs disabled:opacity-40"
          title="Put back the marks accepting replaced, except any you have changed since">Undo accept</button>
      )}
      {message && <span className="text-xs text-violet-200/80">{message}</span>}
    </div>
  );
}

// ── Menus ───────────────────────────────────────────────────────────
//
// A button that opens a small panel under it. The folder view keeps what is
// used now and then in these, so its top stays a few short rows: narrow the
// grid, then act on it. A click elsewhere or Escape closes it.

function Menu({ label, title, active = false, align = "left", caret = true, small = false, testid, children }) {
  const [open, setOpen] = useState(false);
  const box = useRef(null);
  useEffect(() => {
    if (!open) return;
    const away = (e) => { if (box.current && !box.current.contains(e.target)) setOpen(false); };
    // Ahead of the page's own Escape, which would leave the folder.
    const key = (e) => { if (e.key === "Escape") { e.stopPropagation(); setOpen(false); } };
    document.addEventListener("pointerdown", away);
    window.addEventListener("keydown", key, true);
    return () => { document.removeEventListener("pointerdown", away); window.removeEventListener("keydown", key, true); };
  }, [open]);
  return (
    <div ref={box} className="relative">
      <button onClick={() => setOpen((v) => !v)} title={title} data-testid={testid} aria-expanded={open}
        className={`${small ? "px-2 py-0.5 text-xs" : "px-2.5 py-1 text-sm"} rounded border whitespace-nowrap transition-colors ${open || active
          ? "border-neutral-500 bg-neutral-800 text-white" : "border-neutral-700 text-neutral-300 hover:text-white hover:border-neutral-500"}`}>
        {label}{caret && <span className="ml-1 text-xs text-neutral-500">▾</span>}
      </button>
      {open && (
        <div className={`absolute z-[950] mt-1 ${align === "right" ? "right-0" : "left-0"} w-max min-w-[15rem] max-w-[calc(100vw-2rem)] rounded-lg border border-neutral-700 bg-neutral-900 p-1.5 text-sm shadow-xl`}>
          {typeof children === "function" ? children(() => setOpen(false)) : children}
        </div>
      )}
    </div>
  );
}

function MenuItem({ onClick, disabled = false, title, testid, children }) {
  return (
    <button onClick={onClick} disabled={disabled} title={title} data-testid={testid}
      className="flex w-full items-center gap-2 rounded px-2 py-1.5 text-left text-neutral-200 hover:bg-neutral-800 disabled:opacity-40 disabled:hover:bg-transparent">
      {children}
    </button>
  );
}

function MenuLabel({ children }) {
  return <p className="px-2 pt-1.5 pb-0.5 text-[11px] uppercase tracking-wide text-neutral-500">{children}</p>;
}

// ── Filter bar ──────────────────────────────────────────────────────

// A day shot on two cameras is often imported into one folder: one chip per
// camera narrows the grid (and the cull, bursts and slideshow) to one of them.
function CameraChips({ cameras, value, onChange, demoted = new Set() }) {
  if (!cameras || cameras.length < 2) return null;
  const total = cameras.reduce((n, c) => n + c.count, 0);
  const chip = (key, label, count, on, pick, down = false) => (
    <button key={key} onClick={pick} data-testid="camera-chip"
      title={down ? "Less important on this device: hidden from the folder until picked here" : undefined}
      className={`px-2 py-0.5 rounded border text-xs ${on
        ? "bg-sky-800/50 border-sky-600 text-sky-100" : "border-neutral-700 text-neutral-400 hover:text-neutral-200"} ${down && !on ? "opacity-50" : ""}`}>
      {down && "↓ "}{label} <span className="text-neutral-500">{count.toLocaleString()}</span>
    </button>
  );
  return (
    <div className="flex items-center gap-1.5 flex-wrap text-sm" data-testid="camera-chips">
      <span className="text-neutral-600 text-xs mr-0.5">Camera</span>
      {chip("all", "All", total, value == null, () => onChange(null))}
      {cameras.map((c) => chip(c.name || "-", c.name || "Unknown", c.count, value === c.name,
        () => onChange(value === c.name ? null : c.name), demoted.has(c.name)))}
    </div>
  );
}

// The frames of the cameras demoted in this folder are left out of it; this
// says how many, and brings them back (or puts them away again).
function HiddenShots({ cameras, count, showing, onShow }) {
  if (!cameras.length) return null;
  const names = cameras.map((c) => c || "Unknown camera").join(", ");
  if (showing) {
    return (
      <span className="text-xs text-amber-300/80" data-testid="hidden-shots">
        Showing {names} too ·{" "}
        <button onClick={() => onShow(false)} className="underline hover:text-amber-200">hide again</button>
      </span>
    );
  }
  if (!count) return null;
  return (
    <span className="text-xs text-neutral-500" data-testid="hidden-shots"
      title="Less important on this device. Change it in the ⋯ menu.">
      ↓ {count.toLocaleString()} from {names} hidden ·{" "}
      <button onClick={() => onShow(true)} className="underline hover:text-neutral-300">show</button>
    </span>
  );
}

// Less important on this device: a whole folder, or one camera's photos in it
// (the other person's camera on a shared day). A demoted camera's frames leave
// the folder's grid, one tap from coming back; a demoted folder sinks in the
// folder list. Both sink in search and stay out of what is picked for this
// person — On this day, Year in review. Nothing is deleted.
function ShootMenu({ folder, cameras, demoted, onSet }) {
  const has = (camera) => demoted.some((d) => d.folder === folder && d.camera === camera);
  const row = (camera, label, title, testid) => {
    const on = has(camera);
    return (
      <MenuItem key={camera} onClick={() => onSet(folder, camera, !on)} title={title} testid={testid}>
        <span className={`w-4 text-center ${on ? "text-amber-300" : "text-neutral-700"}`}>{on ? "✓" : "○"}</span>
        <span>{label}</span>
      </MenuItem>
    );
  };
  return (
    <Menu label="↓ Less important" small title="Mark this folder, or one camera's photos in it, less important on this device"
      active={demoted.some((d) => d.folder === folder)} testid="shoot-menu">
      <MenuLabel>Less important on this device</MenuLabel>
      {row("*", "This whole folder",
        "It sinks in the folder list and in searches, and stays out of On this day and Year in review.",
        "demote-folder")}
      {cameras.length > 1 && cameras.map((c) => row(c.name, `${c.name || "Unknown camera"} photos`,
        "Hidden from this folder (one tap brings them back), last in searches, and left out of On this day and Year in review.",
        "demote-camera"))}
      <p className="px-2 pt-1 pb-1 text-[11px] text-neutral-600 max-w-[17rem]">
        Only on this device. Nothing is deleted.
      </p>
    </Menu>
  );
}

const FLAG_OPTIONS = [
  { value: null, label: "All" },
  { value: 1, label: "Picks" },
  { value: 0, label: "Unflagged" },
  { value: -1, label: "Rejects" },
];

const PAIR_OPTIONS = [
  { value: "all", label: "All files" },
  { value: "paired", label: "Dedupe pairs" },
  { value: "jpg", label: "JPG only" },
  { value: "raw", label: "RAW only" },
  { value: "jpg_only", label: "JPG (no RAW)" },
  { value: "raw_only", label: "RAW (no JPG)" },
];

// What is narrowed most often stays on the bar: the review pass, the flag and
// the stars. Colour, tag, edited and which files of a pair live under "More".
function FilterBar({ filter, onFilterChange, tags = [] }) {
  const set = (patch) => onFilterChange({ ...filter, ...patch });
  const more = (filter.pairMode !== "paired") + !!filter.edited + !!filter.color + !!filter.tag;
  const any = filter.flag !== null || filter.minRating > 0 || filter.unreviewed || more > 0;

  return (
    <div className="flex items-center gap-2 text-sm flex-wrap" data-testid="filter-bar">
      {/* Review pass: only the frames that carry no decision yet. This is how a
          whole event gets worked through, so it comes first. */}
      <button
        onClick={() => set({ unreviewed: !filter.unreviewed })}
        title="Show only photos with no rating, flag, or colour label. An event tag is not a decision, and a frame whose RAW+JPG twin was already decided is not shown again."
        className={`px-2.5 py-1 rounded border transition-colors ${
          filter.unreviewed
            ? "bg-emerald-700/40 border-emerald-600 text-emerald-200"
            : "border-neutral-700 text-neutral-300 hover:text-white"
        }`}
      >Undecided only</button>

      <div className="flex rounded border border-neutral-700 overflow-hidden">
        {FLAG_OPTIONS.map((opt) => (
          <button
            key={String(opt.value)}
            onClick={() => set({ flag: opt.value })}
            className={`px-2.5 py-1 transition-colors ${
              filter.flag === opt.value ? "bg-neutral-700 text-white" : "text-neutral-400 hover:text-neutral-200"
            }`}
          >{opt.label}</button>
        ))}
      </div>

      <select value={filter.minRating} onChange={(e) => set({ minRating: +e.target.value })}
        title="Least stars"
        className={`bg-neutral-900 border rounded px-2 py-1 text-sm ${filter.minRating > 0
          ? "border-neutral-500 text-yellow-400" : "border-neutral-700 text-neutral-300"}`}>
        <option value={0}>★ Any</option>
        {[1, 2, 3, 4, 5].map((n) => <option key={n} value={n}>{"★".repeat(n)}{n < 5 ? "+" : ""}</option>)}
      </select>

      <Menu label={more ? `More filters (${more})` : "More filters"} active={more > 0} testid="more-filters">
        <MenuLabel>Colour label</MenuLabel>
        <div className="flex items-center gap-2 px-2 py-1.5">
          <button onClick={() => set({ color: null })}
            className={`px-1.5 rounded text-xs ${!filter.color ? "bg-neutral-700 text-white" : "text-neutral-500 hover:text-neutral-300"}`}>Any</button>
          {[1, 2, 3, 4, 5].map((c) => (
            <button key={c} title={COLOR_NAMES[c]} onClick={() => set({ color: filter.color === c ? null : c })}
              className={`w-5 h-5 rounded-full ${COLOR_MAP[c]} transition-all ${
                filter.color === c ? "ring-2 ring-white scale-110" : "opacity-40 hover:opacity-80"}`} />
          ))}
        </div>
        <MenuLabel>Event tag</MenuLabel>
        <div className="px-2 py-1">
          <select value={filter.tag || ""} onChange={(e) => set({ tag: e.target.value || null })}
            className="w-full bg-neutral-800 border border-neutral-700 rounded px-2 py-1 text-sm text-neutral-300">
            <option value="">Any tag</option>
            {tags.map((t) => <option key={t.name} value={t.name}>{t.name} ({t.photo_count})</option>)}
          </select>
        </div>
        <MenuLabel>Files</MenuLabel>
        <div className="px-2 py-1">
          <select value={filter.pairMode} onChange={(e) => set({ pairMode: e.target.value })}
            title="With Dedupe pairs a RAW+JPG pair is one photo"
            className="w-full bg-neutral-800 border border-neutral-700 rounded px-2 py-1 text-sm text-neutral-300">
            {PAIR_OPTIONS.map((opt) => <option key={opt.value} value={opt.value}>{opt.label}</option>)}
          </select>
        </div>
        {/* What was done to a picture, rather than a decision about keeping it:
            how the frames being worked on are found again. */}
        <MenuItem onClick={() => set({ edited: !filter.edited })} title="Show only photos that carry a develop edit.">
          <span className={`w-4 text-center ${filter.edited ? "text-amber-300" : "text-neutral-700"}`}>{filter.edited ? "✓" : "○"}</span>
          Edited only
        </MenuItem>
      </Menu>

      {any && (
        <button
          onClick={() => set({ flag: null, minRating: 0, color: null, tag: null, unreviewed: false, edited: false, pairMode: "paired" })}
          className="text-xs text-neutral-500 hover:text-white"
        >✕ Clear filters</button>
      )}
    </div>
  );
}

// ── Library search ───────────────────────────────────────────────────
//
// One line of text over the whole library: `rating:>=3 iso:>3200 date:2026`.
// The language lives in backend/catalog/search.py; this box only sends it and
// says back how it was read, so a typo shows up as a term that was not meant.

const SEARCH_HELP = [
  ["rating:>=3", "stars — also =, >, <, <=, or 2..4"],
  ["flag:pick", "pick · reject · none"],
  ["label:red,green", "colour label; a comma means either"],
  ['tag:"school play"', "a tag, exactly"],
  ["camera:r6  lens:35", "any part of the name"],
  ["iso:>3200  f:<=2.8", "numbers compare"],
  ["focal:24..35  shutter:<1/100", "ranges and fractions"],
  ["date:2026-04", "a year, month or day; date:>=2026-03, date:2024..2025"],
  ["folder:hawaii", "any part of the folder"],
  ["type:raw", "raw · jpg · cr3 · cr2 · dng"],
  ["is:edited", "edited · unedited · unreviewed · reviewed · pick · reject · basket"],
  ["is:bw", "developed in black and white; is:colour for the rest"],
  ["is:denoised", "developed with noise reduction (or is:nr)"],
  ['shows:"kids at the beach"', "what the picture shows, best first (smart search; or turn on Describe)"],
  ["-tag:family", "a minus turns any term around"],
  ["garden", "a plain word: filename, folder, camera, lens or tags"],
];

// Smart search (backend analysis/clip.py): set up once, then the library is
// described in the background. What it says while that happens.
function SmartSearchStatus({ smart, onSetup }) {
  if (!smart) return null;
  const { available, setup, job, photos, described, download_mb: mb } = smart;
  if (setup?.status === "downloading") {
    return <p className="text-xs text-neutral-500">Fetching the smart search models ({mb} MB, once)…</p>;
  }
  if (!available) {
    return (
      <p className="text-xs text-neutral-500">
        Describe finds photos by what they show — "kids at the beach", "a red car", "snow". It needs a
        one-time {mb} MB download, then describes your photos in the background (roughly 15 minutes per
        40,000).{" "}
        <button onClick={onSetup} className="text-amber-400 hover:text-amber-300" data-testid="smart-setup">Set up</button>
        {setup?.status === "failed" && <span className="text-red-400"> {setup.error}</span>}
      </p>
    );
  }
  if (described < photos) {
    return (
      <p className="text-xs text-neutral-500">
        Describing your photos: {described.toLocaleString()} of {photos.toLocaleString()}
        {job?.status === "running" ? "" : " (waiting)"} — a search finds those done so far.
      </p>
    );
  }
  return null;
}

function SearchBar({ value, onChange, onSubmit, onClear, result, error, describe, onDescribe, smart, onSetup,
  folder = null, inFolder = false, onInFolder }) {
  const [help, setHelp] = useState(false);
  const folderName = folder ? folder.split("/").pop() : "";
  return (
    <div className="space-y-1">
      <div className="flex items-center gap-2">
        {/* Inside a folder a search is of that folder, until the chip is
            turned off; on the folder list it is of the whole library. */}
        {folder && (
          <button onClick={() => onInFolder(!inFolder)} data-testid="search-scope"
            title={inFolder ? `Searching ${folder} only. Click to search the whole library` : `Searching the whole library. Click to search ${folder} only`}
            className={`shrink-0 text-xs px-2 py-1 rounded border ${inFolder
              ? "border-sky-600 bg-sky-900/30 text-sky-200" : "border-neutral-700 text-neutral-500 hover:text-neutral-300"}`}>
            {inFolder ? "This folder ✕" : "Whole library"}
          </button>
        )}
        <input
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") onSubmit(value);
            if (e.key === "Escape") onClear();
            e.stopPropagation();
          }}
          placeholder={folder && inFolder
            ? (describe ? "Describe a photo in this folder" : "Search this folder")
            : describe ? "Describe what's in the photo — kids at the beach, rating:>=3"
            : "Search the library — rating:>=3 iso:>3200 date:2026"}
          spellCheck={false}
          className="flex-1 min-w-0 max-w-xl bg-neutral-900 border border-neutral-700 rounded px-3 py-1.5 text-sm font-mono text-neutral-200 placeholder:text-neutral-600 focus:outline-none focus:border-neutral-500"
        />
        {(result || error || value) && (
          <button onClick={onClear} className="text-xs text-neutral-500 hover:text-white">✕ Clear</button>
        )}
        <button onClick={onDescribe} data-testid="describe-toggle"
          title="Search by what the photos show, in your own words; other terms (rating:>=3, date:2024) still work alongside"
          className={`text-xs px-1.5 rounded border ${describe ? "border-amber-500/70 text-amber-300" : "border-neutral-700 text-neutral-500 hover:text-neutral-300"}`}>
          ✦ Describe
        </button>
        <button onClick={() => setHelp((h) => !h)} title="What can I search for?"
          className={`text-xs px-1.5 rounded border ${help ? "border-neutral-500 text-neutral-200" : "border-neutral-700 text-neutral-500 hover:text-neutral-300"}`}>?</button>
      </div>
      {describe && <SmartSearchStatus smart={smart} onSetup={onSetup} />}
      {error && <p className="text-xs text-red-400">{error}</p>}
      {result && !error && (
        <p className="text-xs text-neutral-500">
          {result.total} {result.total === 1 ? "photo" : "photos"}
          {result.inFolder && ` in ${folderName}`}
          {result.total > result.shown && ` (showing the first ${result.shown})`}
          {result.understood.length > 0 && " · "}
          {result.understood.map((t) => (
            <span key={t} className="font-mono text-neutral-400 bg-neutral-900 border border-neutral-800 rounded px-1 mr-1">{t}</span>
          ))}
          {result.demoted > 0 && (
            <span className="text-neutral-600"> · the last {result.demoted} from shoots less important on this device</span>
          )}
          {result.unclear?.length > 0 && result.total > 0 && (
            <span className="text-amber-500/80" title="Nothing in the library matches the description clearly; these are only the nearest it has.">
              {" "}· no clear match for “{result.unclear.join("”, “")}” — these are the closest
            </span>
          )}
        </p>
      )}
      {help && (
        <div className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-0.5 text-xs border border-neutral-800 rounded p-3 max-w-2xl">
          {SEARCH_HELP.map(([example, meaning]) => (
            <Fragment key={example}>
              <button onClick={() => { onChange(example); onSubmit(example); }}
                className="font-mono text-left text-neutral-300 hover:text-white">{example}</button>
              <span className="text-neutral-500">{meaning}</span>
            </Fragment>
          ))}
          <span className="col-span-2 text-neutral-600 mt-1">Terms separated by spaces must all hold. A RAW+JPEG pair is one photo.</span>
        </div>
      )}
    </div>
  );
}

// ── Review progress ──────────────────────────────────────────────────
//
// How far through a folder the user is, and the way back to where they stopped.
// Counted per image rather than per file, because that is what the grid shows:
// deciding one half of a RAW+JPG pair decides the pair.

function ReviewProgress({ stats, resume, onResume, onClearResume, className = "" }) {
  if (!stats) return null;
  const total = stats.unique_images || stats.total || 0;
  if (!total) return null;
  const decided = stats.decided || 0;
  const left = Math.max(0, stats.undecided ?? total - decided);
  const pct = Math.min(100, Math.round((decided / total) * 100));

  return (
    <div className={`flex items-center gap-3 text-xs text-neutral-500 flex-wrap ${className}`} data-testid="review-progress">
      <div className="h-1.5 w-28 max-w-full bg-neutral-800 rounded overflow-hidden">
        <div className={`h-full ${left === 0 ? "bg-emerald-500" : "bg-emerald-700"}`}
          style={{ width: `${pct}%` }} />
      </div>
      <span>
        <span className="text-neutral-300">{decided}</span> of {total} decided
      </span>
      {left === 0
        ? <span className="text-emerald-500">This folder is reviewed</span>
        : <span>{left} still undecided</span>}
      {stats.picks > 0 && <span className="text-green-600">{stats.picks} picks</span>}
      {stats.rejects > 0 && <span className="text-red-600">{stats.rejects} rejected</span>}
      {resume && (
        <>
          <button onClick={() => onResume(resume)}
            title={`Open the cull view at ${resume.split("/").pop()}`}
            className="text-amber-300/90 hover:text-amber-200">
            Resume at {resume.split("/").pop()}
          </button>
          <button onClick={onClearResume} title="Forget the saved position"
            className="text-neutral-600 hover:text-white">✕</button>
        </>
      )}
    </div>
  );
}

// ── Acting on what is in view ───────────────────────────────────────
//
// The grid's filters define what is in view; these act on all of it at once, so
// a whole school event can be basketed or tagged without clicking photos. The
// slideshow and the basket stay on the bar; the culling help (bursts, faces,
// suggestions, measuring) and tagging open from it.

function FolderActions({ count, tags, busy, message, onAddToBasket, onSlideshow, onCheckFaces, onSuggestBest, facesChecked = false, onApplyTag, bursts = [], burstCoverage = null, onRejectDuplicates, quality = null, measuring = false, onMeasure, selecting = false, onSelect }) {
  const [tagDraft, setTagDraft] = useState("");
  const [bestCount, setBestCount] = useState(20);
  const canTag = tagDraft.trim().length > 0;

  const submitTag = (remove) => {
    if (!canTag || busy) return;
    onApplyTag(tagDraft.trim(), remove);
  };

  const disabled = busy || count === 0;
  // Quality warnings for the folder. Only worth saying when there is something
  // to say: a folder with no warnings should look exactly like it always did.
  const soft = quality ? quality.soft || 0 : 0;
  const blown = quality ? quality.blown || 0 : 0;
  // Any claim about a folder is only as good as the frames measured in it. When
  // frames are still unmeasured, that is the explanation worth giving: the
  // folder has not been read, not that there is nothing in it.
  const unmeasured = burstCoverage ? burstCoverage.unmeasured || 0 : 0;
  const assistNote = bursts.length > 0 ? bursts.length : (soft || blown) ? "⚠" : null;

  return (
    <div className="space-y-1">
      <div className="flex items-center gap-2 flex-wrap text-sm" data-testid="folder-actions">
        <span className="text-neutral-400 mr-1">
          <span className="text-neutral-200 font-medium">{count}</span> in view
        </span>

        {onSelect && count > 0 && (
          <button onClick={onSelect}
            className={`px-2.5 py-1 rounded border transition-colors ${selecting
              ? "border-sky-500 text-sky-300" : "border-neutral-700 text-neutral-300 hover:text-white"}`}
            title="Pick photos to rate, flag, tag or basket together (or Ctrl/Cmd-click, Shift-click)">
            {selecting ? "Done selecting" : "Select"}
          </button>
        )}

        {onSlideshow && count > 0 && (
          <button onClick={onSlideshow}
            className="px-2.5 py-1 rounded bg-neutral-800 text-sky-200 hover:bg-neutral-700 transition-colors"
            title="Show what is in view full screen, one picture at a time (Space pauses, ←/→ step, Esc leaves)"
          >▶ Slideshow</button>
        )}

        <button
          onClick={() => !disabled && onAddToBasket()}
          disabled={disabled}
          className="px-2.5 py-1 rounded bg-orange-900/50 text-orange-200 hover:bg-orange-900 disabled:opacity-40 transition-colors"
          title="Copy every photo matching the current filters into the export basket"
        >⬡ Basket {count}</button>

        <Menu testid="cull-assist" active={bursts.length > 0}
          title="Help with culling: near-identical bursts, faces, the frames most worth keeping"
          label={<>✦ Cull assist{assistNote != null && (
            <span className="ml-1.5 rounded-full bg-amber-600/80 px-1.5 text-[11px] text-white">{assistNote}</span>)}</>}>
          {(close) => (
            <>
              {bursts.length > 0 ? (
                <>
                  <MenuItem disabled={busy} onClick={() => { close(); onRejectDuplicates(); }}
                    title={`Keep the sharpest frame of each of the ${bursts.length} near-identical group${bursts.length === 1 ? "" : "s"} in this folder and reject the rest. Undo one photo with U.`}>
                    <span className="w-4 text-center text-amber-300">⧉</span>
                    Keep the {facesChecked ? "best" : "sharpest"} of {bursts.length} burst{bursts.length === 1 ? "" : "s"}, reject the rest
                  </MenuItem>
                  {onCheckFaces && !facesChecked && (
                    <MenuItem disabled={busy} onClick={() => { close(); onCheckFaces(); }}
                      title="Look at the faces in every burst: each then keeps the frame with its people at their best rather than the sharpest background, and a frame where someone's eyes closed is marked">
                      <span className="w-4 text-center text-sky-300">☺</span>
                      Check faces in the bursts first
                    </MenuItem>
                  )}
                </>
              ) : (
                <p className="px-2 py-1.5 text-xs text-neutral-500">No near-identical bursts in view.</p>
              )}
              {onSuggestBest && count > 0 && (
                <div className="flex items-center gap-2 rounded px-2 py-1 hover:bg-neutral-800">
                  <button disabled={busy} onClick={() => { close(); onSuggestBest(bestCount); }}
                    className="flex flex-1 items-center gap-2 py-0.5 text-left text-neutral-200 disabled:opacity-40"
                    title="Select the frames most worth keeping, one per moment: picks and stars first, then the people at their best and the sharpest; soft frames, blown highlights and closed eyes count against. Nothing is flagged — act on the selection yourself.">
                    <span className="w-4 text-center text-emerald-300">★</span>
                    Select the best
                  </button>
                  <input type="number" min={1} max={500} value={bestCount}
                    onChange={(e) => setBestCount(Math.max(1, Math.min(500, +e.target.value || 1)))}
                    className="w-14 bg-neutral-950 border border-neutral-700 rounded px-1 py-0.5 text-xs text-neutral-200"
                    title="How many to suggest" />
                </div>
              )}
              {(soft > 0 || blown > 0) && (
                <p className="px-2 py-1.5 text-xs text-neutral-400"
                  title="Measured warnings, not decisions. Marked on the thumbnails — nothing is rejected for you.">
                  ⚠ {soft > 0 && `${soft} soft`}
                  {soft > 0 && blown > 0 && " · "}
                  {blown > 0 && `${blown} blown out`}
                  <span className="text-neutral-600"> — marked on the thumbnails</span>
                </p>
              )}
              {unmeasured > 0 && (
                <>
                  <p className="px-2 pt-1.5 text-xs text-neutral-500 max-w-[19rem]"
                    title="Burst grouping and quality warnings are worked out from measurements the app takes in the background. Frames that have not been measured yet cannot appear in either.">
                    {unmeasured} frame{unmeasured === 1 ? "" : "s"} not measured yet, so not in the bursts or warnings.
                  </p>
                  {onMeasure && (
                    <MenuItem disabled={busy || measuring} onClick={() => onMeasure()}
                      title="Measure this folder now instead of waiting for its turn in the background queue. It reads the thumbnails already on disk and changes nothing in your library.">
                      <span className="w-4 text-center">▷</span>
                      {measuring ? "Measuring…" : "Measure this folder now"}
                    </MenuItem>
                  )}
                </>
              )}
            </>
          )}
        </Menu>

        <Menu label="Tag" testid="tag-menu" title="Tag or untag every photo in view">
          <div className="space-y-2 p-1.5">
            <input
              list="tag-suggestions"
              value={tagDraft}
              onChange={(e) => setTagDraft(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") submitTag(false); }}
              placeholder="Event tag…"
              autoFocus
              className="w-full bg-neutral-800 border border-neutral-700 rounded px-2 py-1 text-base sm:text-sm text-neutral-200"
            />
            <datalist id="tag-suggestions">
              {tags.map((t) => <option key={t.name} value={t.name} />)}
            </datalist>
            <div className="flex items-center gap-2">
              <button
                onClick={() => submitTag(false)}
                disabled={!canTag || busy || count === 0}
                className="px-2.5 py-1 rounded bg-neutral-700 text-neutral-100 hover:bg-neutral-600 disabled:opacity-40 transition-colors"
                title="Tag every photo in view. Writes keywords into the sidecar."
              >Tag {count}</button>
              <button
                onClick={() => submitTag(true)}
                disabled={!canTag || busy || count === 0}
                className="px-2 py-1 rounded text-neutral-400 hover:text-neutral-200 disabled:opacity-40"
                title="Remove this tag from every photo in view"
              >Untag</button>
            </div>
          </div>
        </Menu>

      </div>

      {message && <p className="text-xs text-neutral-400">{message}</p>}
    </div>
  );
}

// ── Acting on the photos picked by hand ─────────────────────────────
//
// The bulk bar above acts on everything the filters show; this one acts on the
// photos picked one by one (Ctrl/Cmd-click, Shift-click, or Select on a phone).

function SelectionBar({ photos, selected, tags, busy, message, selecting, onEdit, onBasket, onCollage, onSelectAll, onClear,
  copied, onPaste, onUndoPaste }) {
  const pasteMode = usePasteAs();
  const [tagDraft, setTagDraft] = useState("");
  const picked = photos.filter((p) => selected.has(p.filepath));
  // The tags the picked photos carry, most common first: what there is to remove.
  const present = useMemo(() => {
    const counts = new Map();
    for (const p of picked) for (const t of p.tags || []) counts.set(t, (counts.get(t) || 0) + 1);
    return [...counts.entries()].sort((a, b) => b[1] - a[1]);
  }, [picked]);
  const n = selected.size;
  if (!n && !selecting) return null;
  const off = busy || n === 0;
  const btn = "px-2 py-1 rounded bg-neutral-800 hover:bg-neutral-700 disabled:opacity-40";
  return (
    <div className="fixed inset-x-0 bottom-0 z-[900] bg-neutral-950/95 backdrop-blur border-t border-neutral-800 px-3 py-2"
      style={{ paddingBottom: "max(0.5rem, env(safe-area-inset-bottom))" }}>
      <div className="max-w-6xl mx-auto flex items-center gap-2 flex-wrap text-sm">
        <span className="text-neutral-200 font-medium min-w-[5.5rem]">
          {n ? `${n} selected` : "Tap photos to select"}
        </span>
        <span className="flex items-center gap-0.5" title="Stars for every selected photo (0 clears)">
          {[0, 1, 2, 3, 4, 5].map((r) => (
            <button key={r} disabled={off} onClick={() => onEdit({ rating: r })}
              className="w-7 h-7 rounded hover:bg-neutral-800 text-amber-300 disabled:opacity-40 text-xs">
              {r === 0 ? "☆0" : `${r}★`}
            </button>
          ))}
        </span>
        <button disabled={off} onClick={() => onEdit({ flag: 1 })} className={`${btn} text-green-300`} title="Pick">⚑ Pick</button>
        <button disabled={off} onClick={() => onEdit({ flag: -1 })} className={`${btn} text-red-300`} title="Reject">✕ Reject</button>
        <button disabled={off} onClick={() => onEdit({ flag: 0 })} className={`${btn} text-neutral-300`} title="Neither picked nor rejected">Unflag</button>
        <span className="flex items-center gap-1" title="Colour label">
          {[1, 2, 3, 4, 5].map((c) => (
            <button key={c} disabled={off} onClick={() => onEdit({ color_label: c })} aria-label={COLOR_NAMES[c]}
              className={`w-4 h-4 rounded-full ${COLOR_MAP[c]} disabled:opacity-40 hover:ring-2 ring-white/60`} />
          ))}
          <button disabled={off} onClick={() => onEdit({ color_label: 0 })} className="text-xs text-neutral-500 hover:text-white px-1">none</button>
        </span>
        <span className="flex items-center gap-1">
          <input list="tag-suggestions-sel" value={tagDraft} onChange={(e) => setTagDraft(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter" && tagDraft.trim() && !off) { onEdit({ add_tags: [tagDraft.trim()] }); setTagDraft(""); } }}
            placeholder="Add tag or name…"
            className="bg-neutral-900 border border-neutral-700 rounded px-2 py-1 text-base sm:text-sm text-neutral-200 w-40" />
          <datalist id="tag-suggestions-sel">
            {tags.map((t) => <option key={t.name} value={t.name} />)}
          </datalist>
          <button disabled={off || !tagDraft.trim()} onClick={() => { onEdit({ add_tags: [tagDraft.trim()] }); setTagDraft(""); }} className={btn}>Add</button>
        </span>
        {present.length > 0 && (
          <select value="" disabled={off}
            onChange={(e) => { if (e.target.value) onEdit({ remove_tags: [e.target.value] }); }}
            title="Take a tag off the selected photos. Taking off a person's name also takes their face out of these photos."
            className="bg-neutral-900 border border-neutral-700 rounded px-2 py-1 text-base sm:text-sm text-neutral-300">
            <option value="">Remove tag…</option>
            {present.map(([t, c]) => <option key={t} value={t}>{t} ({c})</option>)}
          </select>
        )}
        <button disabled={off} onClick={onBasket} className="px-2 py-1 rounded bg-orange-900/50 text-orange-200 hover:bg-orange-900 disabled:opacity-40">⬡ Basket</button>
        {/* The edit copied in the develop view (Ctrl+C there), onto the whole
            selection: a session, a search, a day. */}
        <button disabled={off || !copied} onClick={onPaste} data-testid="paste-edit"
          title={copied
            ? `Paste ${copied.sections.join(" + ")} from ${copied.source.split("/").pop()} onto the selected photos. The rest of each photo's edit stays its own.`
            : "Copy an edit in the develop view first (open a photo, press D, then Ctrl+C)"}
          className={`${btn} text-amber-200`}>⧉ Paste edit</button>
        {/* How the paste adjusts each photo: the same choice as the develop view's
            Paste menu (editClipboard.js). */}
        <select disabled={off || !copied} value={pasteMode} onChange={(e) => keepPasteAs(e.target.value)}
          data-testid="paste-as-grid" title={PASTE_AS.find((p) => p.id === pasteMode).title}
          className="bg-neutral-800 border border-neutral-700 rounded px-1 py-1 text-xs text-neutral-300 disabled:opacity-40">
          {PASTE_AS.map((opt) => <option key={opt.id} value={opt.id} title={opt.title}>{opt.short}</option>)}
        </select>
        {onCollage && (
          <button disabled={busy || n < 2} onClick={onCollage} data-testid="make-collage"
            title={n < 2 ? "Select 2 or more photos for a collage" : n > 9 ? "A collage holds up to 9: the first 9 go in, and any of the rest can be swapped in" : "These photos as one picture: a story, a square or a print"}
            className={`${btn} text-sky-200`}>▦ Collage</button>
        )}
        <span className="ml-auto flex items-center gap-2">
          <button onClick={onSelectAll} disabled={n === photos.length}
            className="text-xs text-neutral-400 hover:text-white disabled:opacity-40">Select all {photos.length}</button>
          {/* Esc does the same; this is where a mouse looks for it. */}
          <button onClick={onClear} title="Unselect everything and stay in this folder (Esc)"
            className={n ? `${btn} text-sky-200` : "text-xs text-neutral-400 hover:text-white"}>
            {n ? "Unselect all" : "Done"}
          </button>
        </span>
        {message && (
          <span className="w-full text-xs text-emerald-400">
            {message}
            {onUndoPaste && (
              <button onClick={onUndoPaste} disabled={busy} data-testid="undo-paste"
                className="ml-2 underline text-amber-200 hover:text-white disabled:opacity-40">Undo</button>
            )}
          </span>
        )}
      </div>
    </div>
  );
}

// ── Event tag chips ─────────────────────────────────────────────────

function TagChips({ tags, max = 2, size = "text-[9px]" }) {
  if (!tags || tags.length === 0) return null;
  const shown = tags.slice(0, max);
  return (
    <span className={`flex items-center gap-1 ${size}`}>
      {shown.map((t) => (
        <span key={t} className="px-1 py-0.5 rounded bg-sky-950/90 text-sky-200 border border-sky-900 max-w-24 truncate">
          {t}
        </span>
      ))}
      {tags.length > shown.length && <span className="text-sky-300/70">+{tags.length - shown.length}</span>}
    </span>
  );
}

// ── Image Grid (lazy loading) ───────────────────────────────────────

function ImageGrid({ photos, folderPath, onSelect, onRate, basketSet, onToggleBasket, burstSizes, hintByPath, proposalByPath = null, onDevelop, cellSize = THUMB_SIZES.M, selected = null, selecting = false, onPick = null }) {
  // Prefetch grid thumbnails for next batch as user scrolls
  const prefetchedRef = useRef(new Set());
  
  useEffect(() => {
    // Prefetch all grid thumbnails in batches of 50
    const unprefetched = photos
      .filter((p) => !prefetchedRef.current.has(p.filepath))
      .map((p) => p.filepath);

    for (let i = 0; i < unprefetched.length; i += 50) {
      const batch = unprefetched.slice(i, i + 50);
      setTimeout(() => prefetch(batch, "grid"), i * 10);
      batch.forEach((fp) => prefetchedRef.current.add(fp));
    }
  }, [photos]);

  return (
    <div className="grid gap-1.5"
      style={{ gridTemplateColumns: `repeat(auto-fill, minmax(min(${cellSize}px, 45vw), 1fr))` }}>
      {photos.map((photo, index) => {
        const burstSize = burstSizes ? burstSizes.get(photo.filepath) : 0;
        const hint = hintByPath ? hintByPath[photo.filepath] : null;
        const isPicked = !!selected && selected.has(photo.filepath);
        const anyPicked = selecting || (!!selected && selected.size > 0);
        return (
        <div
          key={photo.filepath}
          className={`group relative rounded overflow-hidden cursor-pointer
            ${photo.flag === -1 && !isPicked ? "opacity-35" : ""}
            ${isPicked ? "ring-2 ring-sky-400" : photo.flag === 1 ? "ring-2 ring-green-500/50" : ""}
          `}
        >
          <button
            onClick={(e) => {
              // Ctrl/Cmd-click picks a photo, Shift-click a run of them; once
              // anything is picked (or Select is on, for a phone) a plain click
              // picks too, and the cull view waits until the selection is gone.
              if (onPick && (anyPicked || e.ctrlKey || e.metaKey || e.shiftKey)) onPick(photo, index, e);
              else onSelect(photo);
            }}
            className="aspect-square w-full focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <LazyImage
              src={thumbUrl(photo.folder || folderPath, photo.filename, "grid", editKey(photo))}
              alt={photo.filename}
              className="w-full h-full"
            />
          </button>

          {/* Top overlay: flag + color + ext */}
          <div className="absolute top-1 left-1 flex items-center gap-1">
            <FlagBadge flag={photo.flag} />
            <ColorDot color={photo.color_label} />
            <ExtBadge ext={photo.extension} />
            <VideoBadge photo={photo} />
            {burstSize > 1 && (
              <span
                className="px-1 py-0.5 rounded bg-neutral-900/85 border border-neutral-700 text-amber-300/90 text-[9px] leading-none"
                title={`${burstSize} near-identical frames in a row — open one and press Shift+R to keep the sharpest`}
              >⧉{burstSize}</span>
            )}
            <HintBadges hint={hint} />
            <ProposalChip proposal={proposalByPath ? proposalByPath[photo.filepath] : null} />
            {/* A frame that carries an edit says so, so a folder worked on in
                the develop view can be told apart from one that was not. */}
            {photo.edited && (
              <span
                className="px-1 py-0.5 rounded bg-amber-900/80 border border-amber-700 text-amber-200 text-[9px] leading-none"
                title="This frame carries a develop edit — open it and press D to change it"
              >✎</span>
            )}
            {denoised(photo) && (
              <span
                className="px-1 py-0.5 rounded bg-sky-900/80 border border-sky-700 text-sky-200 text-[9px] leading-none font-semibold"
                title={`Noise reduction: luminance ${Number(photo.edit.nr_luma) || 0}, colour ${Number(photo.edit.nr_color) || 0}`}
              >NR</span>
            )}
          </div>

          {isPicked && <div className="absolute inset-0 bg-sky-400/15 pointer-events-none" />}
          {onPick && (
            <button
              onClick={(e) => { e.stopPropagation(); onPick(photo, index, e); }}
              className={`absolute top-1 right-8 w-6 h-6 rounded-full border-2 flex items-center justify-center text-xs transition-opacity
                ${isPicked ? "bg-sky-400 border-sky-400 text-black opacity-100"
                  : `bg-black/40 border-white/70 text-transparent ${anyPicked ? "opacity-100" : "opacity-0 group-hover:opacity-100"}`}`}
              title={isPicked ? "Unselect" : "Select (Ctrl-click, Shift-click for a run)"}
              aria-label="Select photo"
            >✓</button>
          )}

          {/* Develop, next to the basket toggle: the cull view is where a frame
              is judged, but the grid is where a folder is worked through, so
              the edit has to be one click from here too. */}
          {!isVideo(photo) && (
            <button
              onClick={(e) => { e.stopPropagation(); onDevelop?.(photo); }}
              className="absolute top-8 right-1 w-6 h-6 rounded-full flex items-center justify-center text-xs
                bg-black/50 text-neutral-400 opacity-0 group-hover:opacity-100 hover:text-amber-300"
              title="Develop this photo (D)"
            >✎</button>
          )}

          {/* Event tags — hidden while the hover overlay shows the filename */}
          <div className="absolute bottom-1 left-1 right-8 pointer-events-none group-hover:opacity-0 transition-opacity">
            <TagChips tags={photo.tags} />
          </div>

          {/* Basket toggle top-right */}
          <button
            onClick={(e) => { e.stopPropagation(); onToggleBasket(photo); }}
            className={`absolute top-1 right-1 w-6 h-6 rounded-full flex items-center justify-center text-xs transition-all
              ${basketSet.has(photo.filepath)
                ? "bg-orange-500 text-white opacity-100"
                : "bg-black/50 text-neutral-400 opacity-0 group-hover:opacity-100 hover:text-orange-300"}`}
            title={basketSet.has(photo.filepath) ? "Remove from export basket" : "Add to export basket"}
          >⬡</button>

          {/* Bottom overlay on hover */}
          <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent px-1.5 pb-1.5 pt-4 opacity-0 group-hover:opacity-100 transition-opacity">
            <Stars rating={photo.rating} onRate={(r) => onRate(photo.filepath, "rating", r)} />
            <p className="text-xs text-neutral-400 truncate mt-0.5">{photo.filename}</p>
          </div>
        </div>
        );
      })}
    </div>
  );
}

// ── Compare (2-up / 3-up) ───────────────────────────────────────────
//
// The burst strip answers "which frames are the same moment". This answers
// "which one of them do I want": every frame at the same size, in capture order,
// with the suggested keeper marked. Z swaps to a 1:1 centre crop of each frame at
// the same coordinates, which is the only way to compare focus on a 2000px
// preview without scrolling through it.
//
// It is deliberately a layout and not a second viewer: the frame with the white
// border is the photo the top bar, the star keys, and the basket all act on, so
// there is one notion of "the current photo" instead of two that can disagree.

// ── A video in the cull view ─────────────────────────────────────────
//
// The file itself is played: Edge on a PC plays a DJI Pocket's HEVC as it is. A
// browser that cannot (a phone, say) reports an error, and the player switches
// to a copy the server makes once for any browser (backend thumbnails/video.py),
// asking until it is ready. Focus goes back to the viewer, so the arrow keys
// still move through the folder and the number keys still rate.

function VideoPlayer({ photo, folderPath, videoRef, onFocusBack }) {
  const [compatible, setCompatible] = useState(false);
  const [note, setNote] = useState("");
  const path = photo.filepath.split("/").map(encodeURIComponent).join("/");
  const src = `${API}/video/${path}`;

  useEffect(() => {
    setCompatible(false);
    setNote("");
  }, [photo.filepath]);

  const onError = useCallback(async () => {
    if (compatible) { setNote("This video cannot be played in this browser."); return; }
    setNote("Making a copy this browser can play…");
    for (let tries = 0; tries < 600; tries += 1) {
      try {
        const res = await fetch(`${src}?compatible=1`, { headers: { Range: "bytes=0-0" } });
        if (res.status === 200 || res.status === 206) { setNote(""); setCompatible(true); return; }
        const state = res.status === 202 ? (await res.json()).state : "failed";
        if (state !== "working") {
          setNote(state === "unavailable" ? "This video cannot be played here: ffmpeg is not installed."
            : "This video cannot be played in this browser.");
          return;
        }
      } catch { /* the server is busy: ask again */ }
      await new Promise((r) => setTimeout(r, 2000));
    }
  }, [src, compatible]);

  return (
    <div className="relative max-h-full max-w-full flex items-center justify-center">
      <video
        key={`${photo.filepath}${compatible ? "?compatible" : ""}`}
        ref={videoRef}
        src={compatible ? `${src}?compatible=1` : src}
        poster={thumbUrl(photo.folder || folderPath, photo.filename, "preview")}
        controls
        preload="metadata"
        onError={onError}
        onFocus={onFocusBack}
        className={`max-h-[calc(100vh-9rem)] max-w-full ${photo.flag === -1 ? "opacity-40" : ""}`}
        data-testid="video-player"
      />
      {note && (
        <span className="absolute bottom-14 left-1/2 -translate-x-1/2 px-3 py-1.5 rounded bg-black/80 text-sm text-neutral-300">
          {note}
        </span>
      )}
    </div>
  );
}

function CompareGrid({ folderPath, frames, currentPath, zoom, onPick }) {
  // Two or three across, whichever makes the frames bigger. Four frames is a
  // 2x2 grid rather than a row of four, because a row of four is too small to
  // judge anything, which is the point of comparing them.
  const cols = frames.length === 2 ? 2 : frames.length <= 4 ? 2 : 3;
  const rows = Math.ceil(frames.length / cols);
  const scrolling = rows > 2;

  return (
    <div
      className="flex-1 min-h-0 grid gap-2 p-2 overflow-auto"
      style={{
        gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))`,
        gridTemplateRows: scrolling ? "auto" : `repeat(${rows}, minmax(0, 1fr))`,
      }}
    >
      {frames.map((frame) => (
        <button
          key={frame.filepath}
          onClick={() => onPick(frame.filepath)}
          title={frame.filename}
          style={scrolling ? { height: "45vh" } : undefined}
          className={`relative min-h-0 min-w-0 rounded overflow-hidden border bg-neutral-900 transition-all ${
            frame.filepath === currentPath
              ? "border-white"
              : "border-neutral-800 hover:border-neutral-500"
          }`}
        >
          <img
            src={thumbUrl(frame.folder || folderPath, frame.filename, "preview", editKey(frame))}
            alt={frame.filename}
            className={`w-full h-full ${zoom ? "object-none" : "object-contain"} ${
              frame.flag === -1 ? "opacity-40" : ""
            }`}
          />
          <span className="absolute left-1.5 top-1.5 px-1.5 py-0.5 rounded bg-neutral-900/85 border border-neutral-700 text-[10px] font-mono text-neutral-300">
            {frame.filename}
          </span>
          {frame.flag === -1 && (
            <span className="absolute right-1.5 top-1.5 px-1.5 py-0.5 rounded bg-red-950/90 border border-red-900 text-[10px] text-red-300">
              rejected
            </span>
          )}
          {frame.sharpest && frame.flag !== -1 && (
            <span className="absolute bottom-1.5 left-1.5 px-1.5 py-0.5 rounded bg-emerald-900/90 text-emerald-200 text-[10px]">
              sharpest
            </span>
          )}
          {frame.rating > 0 && (
            <span className="absolute bottom-1.5 right-1.5 text-[10px] text-yellow-400">
              {"★".repeat(frame.rating)}
            </span>
          )}
        </button>
      ))}
    </div>
  );
}

// ── Cull View ───────────────────────────────────────────────────────

// The faces of the photo on screen, drawn over it: who each one is, and a way to
// say who when the grouping did not. A face the scan never found — small, at the
// back of a hall — is what "Look closer" is for: it reads the photo at full size.
function FaceLayer({ photo, imgRef, onChanged }) {
  const [data, setData] = useState(null);
  const [named, setNamed] = useState([]);
  const [open, setOpen] = useState(null);       // the face_index being named
  const [draft, setDraft] = useState("");
  const [box, setBox] = useState(null);         // where the image sits, in px
  const [job, setJob] = useState(null);
  const [note, setNote] = useState("");
  // Drawing a face the detector missed: the drag in progress, then the box
  // waiting to be told who it is.
  const [drawing, setDrawing] = useState(false);
  const [drag, setDrag] = useState(null);
  const [pending, setPending] = useState(null);
  const overlayRef = useRef(null);

  const load = useCallback(() => {
    if (!photo) return;
    fetch(`${API}/faces/photo/${photo.filepath}`).then((r) => r.json())
      .then(setData).catch(() => setData(null));
  }, [photo?.filepath]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    setData(null); setOpen(null); setNote(""); setDrawing(false); setPending(null); load();
  }, [load]);

  // Escape leaves drawing, not the viewer: heard before the viewer hears it.
  useEffect(() => {
    if (!drawing && !pending) return;
    const onKey = (e) => {
      if (e.key !== "Escape") return;
      e.stopPropagation();
      e.preventDefault();
      setDrawing(false); setDrag(null); setPending(null); setOpen(null);
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [drawing, pending]);
  useEffect(() => {
    fetch(`${API}/faces/named`).then((r) => r.json()).then((d) => setNamed(d.people || [])).catch(() => {});
  }, []);

  // The overlay follows the image: it is sized by the window, not by the photo.
  useEffect(() => {
    const img = imgRef.current;
    if (!img) return;
    const measure = () => setBox({ left: img.offsetLeft, top: img.offsetTop,
      width: img.offsetWidth, height: img.offsetHeight });
    measure();
    img.addEventListener("load", measure);
    const observer = new ResizeObserver(measure);
    observer.observe(img);
    return () => { img.removeEventListener("load", measure); observer.disconnect(); };
  }, [imgRef, photo?.filepath]);

  // A closer look runs as the face job; follow it, then show what it found.
  useEffect(() => {
    if (!job || job.status !== "running") return;
    const timer = setInterval(async () => {
      try {
        const next = await fetch(`${API}/faces/scan/status`).then((r) => r.json());
        setJob(next);
        if (next.status !== "running") {
          load();
          onChanged?.();
          setNote(next.status === "completed"
            ? (next.found > 0 ? `Found ${next.found} more face${next.found === 1 ? "" : "s"}` : "No more faces at full size")
            : next.error || "Stopped");
        }
      } catch {}
    }, 1000);
    return () => clearInterval(timer);
  }, [job, load, onChanged]);

  const lookCloser = async () => {
    setNote("");
    const res = await fetch(`${API}/faces/closer`, { method: "POST",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify({ filepath: photo.filepath }) });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) { setNote(body.detail || "Could not look closer"); return; }
    setJob({ status: "running", done: 0, total: 1 });
  };

  const act = async (path, payload) => {
    const res = await fetch(`${API}${path}`, { method: "POST",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      setNote(body.detail || "That did not work");
      return;
    }
    setOpen(null);
    setDraft("");
    load();
    onChanged?.();
    fetch(`${API}/faces/named`).then((r) => r.json()).then((d) => setNamed(d.people || [])).catch(() => {});
  };

  const ref = (face) => ({ filepath: data.filepath, face_index: face.face_index });
  const personFor = (name) => named.find((p) => p.name.toLowerCase() === name.trim().toLowerCase());
  const assign = (face, name) => {
    const person = personFor(name);
    if (person) act("/faces/move", { faces: [ref(face)], into: person.id });
    else if (name.trim()) act("/faces/move", { faces: [ref(face)], name: name.trim() });
  };
  const assignDrawn = async (name) => {
    if (!pending || !name.trim()) return;
    const person = personFor(name);
    // Drawn on the picture shown, which may be cropped: the server puts it on the frame.
    await act("/faces/drawn", { filepath: photo.filepath, ...pending, shown: true,
      ...(person ? { into: person.id } : { name: name.trim() }) });
    setPending(null);
    setDrawing(false);
  };

  // The drag, in fractions of the photo, clamped to it.
  const at = (e) => {
    const r = overlayRef.current.getBoundingClientRect();
    return { x: Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)),
             y: Math.min(1, Math.max(0, (e.clientY - r.top) / r.height)) };
  };
  const rect = (a, b) => ({ x: Math.min(a.x, b.x), y: Math.min(a.y, b.y),
                            w: Math.abs(a.x - b.x), h: Math.abs(a.y - b.y) });
  const startDrag = (e) => {
    if (!drawing || e.button !== 0) return;
    e.preventDefault();
    const p = at(e);
    setPending(null);
    setOpen(null);
    setDrag({ from: p, to: p });
  };
  const moveDrag = (e) => { if (drag) setDrag({ ...drag, to: at(e) }); };
  const endDrag = (e) => {
    if (!drag) return;
    const box = rect(drag.from, at(e));
    setDrag(null);
    if (box.w < 0.01 || box.h < 0.01) return;      // a click, not a box
    setPending(box);
    setOpen("new");
    setDraft("");
  };

  const faces = data?.faces || [];
  const running = job && job.status === "running";
  const matches = draft.trim()
    ? named.filter((p) => p.name.toLowerCase().includes(draft.trim().toLowerCase())).slice(0, 6)
    : named.slice(0, 6);

  // "Who is this?" — for a face found or drawn, or a box just drawn.
  const picker = ({ current = "", below, onPick, extra = null }) => (
    <div onMouseDown={(e) => e.stopPropagation()}
      className={`absolute left-1/2 -translate-x-1/2 z-20 w-56 p-2 rounded border border-neutral-700 bg-neutral-900 shadow-xl space-y-1.5 pointer-events-auto ${
        below ? "top-full mt-6" : "bottom-full mb-6"}`}>
      <input autoFocus value={draft} onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => {
          e.stopPropagation();
          if (e.key === "Enter") onPick(draft);
          if (e.key === "Escape") { setOpen(null); setPending(null); }
        }}
        placeholder={current ? `Not ${current}? Who is it?` : "Who is this?"}
        className="w-full bg-neutral-800 border border-neutral-700 rounded px-2 py-1 text-xs text-neutral-100" />
      <div className="flex flex-wrap gap-1">
        {matches.filter((p) => p.name !== current).map((p) => (
          <button key={p.id} onClick={() => onPick(p.name)}
            className="px-1.5 py-0.5 rounded bg-neutral-800 hover:bg-emerald-900 text-[11px] text-neutral-200">{p.name}</button>
        ))}
        {draft.trim() && !personFor(draft) && (
          <button onClick={() => onPick(draft)}
            className="px-1.5 py-0.5 rounded bg-sky-900 hover:bg-sky-800 text-[11px] text-sky-100">+ New: {draft.trim()}</button>
        )}
      </div>
      {extra}
    </div>
  );

  const shown = drag ? rect(drag.from, drag.to) : pending;

  return (
    <>
      {box && (
        <div ref={overlayRef} style={box}
          onMouseDown={startDrag} onMouseMove={moveDrag} onMouseUp={endDrag}
          className={`absolute ${drawing ? "pointer-events-auto cursor-crosshair" : "pointer-events-none"}`}>
          {faces.map((f) => {
            // A box is kept on the whole frame; on a cropped or levelled photo the
            // server says where it sits on the picture shown (none: cropped out).
            const b = "shown" in f ? f.shown : f;
            if (!b) return null;
            const tone = f.name ? "border-emerald-400" : f.grouped ? "border-amber-400" : "border-neutral-300 border-dashed";
            const below = b.y + b.h < 0.8;
            return (
              <div key={f.face_index} className={`absolute ${drawing ? "pointer-events-none" : "pointer-events-auto"}`}
                style={{ left: `${b.x * 100}%`, top: `${b.y * 100}%`, width: `${b.w * 100}%`, height: `${b.h * 100}%` }}>
                <button onClick={() => { setOpen(open === f.face_index ? null : f.face_index); setDraft(""); }}
                  title={f.drawn ? `${f.name} (drawn by hand)`
                    : f.name || (f.grouped ? "Not named yet — click to say who" : "Not matched to anyone — click to say who")}
                  className={`w-full h-full border-2 rounded-sm ${tone} hover:bg-white/10`} />
                {/* Only names: a crowd of "?" would hide the photo. The box's
                    colour says the rest, and its tooltip what clicking does. */}
                {f.name && (
                  <span className={`absolute left-1/2 -translate-x-1/2 whitespace-nowrap text-[11px] px-1 rounded bg-black/70 text-emerald-200 ${
                    below ? "top-full mt-0.5" : "bottom-full mb-0.5"}`}>
                    {f.name}{f.drawn && <span className="text-neutral-500"> · drawn</span>}
                  </span>
                )}
                {open === f.face_index && picker({
                  current: f.name, below, onPick: (name) => assign(f, name),
                  extra: f.drawn ? (
                    <button onClick={() => act("/faces/drawn/remove", ref(f))}
                      className="text-[11px] text-red-300/80 hover:text-red-300">Remove this box</button>
                  ) : f.cluster_id > 0 && (
                    <button onClick={() => act("/faces/remove", { faces: [ref(f)] })}
                      className="text-[11px] text-red-300/80 hover:text-red-300">
                      {f.name ? `Not ${f.name}` : "Not this group"} — take it out
                    </button>
                  ),
                })}
              </div>
            );
          })}
          {shown && (
            <div className="absolute pointer-events-none"
              style={{ left: `${shown.x * 100}%`, top: `${shown.y * 100}%`, width: `${shown.w * 100}%`, height: `${shown.h * 100}%` }}>
              <div className="w-full h-full border-2 border-sky-400 border-dashed rounded-sm bg-sky-400/10" />
              {pending && open === "new" && picker({
                below: shown.y + shown.h < 0.8, onPick: assignDrawn,
                extra: <p className="text-[11px] text-neutral-500">Names the photo; does not teach recognition.</p>,
              })}
            </div>
          )}
        </div>
      )}
      <div className="absolute left-3 bottom-3 flex items-center gap-2 text-xs bg-black/70 rounded px-2 py-1 pointer-events-auto">
        <span className="text-neutral-300">
          {!data ? "…" : !data.scanned && !faces.length ? "Not looked at for faces yet"
            : `${faces.length} face${faces.length === 1 ? "" : "s"}`
              + (faces.some((f) => f.name) ? ` · ${faces.filter((f) => f.name).map((f) => f.name).join(", ")}` : "")}
        </span>
        <button onClick={lookCloser} disabled={running}
          title="Read this photo at full size: finds faces too small for the scan (a few seconds)"
          className="px-2 py-0.5 rounded bg-neutral-800 hover:bg-neutral-700 text-neutral-200 disabled:opacity-50">
          {running ? "Looking closer…" : "Look closer"}
        </button>
        <button onClick={() => { setDrawing((v) => !v); setPending(null); setOpen(null); }}
          title="Drag a box around someone the scan missed, then say who it is (Esc to stop)"
          className={`px-2 py-0.5 rounded text-neutral-200 ${drawing ? "bg-sky-800" : "bg-neutral-800 hover:bg-neutral-700"}`}>
          {drawing ? "Drag around the face…" : "Draw a face"}
        </button>
        {note && <span className="text-amber-200/90">{note}</span>}
      </div>
    </>
  );
}

function CullView({ photos, folderPath, startIndex, onClose, onRate, basketSet, onToggleBasket, allTags = [], onSetTags, burst = null, hint = null, proposal = null, onProposal = null, onRejectBurst, onDevelop, developing = false, syncToken = 0, onFacesChanged }) {
  const [index, setIndex] = useState(startIndex);
  const [showExif, setShowExif] = useState(false);
  const [compare, setCompare] = useState(false);
  const [zoom, setZoom] = useState(false);
  const [exif, setExif] = useState(null);
  const [tagDraft, setTagDraft] = useState("");
  const [showFaces, setShowFaces] = useStoredState("photool.viewer.faces", false);
  const phone = usePhone();
  const imgRef = useRef(null);
  const videoRef = useRef(null);
  const containerRef = useRef(null);
  const tagInputRef = useRef(null);
  const prefetchedRef = useRef(new Set());

  const photo = photos[index];
  useEffect(() => { setIndex(startIndex); }, [startIndex, syncToken]);
  const photoByPath = useMemo(() => new Map(photos.map((p) => [p.filepath, p])), [photos]);

  // Focus is how this view hears the keyboard, so it has to be taken back when
  // the develop view that covered it closes — otherwise the first key pressed
  // after developing goes nowhere.
  useEffect(() => {
    if (!developing) containerRef.current?.focus();
  }, [developing]);

  // A new photo should never inherit the half-typed tag of the previous one.
  useEffect(() => { setTagDraft(""); }, [photo?.filepath]);

  const replaceTags = useCallback(
    (next) => { if (photo) onSetTags(photo.filepath, next); },
    [photo, onSetTags]
  );

  const addTag = useCallback(() => {
    const name = tagDraft.trim();
    if (!name || !photo) return;
    const current = photo.tags || [];
    if (current.some((t) => t.toLowerCase() === name.toLowerCase())) { setTagDraft(""); return; }
    replaceTags([...current, name]);
    setTagDraft("");
  }, [tagDraft, photo, replaceTags]);

  // Fetch EXIF
  useEffect(() => {
    if (!photo) return;
    fetch(`${API}/exif/${photo.folder || folderPath}/${encodeURIComponent(photo.filename)}`)
      .then((r) => r.json()).then(setExif).catch(() => setExif(null));
  }, [photo?.filename, photo?.folder, folderPath]);

  // Prefetch preview thumbnails ahead of current position (YouTube-style buffer)
  useEffect(() => {
    if (!photo) return;
    const BUFFER_AHEAD = 10;
    const BUFFER_BEHIND = 3;
    const start = Math.max(0, index - BUFFER_BEHIND);
    const end = Math.min(photos.length, index + BUFFER_AHEAD);
    const toFetch = [];

    for (let i = start; i < end; i++) {
      const fp = photos[i].filepath;
      if (!prefetchedRef.current.has(fp)) {
        toFetch.push(fp);
        prefetchedRef.current.add(fp);
      }
    }
    if (toFetch.length > 0) {
      prefetch(toFetch, "preview");
    }
  }, [index, photos]);

  const go = useCallback(
    (delta) => setIndex((i) => Math.max(0, Math.min(photos.length - 1, i + delta))),
    [photos.length]
  );

  // Jump to a frame by path — used by the burst strip. A frame the current
  // filter hides is not in the list, so this is a no-op rather than a jump.
  const jumpTo = useCallback(
    (filepath) => {
      const i = photos.findIndex((p) => p.filepath === filepath);
      if (i >= 0) setIndex(i);
    },
    [photos]
  );

  const rate = useCallback(
    (field, value) => { if (photo) onRate(photo.filepath, field, value); },
    [photo, onRate]
  );

  // Shift+R: keep one frame of this burst and reject the rest. The keeper is
  // the sharpest frame the measurements found, not whichever one is on screen.
  // A burst's member flags come from the grouping query, which predates anything
  // the user has flagged since, so prefer the live value from the list.
  const memberFlag = useCallback(
    (member) => {
      const live = photos.find((p) => p.filepath === member.filepath);
      return live ? live.flag : member.flag;
    },
    [photos]
  );

  const rejectBurst = useCallback(() => {
    if (!burst || !onRejectBurst) return;
    const keep = burst.keep || photo?.filepath;
    const others = (burst.members || [])
      .filter((m) => m.filepath !== keep && memberFlag(m) !== -1)
      .map((m) => m.filepath);
    if (others.length > 0) onRejectBurst(keep, others);
  }, [burst, photo, onRejectBurst, memberFlag]);

  // The frames to lay side by side, with the flags and ratings as they are right
  // now rather than as they were when the grouping was fetched.
  const compareFrames = useMemo(() => {
    if (!compare || !burst) return [];
    return (burst.members || []).map((m) => {
      const live = photos.find((p) => p.filepath === m.filepath);
      return {
        filepath: m.filepath,
        filename: m.filename,
        flag: live ? live.flag : m.flag,
        rating: live ? live.rating : 0,
        edit: live ? live.edit : undefined,
        sharpest: m.filepath === burst.keep,
      };
    });
  }, [compare, burst, photos]);

  const showCompare = compare && compareFrames.length >= 2;

  // If the grouping changes underneath the compare view — a filter, a re-index —
  // fall back to the single frame instead of holding on to a set that is no
  // longer there.
  useEffect(() => {
    if (!compare) return;
    const inSet = (burst?.members || []).some((m) => m.filepath === photo?.filepath);
    if (!inSet) {
      setCompare(false);
      setZoom(false);
    }
  }, [compare, burst, photo]);

  // The frames about to be compared are what the user is going to look at, and
  // they are not always next to each other in the list, so the buffer that
  // follows the cull position would not cover them.
  useEffect(() => {
    if (!showCompare) return;
    const missing = compareFrames
      .map((f) => f.filepath)
      .filter((fp) => !prefetchedRef.current.has(fp));
    if (!missing.length) return;
    missing.forEach((fp) => prefetchedRef.current.add(fp));
    prefetch(missing, "preview");
  }, [showCompare, compareFrames]);

  // Inside compare the arrow keys move between the frames being compared, not
  // through the folder: showing a frame that is not on screen would break the
  // one thing the layout exists for.
  const step = useCallback(
    (delta) => {
      if (!showCompare) {
        go(delta);
        return;
      }
      const at = compareFrames.findIndex((f) => f.filepath === photo?.filepath);
      if (at < 0) {
        jumpTo(compareFrames[0].filepath);
        return;
      }
      const next = Math.max(0, Math.min(compareFrames.length - 1, at + delta));
      if (next !== at) jumpTo(compareFrames[next].filepath);
    },
    [showCompare, compareFrames, photo, go, jumpTo]
  );

  // Shift+R in compare mode keeps the frame you are looking at rather than the
  // one the measurements prefer. You have just compared them side by side, so
  // your eye is the better instrument.
  const keepThisFrame = useCallback(() => {
    if (!burst || !onRejectBurst || !photo) return;
    const others = (burst.members || [])
      .filter((m) => m.filepath !== photo.filepath && memberFlag(m) !== -1)
      .map((m) => m.filepath);
    if (others.length > 0) onRejectBurst(photo.filepath, others);
  }, [burst, photo, onRejectBurst, memberFlag]);

  // Compare always starts fitted to the frame. The 1:1 crop is a deliberate
  // second look, not a setting that should survive leaving the view.
  const toggleCompare = useCallback(() => {
    if (!burst) return;
    setCompare(!compare);
    if (compare) setZoom(false);
  }, [burst, compare]);

  const handleKey = useCallback(
    (e) => {
      if (!photo) return;
      // The develop view sits on top of this one and has its own keys; this
      // view is still mounted underneath, and must not act on them as well.
      if (developing) return;
      // Typing in the tag box is text, not navigation.
      if (e.target.tagName === "INPUT") {
        if (e.key === "Escape") e.target.blur();
        return;
      }
      switch (e.key) {
        case "Escape":
          // Out of compare first, then out of the cull view.
          if (compare) {
            setCompare(false);
            setZoom(false);
          } else {
            onClose();
          }
          break;
        case "ArrowRight": case "ArrowDown": case "j": step(1); break;
        case "ArrowLeft": case "ArrowUp": case "k": step(-1); break;
        case "0": rate("rating", 0); break;
        case "1": rate("rating", 1); break;
        case "2": rate("rating", 2); break;
        case "3": rate("rating", 3); break;
        case "4": rate("rating", 4); break;
        case "5": rate("rating", 5); break;
        case "p": case "P": rate("flag", photo.flag === 1 ? 0 : 1); break;
        case "x": case "X": rate("flag", photo.flag === -1 ? 0 : -1); break;
        case "u": case "U": rate("flag", 0); break;
        // Shift+R is uppercase R; plain r is left free.
        case "R": (showCompare ? keepThisFrame : rejectBurst)(); break;
        case "c": case "C": toggleCompare(); break;
        // Develop opens on the frame being looked at, which is the whole point
        // of judging it here first: the cull view decides whether a picture is
        // worth keeping, and the develop view decides how it should look.
        case "d": case "D": if (!isVideo(photo)) onDevelop?.(photo); break;
        // A video plays and pauses on the space bar, as in any player.
        case " ": {
          const v = videoRef.current;
          if (v) { e.preventDefault(); if (v.paused) v.play().catch(() => {}); else v.pause(); }
          break;
        }
        case "z": case "Z": if (showCompare) setZoom((v) => !v); break;
        case "6": rate("color_label", photo.color_label === 1 ? 0 : 1); break;
        case "7": rate("color_label", photo.color_label === 2 ? 0 : 2); break;
        case "8": rate("color_label", photo.color_label === 3 ? 0 : 3); break;
        case "9": rate("color_label", photo.color_label === 4 ? 0 : 4); break;
        case "i": case "I": setShowExif((s) => !s); break;
        case "f": case "F": setShowFaces((v) => !v); break;
        case "b": case "B": onToggleBasket(photo); break;
        case "t": case "T": tagInputRef.current?.focus(); break;
        case "]": rate("flag", 1); setTimeout(() => step(1), 50); break;
        // An agent's suggestion for this frame: A takes it, N sets it aside.
        case "a": case "A": if (proposal && onProposal) onProposal("accept", photo); break;
        case "n": case "N": if (proposal && onProposal) onProposal("dismiss", photo); break;
        case "[": rate("flag", -1); setTimeout(() => step(1), 50); break;
      }
    },
    [photo, step, rate, onClose, onToggleBasket, rejectBurst, keepThisFrame, toggleCompare, showCompare, compare, developing, onDevelop, setShowFaces, proposal, onProposal]
  );

  if (!photo) return null;

  if (phone) {
    return (
      <PhoneCull photos={photos} index={index} folderPath={folderPath} photo={photo} onGo={go} onClose={onClose}
        rate={rate} basketSet={basketSet} onToggleBasket={onToggleBasket} exif={exif} burst={burst}
        photoByPath={photoByPath} memberFlag={memberFlag} onJump={jumpTo} onRejectBurst={rejectBurst}
        hint={hint} proposal={proposal} onProposal={onProposal} imgRef={imgRef} videoRef={videoRef} />
    );
  }

  return (
    <div
      ref={containerRef} tabIndex={0} onKeyDown={handleKey} data-modal="viewer"
      className="fixed inset-0 z-50 bg-black flex flex-col focus:outline-none"
    >
      {/* Top bar */}
      <div className="flex items-center justify-between px-4 py-2.5 bg-neutral-950/90 border-b border-neutral-800 z-10">
        <div className="flex items-center gap-4">
          <button onClick={onClose} className="text-neutral-400 hover:text-white text-sm">← Grid</button>
          <span className="text-neutral-500 text-sm font-mono">{index + 1} / {photos.length}</span>
          <FlagBadge flag={photo.flag} size="text-sm" />
          <ColorDot color={photo.color_label} size="w-3 h-3" />
          <ExtBadge ext={photo.extension} />
        </div>
        <div className="flex items-center gap-4">
          <Stars rating={photo.rating} onRate={(r) => rate("rating", r)} size="text-lg" />
          <span className="text-neutral-500 text-sm font-mono">{photo.filename}</span>
          <div className="flex gap-1">
            <button onClick={() => rate("flag", photo.flag === 1 ? 0 : 1)}
              className={`px-2 py-1 rounded text-xs font-bold ${photo.flag === 1 ? "bg-green-900 text-green-300" : "text-neutral-600 hover:text-green-400"}`}>P</button>
            <button onClick={() => rate("flag", photo.flag === -1 ? 0 : -1)}
              className={`px-2 py-1 rounded text-xs font-bold ${photo.flag === -1 ? "bg-red-900 text-red-300" : "text-neutral-600 hover:text-red-400"}`}>✕</button>
          </div>
          <div className="flex gap-1">
            {[1, 2, 3, 4, 5].map((c) => (
              <button key={c} onClick={() => rate("color_label", photo.color_label === c ? 0 : c)}
                className={`w-4 h-4 rounded-full border-2 transition-all ${COLOR_MAP[c]} ${
                  photo.color_label === c ? "border-white scale-125" : "border-transparent opacity-40 hover:opacity-80"
                }`} />
            ))}
          </div>
          <button onClick={() => setShowExif((s) => !s)}
            className={`text-xs px-2 py-1 rounded ${showExif ? "bg-neutral-700 text-white" : "text-neutral-500 hover:text-white"}`}>EXIF</button>
          <button onClick={() => setShowFaces((v) => !v)} title="Show who is in this photo (F)"
            className={`text-xs px-2 py-1 rounded ${showFaces ? "bg-neutral-700 text-white" : "text-neutral-500 hover:text-white"}`}>Faces</button>
          {onDevelop && !isVideo(photo) && (
            <button onClick={() => onDevelop(photo)}
              title="Open this frame in the develop view (D)"
              className={`text-xs px-2 py-1 rounded ${photo.edited ? "text-amber-400 hover:text-amber-300" : "text-neutral-500 hover:text-white"}`}>
              Develop{photo.edited ? " ●" : ""} <span className="text-neutral-600">D</span>
            </button>
          )}
          <button onClick={() => onToggleBasket(photo)}
            className={`text-xs px-2 py-1 rounded font-bold transition-colors ${
              basketSet.has(photo.filepath) ? "bg-orange-600 text-white" : "text-neutral-500 hover:text-orange-400"
            }`}>⬡ {basketSet.has(photo.filepath) ? "In Basket" : "Basket"}</button>
        </div>
      </div>

      {/* Event tags — T focuses the box, Enter adds, Escape leaves it */}
      <div className="flex items-center gap-2 px-4 py-1.5 bg-neutral-950/80 border-b border-neutral-800 z-10 flex-wrap">
        <span className="text-[11px] uppercase tracking-wide text-neutral-600">Event tags</span>
        {(photo.tags || []).map((t) => (
          <span key={t} className="flex items-center gap-1 px-1.5 py-0.5 rounded bg-sky-950 border border-sky-900 text-xs text-sky-200">
            {t}
            <button onClick={() => replaceTags((photo.tags || []).filter((x) => x !== t))}
              className="text-sky-600 hover:text-white">✕</button>
          </span>
        ))}
        <input
          ref={tagInputRef}
          list="cull-tag-suggestions"
          value={tagDraft}
          onChange={(e) => setTagDraft(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); addTag(); } }}
          placeholder="Add tag, Enter"
          className="bg-neutral-800 border border-neutral-700 rounded px-2 py-0.5 text-xs text-neutral-200 w-40"
        />
        <datalist id="cull-tag-suggestions">
          {allTags.map((t) => <option key={t.name} value={t.name} />)}
        </datalist>
      </div>

      {/* Main image — or, in compare mode, every frame of this burst at once. */}
      {showCompare ? (
        <CompareGrid
          folderPath={folderPath}
          frames={compareFrames}
          currentPath={photo.filepath}
          zoom={zoom}
          onPick={jumpTo}
        />
      ) : (
        <div className="flex-1 flex items-center justify-center relative min-h-0">
          {isVideo(photo) ? (
            <VideoPlayer photo={photo} folderPath={folderPath} videoRef={videoRef}
              onFocusBack={() => containerRef.current?.focus()} />
          ) : (
            <img
              ref={imgRef}
              src={thumbUrl(photo.folder || folderPath, photo.filename, "preview", editKey(photo))}
              alt={photo.filename}
              className={`max-h-full max-w-full object-contain ${photo.flag === -1 ? "opacity-40" : ""}`}
            />
          )}
          {showFaces && !isVideo(photo) && <FaceLayer photo={photo} imgRef={imgRef} onChanged={onFacesChanged} />}
          {index > 0 && (
            <button onClick={() => go(-1)} className="absolute left-3 top-1/2 -translate-y-1/2 text-white/30 hover:text-white text-5xl">‹</button>
          )}
          {index < photos.length - 1 && (
            <button onClick={() => go(1)} className="absolute right-3 top-1/2 -translate-y-1/2 text-white/30 hover:text-white text-5xl">›</button>
          )}
        </div>
      )}

      {/* EXIF bar */}
      {showExif && exif && (
        <div className="px-5 py-3 bg-neutral-950/95 border-t border-neutral-800 text-sm font-mono">
          <div className="flex flex-wrap gap-x-6 gap-y-1 justify-center text-neutral-400">
            {/* The catalog's date is corrected for a camera clock set to another zone;
                the file's own is offered on hover when the two differ. */}
            {(photo.capture_date || exif.date) && (
              <Stat label="Taken" value={takenAt(photo.capture_date || exif.date)}
                title={exif.date && photo.capture_date && exif.date !== photo.capture_date
                  ? `Camera clock: ${takenAt(exif.date)}` : undefined} />
            )}
            {exif.camera && <Stat label="Camera" value={exif.camera} />}
            {exif.lens && <Stat label="Lens" value={exif.lens} />}
            {exif.focal_length && <Stat label="FL" value={exif.focal_length} />}
            {exif.aperture && <Stat label="Aperture" value={exif.aperture} />}
            {exif.shutter_speed && <Stat label="Shutter" value={exif.shutter_speed} />}
            {exif.iso && <Stat label="ISO" value={exif.iso} />}
            {exif.dimensions && <Stat label="Size" value={exif.dimensions} />}
          </div>
        </div>
      )}

      {/* Burst: the frames that look the same, with the suggested keeper marked.
          Keeping is implicit — Shift+R rejects the rest and leaves this one.
          Hidden while comparing, because the grid below is the same set. */}
      {burst && !showCompare && (
        <div className="px-4 py-2 bg-neutral-950/95 border-t border-neutral-800 flex items-center gap-3 z-10">
          <span className="text-[11px] uppercase tracking-wide text-neutral-600 shrink-0">
            Same shot ×{burst.size}
          </span>
          <div className="flex gap-1.5 overflow-x-auto">
            {(burst.members || []).map((m) => (
              <button key={m.filepath} onClick={() => jumpTo(m.filepath)} title={m.filename}
                className={`relative shrink-0 rounded overflow-hidden border transition-all
                  ${m.filepath === photo.filepath ? "border-white" : "border-neutral-800 hover:border-neutral-600"}
                  ${memberFlag(m) === -1 ? "opacity-30" : ""}`}>
                <img src={thumbUrl(m.folder || folderPath, m.filename, "grid", editKey(photoByPath.get(m.filepath)))} alt={m.filename} className="h-14 w-14 object-cover" />
                {memberFlag(m) !== -1 && m.filepath === (burst.keep || photo.filepath) && (
                  <span className="absolute bottom-0 inset-x-0 text-[9px] leading-tight text-center bg-emerald-900/90 text-emerald-200">
                    sharpest
                  </span>
                )}
              </button>
            ))}
          </div>
          <button onClick={() => setCompare(true)}
            className="ml-auto shrink-0 px-2.5 py-1 rounded bg-neutral-800 text-neutral-300 hover:bg-neutral-700 hover:text-white text-xs">
            Compare
          </button>
          <button onClick={rejectBurst}
            className="shrink-0 px-2.5 py-1 rounded bg-red-900/60 text-red-200 hover:bg-red-900 text-xs">
            Reject the rest
          </button>
        </div>
      )}

      {/* Compare toolbar. The strip above is the same set laid out large, so it
          is replaced rather than repeated. */}
      {showCompare && (
        <div className="px-4 py-2 bg-neutral-950/95 border-t border-neutral-800 flex items-center gap-3 z-10">
          <span className="text-[11px] uppercase tracking-wide text-neutral-600 shrink-0">
            Comparing ×{compareFrames.length}
          </span>
          <button onClick={() => setZoom((v) => !v)}
            className={`shrink-0 px-2.5 py-1 rounded text-xs ${
              zoom ? "bg-neutral-700 text-white" : "bg-neutral-800 text-neutral-300 hover:bg-neutral-700 hover:text-white"
            }`}
            title="Fill each frame with the same centre crop at full size, so focus can be compared directly">
            {zoom ? "Fit frame" : "1:1 crop"}
          </button>
          <button onClick={keepThisFrame}
            className="shrink-0 px-2.5 py-1 rounded bg-red-900/60 text-red-200 hover:bg-red-900 text-xs">
            Keep this one, reject the rest
          </button>
          <span className="ml-auto shrink-0 text-[11px] text-neutral-600">
            The white border is the frame the keys act on
          </span>
        </div>
      )}

      {/* Quality warnings. Measured off the frame, not judged: a soft frame can
          be the right photo and clipped sky can be what you wanted. Nothing here
          rejects anything for you — it only says where to look closer. */}
      {hint && (hint.soft_frame || hint.blown_highlights || hint.eyes_closed) && (
        <div className="px-4 py-1.5 bg-neutral-950/95 border-t border-neutral-800 text-xs flex items-center gap-5">
          {hint.soft_frame && (
            <span className="text-amber-300/90">⚠ Softer than the other frames in this folder</span>
          )}
          {hint.blown_highlights && (
            <span className="text-red-300/90">⚠ Highlights clipped to white</span>
          )}
          {hint.eyes_closed && (
            <span className="text-sky-300/90">⚠ Someone's eyes may be closed (against the rest of this burst)</span>
          )}
        </div>
      )}

      {/* An agent's suggestion, apart from the marks above until it is taken. */}
      {proposal && onProposal && (
        <div data-testid="cull-proposal"
          className="px-4 py-1.5 bg-violet-950/70 border-t border-violet-900 text-xs flex items-center gap-3">
          <span className="text-violet-200 shrink-0">✦ {proposal.agent || "Agent"} suggests {proposalText(proposal)}</span>
          {proposal.reason && <span className="text-violet-300/80 truncate">— {proposal.reason}</span>}
          <button onClick={() => onProposal("accept", photo)}
            className="ml-auto shrink-0 px-2 py-0.5 rounded bg-violet-700 text-white hover:bg-violet-600">Accept (A)</button>
          <button onClick={() => onProposal("dismiss", photo)}
            className="shrink-0 px-2 py-0.5 rounded border border-neutral-700 text-neutral-300 hover:text-white">Dismiss (N)</button>
        </div>
      )}

      {/* Keyboard hints. Compare has its own set, because the arrow keys and
          Shift+R mean something different there. */}
      <div className="px-4 py-1.5 bg-neutral-950 border-t border-neutral-800 text-neutral-600 text-xs flex justify-center gap-4">
        {showCompare ? (
          <>
            <span>← → switch frame</span><span>1-5 rate</span><span>P pick</span>
            <span>X reject</span><span>Z 1:1 crop</span><span>C fit</span>
            <span>B basket</span><span>T tag</span><span>I exif</span>
            <span className="text-amber-400/80">Shift+R keep this one</span>
            <span>Esc one frame</span>
          </>
        ) : (
          <>
            <span>← → navigate</span><span>1-5 rate</span><span>P pick</span>
            <span>X reject</span><span>] pick+next</span><span>[ rej+next</span>
            <span>6-9 color</span><span>B basket</span><span>T tag</span><span>F faces</span><span>I exif</span><span>Esc grid</span>
            {burst && <span className="text-amber-400/80">Shift+R keep sharpest</span>}
            {burst && <span className="text-neutral-500">C compare</span>}
          </>
        )}
      </div>
    </div>
  );
}

// Gestures on the photo: swipe sideways for the next or previous one, or tap
// the screen's left or right edge; swipe down to leave; pinch or double-tap to
// look closer, then drag about. At the whole picture a swipe moves on; zoomed
// in, a drag moves the picture instead.
const EDGE = 0.2;          // the share of the width at each side a tap moves on from
function usePhotoGestures({ onNext, onPrev, onClose, canNext, canPrev, resetKey }) {
  const [view, setView] = useState({ s: 1, x: 0, y: 0 });
  const [drag, setDrag] = useState({ x: 0, y: 0 });
  // A photo leaving slides on the way it was pushed and fades; the next fades in.
  const [anim, setAnim] = useState({ x: 0, o: 1, t: "" });
  const pointers = useRef(new Map());
  const start = useRef(null);
  const lastTap = useRef({ t: 0, x: 0, y: 0 });
  const tapTimer = useRef(null);
  const leaving = useRef(false);
  const viewRef = useRef(view);
  viewRef.current = view;
  useEffect(() => {
    setView({ s: 1, x: 0, y: 0 });
    setDrag({ x: 0, y: 0 });
    if (!leaving.current) return undefined;
    leaving.current = false;
    setAnim({ x: 0, o: 0, t: "none" });
    let raf = requestAnimationFrame(() => {
      raf = requestAnimationFrame(() => setAnim({ x: 0, o: 1, t: "opacity 0.18s ease-out" }));
    });
    return () => cancelAnimationFrame(raf);
  }, [resetKey]);
  useEffect(() => () => clearTimeout(tapTimer.current), []);

  // dir +1 for the next photo, -1 for the previous; from where the finger left it.
  const leave = (dir, width, from = 0) => {
    if (leaving.current) return;
    if (dir > 0 ? !canNext : !canPrev) {
      setDrag({ x: 0, y: 0 });
      return;
    }
    leaving.current = true;
    setDrag({ x: 0, y: 0 });
    setAnim({ x: from, o: 1, t: "none" });
    requestAnimationFrame(() => setAnim({ x: -dir * width * 0.6, o: 0,
      t: "transform 0.18s ease-out, opacity 0.18s linear" }));
    setTimeout(() => (dir > 0 ? onNext : onPrev)(), 180);
  };

  const centre = (el) => {
    const r = el.getBoundingClientRect();
    return { cx: r.left + r.width / 2, cy: r.top + r.height / 2, w: r.width, h: r.height };
  };
  const clampView = (v, box) => {
    const mx = ((v.s - 1) * box.w) / 2;
    const my = ((v.s - 1) * box.h) / 2;
    return { s: v.s, x: Math.max(-mx, Math.min(mx, v.x)), y: Math.max(-my, Math.min(my, v.y)) };
  };

  const onPointerDown = (e) => {
    e.currentTarget.setPointerCapture?.(e.pointerId);
    pointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    const pts = [...pointers.current.values()];
    const box = centre(e.currentTarget);
    if (pts.length === 2) {
      const [a, b] = pts;
      start.current = { kind: "pinch", dist: Math.hypot(a.x - b.x, a.y - b.y), view: viewRef.current,
        mid: { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }, box };
      setDrag({ x: 0, y: 0 });
    } else if (pts.length === 1) {
      start.current = { kind: viewRef.current.s > 1 ? "pan" : "swipe", x: e.clientX, y: e.clientY,
        view: viewRef.current, box, t: e.timeStamp };
    }
  };
  const onPointerMove = (e) => {
    if (!pointers.current.has(e.pointerId) || !start.current) return;
    pointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    const st = start.current;
    if (st.kind === "pinch" && pointers.current.size >= 2) {
      const [a, b] = [...pointers.current.values()];
      const s = Math.max(1, Math.min(6, (st.view.s * Math.hypot(a.x - b.x, a.y - b.y)) / st.dist));
      // The point under the fingers' middle stays under it.
      const qx = (st.mid.x - st.box.cx - st.view.x) / st.view.s;
      const qy = (st.mid.y - st.box.cy - st.view.y) / st.view.s;
      const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
      setView(clampView({ s, x: mid.x - st.box.cx - s * qx, y: mid.y - st.box.cy - s * qy }, st.box));
    } else if (st.kind === "pan") {
      setView(clampView({ s: st.view.s, x: st.view.x + e.clientX - st.x, y: st.view.y + e.clientY - st.y }, st.box));
    } else if (st.kind === "swipe") {
      setDrag({ x: e.clientX - st.x, y: Math.max(0, e.clientY - st.y) });
    }
  };
  const onPointerUp = (e) => {
    pointers.current.delete(e.pointerId);
    const st = start.current;
    if (!st) return;
    if (st.kind === "pinch") {
      if (pointers.current.size === 0) start.current = null;
      if (viewRef.current.s < 1.05) setView({ s: 1, x: 0, y: 0 });
      return;
    }
    start.current = null;
    const dx = e.clientX - st.x;
    const dy = e.clientY - st.y;
    const moved = Math.hypot(dx, dy);
    if (moved < 10 && e.timeStamp - st.t < 300) {
      // A double tap: in to 2.5x about the tap, or back out to the whole picture.
      const prev = lastTap.current;
      if (e.timeStamp - prev.t < 320 && Math.hypot(e.clientX - prev.x, e.clientY - prev.y) < 30) {
        clearTimeout(tapTimer.current);
        lastTap.current = { t: 0, x: 0, y: 0 };
        if (viewRef.current.s > 1) setView({ s: 1, x: 0, y: 0 });
        else {
          const d = { x: e.clientX - st.box.cx, y: e.clientY - st.box.cy };
          setView(clampView({ s: 2.5, x: -1.5 * d.x, y: -1.5 * d.y }, st.box));
        }
      } else {
        lastTap.current = { t: e.timeStamp, x: e.clientX, y: e.clientY };
        // A tap at either edge moves on, once it is clear it is not the first
        // of a double tap.
        const at = (e.clientX - (st.box.cx - st.box.w / 2)) / st.box.w;
        if (viewRef.current.s === 1 && (at < EDGE || at > 1 - EDGE)) {
          clearTimeout(tapTimer.current);
          tapTimer.current = setTimeout(() => leave(at < EDGE ? -1 : 1, st.box.w), 300);
        }
      }
      setDrag({ x: 0, y: 0 });
      return;
    }
    if (st.kind === "swipe") {
      if (Math.abs(dx) > 60 && Math.abs(dx) > Math.abs(dy)) leave(dx < 0 ? 1 : -1, st.box.w, dx);
      else if (dy > 110 && dy > Math.abs(dx)) onClose();
      else setDrag({ x: 0, y: 0 });
    }
  };
  const handlers = { onPointerDown, onPointerMove, onPointerUp, onPointerCancel: onPointerUp };
  const dragging = !!(drag.x || drag.y);
  const zoomed = view.s > 1;
  const style = {
    transform: zoomed
      ? `translate(${view.x}px, ${view.y}px) scale(${view.s})`
      : `translate(${drag.x + anim.x}px, ${drag.y}px)`,
    opacity: anim.o,
    transition: dragging || zoomed ? "none" : anim.t || "transform 0.2s ease-out",
  };
  // Moving on from a button (a reject) goes the same way as a swipe.
  const next = () => leave(1, window.innerWidth);
  return { handlers, style, zoomed, next };
}

// The viewer on a phone: the photo as large as the screen allows, stars and
// Reject under the thumb. Picks, colours, tags and developing are left to a
// bigger screen.
function PhoneCull({ photos, index, folderPath, photo, onGo, onClose, rate, basketSet, onToggleBasket,
  exif, burst, photoByPath, memberFlag, onJump, onRejectBurst, hint, proposal, onProposal, imgRef, videoRef }) {
  const [info, setInfo] = useState(false);
  const nextRef = useRef(null);
  const { handlers, style, zoomed, next } = usePhotoGestures({
    onNext: () => onGo(1), onPrev: () => onGo(-1), onClose, resetKey: photo.filepath,
    canNext: index < photos.length - 1, canPrev: index > 0,
  });
  nextRef.current = next;
  const rejected = photo.flag === -1;
  // A reject is usually the end of looking at a frame: it moves on, after a
  // moment to see it marked.
  const reject = () => {
    if (rejected) { rate("flag", 0); return; }
    rate("flag", -1);
    if (index < photos.length - 1) setTimeout(() => nextRef.current?.(), 350);
  };
  const video = isVideo(photo);
  return (
    <div data-modal="viewer" data-testid="phone-viewer"
      className="fixed inset-0 z-50 bg-black flex flex-col select-none"
      style={{ paddingTop: "env(safe-area-inset-top)", paddingBottom: "env(safe-area-inset-bottom)" }}>
      <div className="flex items-center gap-1 px-1 h-12 shrink-0 text-neutral-300">
        <button onClick={onClose} aria-label="Back to the grid" className="w-11 h-11 text-xl">‹</button>
        <span className="text-sm font-mono text-neutral-400">{index + 1} / {photos.length}</span>
        <span className="flex-1" />
        <button onClick={() => setInfo((v) => !v)} aria-label="About this photo" data-testid="phone-info"
          className={`w-11 h-11 rounded-full text-lg ${info ? "text-white" : "text-neutral-500"}`}>ⓘ</button>
        <button onClick={() => onToggleBasket(photo)} aria-label="Basket" data-testid="phone-basket"
          className={`h-9 px-3 rounded-full text-sm ${basketSet.has(photo.filepath)
            ? "bg-orange-600 text-white" : "text-neutral-500 border border-neutral-800"}`}>
          ⬡{basketSet.has(photo.filepath) ? " In basket" : ""}
        </button>
      </div>

      <div className="flex-1 min-h-0 relative overflow-hidden" style={{ touchAction: "none" }}
        {...(video ? {} : handlers)} data-testid="phone-photo">
        {video ? (
          <div className="absolute inset-0 flex items-center justify-center">
            <VideoPlayer photo={photo} folderPath={folderPath} videoRef={videoRef} onFocusBack={() => {}} />
          </div>
        ) : (
          <div className="absolute inset-0 flex items-center justify-center" style={style}>
            <img ref={imgRef} draggable={false} alt={photo.filename}
              src={thumbUrl(photo.folder || folderPath, photo.filename, "preview", editKey(photo))}
              className={`max-h-full max-w-full object-contain ${rejected ? "opacity-40" : ""}`} />
          </div>
        )}
        {rejected && (
          <span className="absolute top-2 left-2 px-2 py-0.5 rounded bg-red-900/80 text-red-100 text-xs">Rejected</span>
        )}
        {/* A tap at either edge moves on: said, faintly, by a chevron there. */}
        {!video && !zoomed && index > 0 && (
          <span className="absolute left-1 top-1/2 -translate-y-1/2 text-3xl text-white/20 pointer-events-none">‹</span>
        )}
        {!video && !zoomed && index < photos.length - 1 && (
          <span className="absolute right-1 top-1/2 -translate-y-1/2 text-3xl text-white/20 pointer-events-none">›</span>
        )}
        {video && index > 0 && (
          <button onClick={() => onGo(-1)} className="absolute left-0 top-1/2 -translate-y-1/2 w-11 h-20 text-4xl text-white/40">‹</button>
        )}
        {video && index < photos.length - 1 && (
          <button onClick={() => onGo(1)} className="absolute right-0 top-1/2 -translate-y-1/2 w-11 h-20 text-4xl text-white/40">›</button>
        )}
      </div>

      {info && (
        <div className="px-4 py-2 text-xs text-neutral-400 space-y-0.5 border-t border-neutral-900" data-testid="phone-exif">
          <div className="text-neutral-300 truncate">{photo.filename}</div>
          {(photo.capture_date || exif?.date) && <div>{takenAt(photo.capture_date || exif.date)}</div>}
          {exif && (
            <div className="font-mono">
              {[exif.camera, exif.lens, exif.focal_length, exif.aperture, exif.shutter_speed, exif.iso && `ISO ${exif.iso}`]
                .filter(Boolean).join(" · ")}
            </div>
          )}
        </div>
      )}

      {hint && (hint.soft_frame || hint.blown_highlights || hint.eyes_closed) && (
        <div className="px-4 py-1 text-[11px] flex flex-wrap gap-x-3">
          {hint.soft_frame && <span className="text-amber-300/90">⚠ softer than the rest</span>}
          {hint.blown_highlights && <span className="text-red-300/90">⚠ highlights clipped</span>}
          {hint.eyes_closed && <span className="text-sky-300/90">⚠ eyes may be closed</span>}
        </div>
      )}

      {proposal && onProposal && (
        <div className="px-3 py-1.5 bg-violet-950/70 text-xs flex items-center gap-2" data-testid="phone-proposal">
          <span className="text-violet-200 flex-1 truncate">✦ {proposalText(proposal)}{proposal.reason ? ` — ${proposal.reason}` : ""}</span>
          <button onClick={() => onProposal("accept", photo)} className="px-3 py-1.5 rounded bg-violet-700 text-white">Accept</button>
          <button onClick={() => onProposal("dismiss", photo)} className="px-3 py-1.5 rounded border border-neutral-700 text-neutral-300">No</button>
        </div>
      )}

      {burst && (
        <div className="px-2 py-1.5 flex items-center gap-2 border-t border-neutral-900" data-testid="phone-burst">
          <div className="flex gap-1 overflow-x-auto flex-1">
            {(burst.members || []).map((m) => (
              <button key={m.filepath} onClick={() => onJump(m.filepath)}
                className={`relative shrink-0 rounded overflow-hidden border ${m.filepath === photo.filepath
                  ? "border-white" : "border-transparent"} ${memberFlag(m) === -1 ? "opacity-30" : ""}`}>
                <img src={thumbUrl(m.folder || folderPath, m.filename, "grid", editKey(photoByPath.get(m.filepath)))}
                  alt="" className="h-11 w-11 object-cover" />
                {memberFlag(m) !== -1 && m.filepath === burst.keep && (
                  <span className="absolute bottom-0 inset-x-0 text-[8px] text-center bg-emerald-900/90 text-emerald-200">sharpest</span>
                )}
              </button>
            ))}
          </div>
          <button onClick={onRejectBurst} className="shrink-0 px-2.5 py-2 rounded bg-red-900/60 text-red-200 text-xs"
            title="Keep the sharpest frame and reject the rest">Keep sharpest</button>
        </div>
      )}

      <div className="flex items-center justify-between px-2 py-1 border-t border-neutral-900" data-testid="phone-marks">
        <div className="flex">
          {[1, 2, 3, 4, 5].map((n) => (
            <button key={n} onClick={() => rate("rating", photo.rating === n ? 0 : n)}
              aria-label={`${n} star${n > 1 ? "s" : ""}`} data-testid={`phone-star-${n}`}
              className={`w-12 h-14 text-[28px] leading-none ${n <= (photo.rating || 0) ? "text-yellow-400" : "text-neutral-700"}`}>★</button>
          ))}
        </div>
        <button onClick={reject} data-testid="phone-reject"
          className={`h-11 px-4 rounded-full text-sm font-medium ${rejected
            ? "bg-red-700 text-white" : "border border-red-900 text-red-300"}`}>
          ✕ {rejected ? "Rejected" : "Reject"}
        </button>
      </div>
    </div>
  );
}

function Stat({ label, value, title }) {
  return <span title={title}><span className="text-neutral-600">{label} </span><span className="text-neutral-300">{value}</span></span>;
}

// ── People ──────────────────────────────────────────────────────────
// Faces found on this machine and grouped by who they are. A name typed here
// becomes an ordinary event tag, so a person can then be filtered, tagged,
// basketed and exported by everything that already exists. Nothing in this
// panel is a new kind of thing.

function faceCropUrl(filepath, index, size = 96) {
  return `${API}/faces/crop/${filepath}?index=${index}&size=${size}`;
}

function PersonRow({ person, people, onChanged, onOpen, open }) {
  const [name, setName] = useState(person.name || "");
  const [busy, setBusy] = useState(false);
  // The merge list is built only when asked for. A library holds thousands of
  // groups, and a list of every other group in every row is millions of
  // options — which is what froze this panel once the library was scanned.
  const [picking, setPicking] = useState(false);
  const targets = useMemo(() => {
    if (!picking) return [];
    const others = people.filter((o) => o.id !== person.id);
    // Named people first, then the groups seen most: the likely answers on top.
    return [...others.filter((o) => o.name), ...others.filter((o) => !o.name)];
  }, [picking, people, person.id]);

  // The server is the record; the input follows it when the row is redrawn
  // after a rename or a merge.
  useEffect(() => { setName(person.name || ""); }, [person.id, person.name]);

  const call = async (path, body, method = "POST") => {
    setBusy(true);
    try {
      const res = await fetch(`${API}${path}`, {
        method,
        headers: body ? { "Content-Type": "application/json" } : undefined,
        body: body ? JSON.stringify(body) : undefined,
      });
      if (res.ok) onChanged();
    } catch {} finally { setBusy(false); }
  };

  const save = () => {
    if (name.trim() === (person.name || "")) return;
    call(`/faces/people/${person.id}/name`, { name });
  };

  return (
    <div className={`flex items-center gap-2 rounded border px-2 py-1.5 ${
      open ? "border-amber-500/50 bg-amber-500/5" : "border-neutral-800"
    }`}>
      <button onClick={() => onOpen(person)} title="Check this person's faces"
        className="w-10 h-10 rounded overflow-hidden flex-shrink-0 bg-neutral-900">
        {person.sample && (
          <img src={faceCropUrl(person.sample.filepath, person.sample.face_index, 80)} loading="lazy"
            alt={person.name || "unnamed person"} className="w-full h-full object-cover" />
        )}
      </button>
      <div className="flex-1 min-w-0">
        <input value={name} onChange={(e) => setName(e.target.value)}
          onBlur={save}
          onKeyDown={(e) => { if (e.key === "Enter") e.currentTarget.blur(); }}
          placeholder="Name this person"
          className="w-full bg-neutral-900 border border-neutral-800 rounded px-2 py-0.5 text-sm text-neutral-200 placeholder-neutral-600" />
        <p className="text-[10px] text-neutral-500 mt-0.5">
          {person.photos} photo{person.photos === 1 ? "" : "s"}
          {person.faces !== person.photos ? ` · ${person.faces} faces` : ""}
        </p>
      </div>
      {/* "Is this Alice?" — an unnamed group close to a named person. Yes folds
          it in; no is remembered on its faces, so it is not asked again. */}
      {!person.name && person.suggest_id > 0 && person.suggest_name && (
        <div className="flex items-center gap-1 text-[11px] text-amber-200/90 shrink-0"
          title={`Similarity ${Number(person.suggest_score).toFixed(2)}`}>
          <span className="truncate max-w-[110px]">{person.suggest_name}?</span>
          <button disabled={busy}
            onClick={() => call(`/faces/people/${person.id}/merge`, { into: person.suggest_id })}
            className="px-1.5 rounded bg-emerald-800/60 hover:bg-emerald-700 text-emerald-100 disabled:opacity-40">yes</button>
          <button disabled={busy}
            onClick={() => call(`/faces/people/${person.id}/not/${person.suggest_id}`, null)}
            className="px-1.5 rounded bg-neutral-800 hover:bg-neutral-700 text-neutral-300 disabled:opacity-40">no</button>
        </div>
      )}
      {/* Folding a group into another is the correction for a split; taking one
          apart is the correction for a merge. Both are one click. */}
      {picking ? (
        <select value="" disabled={busy} autoFocus
          onChange={(e) => { if (e.target.value) { setPicking(false); call(`/faces/people/${person.id}/merge`, { into: +e.target.value }); } }}
          onBlur={() => setPicking(false)}
          title="This is the same person as…"
          className="bg-neutral-900 border border-neutral-700 rounded text-[10px] text-neutral-300 max-w-[120px]">
          <option value="">same person as…</option>
          {targets.map((o) => (
            <option key={o.id} value={o.id}>
              {o.name || `#${o.id}`} · {o.photos} photo{o.photos === 1 ? "" : "s"}
            </option>
          ))}
        </select>
      ) : (
        <button onClick={() => setPicking(true)} disabled={busy} title="This is the same person as…"
          className="text-[10px] px-1.5 py-0.5 rounded border border-neutral-800 text-neutral-400 hover:text-neutral-200 disabled:opacity-40">
          merge
        </button>
      )}
      <button onClick={() => call(`/faces/people/${person.id}/detach`, null)}
        disabled={busy} title="Not one person after all — take the group apart"
        className="text-xs text-neutral-600 hover:text-red-400 disabled:opacity-40">✕</button>
    </div>
  );
}

function PeoplePanel({ folder }) {
  const [status, setStatus] = useState(null);
  const [people, setPeople] = useState([]);
  const [job, setJob] = useState(null);
  const [note, setNote] = useState("");
  const [openPerson, setOpenPerson] = useState(null);
  const [showSingles, setShowSingles] = useState(false);
  const [showUnclear, setShowUnclear] = useState(false);
  const [regrouping, setRegrouping] = useState(false);
  // Rows are drawn a page at a time: the list comes named-first and largest-first,
  // so the people worth naming are on the first page, and a scanned library's
  // thousand-odd groups are not a thousand rows and face crops at once.
  const PAGE = 60;
  const [limit, setLimit] = useState(PAGE);

  const load = useCallback(async () => {
    try {
      const [statusRes, peopleRes] = await Promise.all([
        fetch(`${API}/faces/status`).then((r) => r.json()),
        fetch(`${API}/faces/people`).then((r) => r.json()),
      ]);
      setStatus(statusRes);
      setJob(statusRes.job);
      setPeople(peopleRes.people || []);
    } catch { setNote("The backend is not reachable"); }
  }, []);

  useEffect(() => { load(); }, [load]);

  // While a scan runs, follow it. The scan itself is a background job on the
  // server, so this is a poll rather than a stream, and it stops as soon as the
  // job does.
  useEffect(() => {
    if (!job || job.status !== "running") return;
    const timer = setInterval(async () => {
      try {
        const next = await fetch(`${API}/faces/scan/status`).then((r) => r.json());
        setJob(next);
        if (next.status !== "running") load();
      } catch {}
    }, 1500);
    return () => clearInterval(timer);
  }, [job, load]);

  const scan = async (body) => {
    setNote("");
    try {
      const res = await fetch(`${API}/faces/scan`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (!res.ok) { setNote(data.detail || "Could not start the scan"); return; }
      setJob({ status: "running", phase: "models", scope: data.scope, done: 0, total: 0 });
    } catch { setNote("Could not start the scan"); }
  };

  const stop = async () => {
    try { await fetch(`${API}/faces/scan/stop`, { method: "POST" }); } catch {}
  };

  // Full size, for a folder where faces were too small to find: a school
  // concert, a hall. A few seconds a photo, so it asks first.
  const lookCloser = async () => {
    if (!window.confirm(`Look at every photo in ${folder} at full size? It finds faces too small `
      + "for the normal scan, at a few seconds a photo.")) return;
    setNote("");
    try {
      const res = await fetch(`${API}/faces/closer`, { method: "POST",
        headers: { "Content-Type": "application/json" }, body: JSON.stringify({ folder }) });
      const data = await res.json();
      if (!res.ok) { setNote(data.detail || "Could not start"); return; }
      setJob({ status: "running", phase: "models", scope: data.scope, closer: true, done: 0, total: 0 });
    } catch { setNote("Could not start"); }
  };

  const showPerson = (person) => {
    setOpenPerson((current) => (current && current.id === person.id ? null : person));
  };

  const running = job && job.status === "running";
  const engineMissing = status && status.engine === false;
  // A group seen in one photograph is a face, not a person: there is no second
  // view to agree with it and nothing to name it against. Across a whole library
  // they are the bulk of the list — on one real run, 427 of 699 groups — so they
  // are behind a count rather than in the way.
  // A single-photo group with a suggested name is worth showing: answering
  // "is this Alice?" is how it stops being a single.
  const asked = (p) => p.suggest_id > 0 && !p.name;
  // A group of mostly blurred faces is a guess, not a person: under 15% of its
  // faces are clear enough to start a group of their own. Some are one person
  // seen from far away, so they are kept — behind a count, below the rest.
  const unclear = (p) => !p.name && !asked(p) && p.photos >= 2 && (p.clear ?? p.faces) < 0.15 * p.faces;
  const singles = useMemo(() => people.filter((p) => p.photos < 2 && !asked(p)), [people]);
  const unclearGroups = useMemo(() => people.filter(unclear), [people]); // eslint-disable-line react-hooks/exhaustive-deps
  const listed = useMemo(
    () => people.filter((p) => (showSingles || p.photos >= 2 || asked(p)) && (showUnclear || !unclear(p))),
    [people, showSingles, showUnclear]); // eslint-disable-line react-hooks/exhaustive-deps
  const shown = listed.slice(0, limit);

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3 flex-wrap">
        <h2 className="text-xl font-medium">People</h2>
        {status && (
          <span className="text-xs text-neutral-500">
            {status.faces} faces in {status.scanned} of {status.photos} photos
            {status.named > 0 && ` · ${status.named} named`}
            {status.unscanned > 0 && ` · ${status.unscanned} not looked at yet`}
            {status.model && ` · ${status.model}`}
          </span>
        )}
      </div>

      {status && status.other_model > 0 && !running && (
        <p className="text-xs text-sky-200/90 bg-sky-950/30 border border-sky-900/50 rounded p-2">
          {status.other_model} photo{status.other_model === 1 ? " was" : "s were"} described by an
          older face model. Scan again (<b>Whole library</b>) to describe them with {status.model};
          the names you have given are kept.
        </p>
      )}

      <div className="flex items-center gap-2 flex-wrap">
        <button onClick={() => scan({ folder })} disabled={!folder || running || engineMissing}
          className="px-3 py-1.5 rounded text-sm bg-amber-600/80 hover:bg-amber-600 text-white disabled:opacity-40">
          Find faces in this folder
        </button>
        <button onClick={lookCloser} disabled={!folder || running || engineMissing}
          title="Read every photo in this folder at full size, for faces too small for the scan"
          className="px-3 py-1.5 rounded text-sm border border-neutral-700 text-neutral-300 hover:text-white disabled:opacity-40">
          Look closer
        </button>
        <button onClick={() => scan({})} disabled={running || engineMissing}
          className="px-3 py-1.5 rounded text-sm border border-neutral-700 text-neutral-300 hover:text-white disabled:opacity-40">
          Whole library
        </button>
        {people.length > 0 && !running && (
          <button disabled={regrouping} onClick={async () => {
            // Some seconds over a whole library, so it says it is working and,
            // after, what changed — the list itself can look much the same.
            setRegrouping(true);
            setNote("");
            const started = Date.now();
            try {
              const res = await fetch(`${API}/faces/people/rebuild`, { method: "POST",
                headers: { "Content-Type": "application/json" }, body: JSON.stringify({}) });
              const r = await res.json();
              if (!res.ok) throw new Error(r.detail || "failed");
              const seconds = Math.round((Date.now() - started) / 1000);
              setNote(`Regrouped ${r.faces.toLocaleString()} faces in ${seconds}s: ${r.people.toLocaleString()} groups`
                + (r.unclear ? `, ${r.unclear.toLocaleString()} too blurred to say who, left out` : "")
                + (r.suggestions ? `, ${r.suggestions} asking "is this …?"` : "") + ".");
              load();
            } catch (err) {
              setNote(`Regroup did not finish: ${err.message}`);
            } finally {
              setRegrouping(false);
            }
          }} title="Sort the stored faces into people again, without reading any photo. Named people keep their faces."
            className="px-3 py-1.5 rounded text-sm border border-neutral-800 text-neutral-400 hover:text-white disabled:opacity-50">
            {regrouping ? "Regrouping…" : "Regroup"}
          </button>
        )}
        {!folder && (
          <span className="text-xs text-neutral-600">open a folder to scan just that one</span>
        )}
      </div>

      {engineMissing && (
        <p className="text-sm text-amber-300/90 bg-amber-950/30 border border-amber-900/50 rounded p-3">
          {status.reason}
        </p>
      )}

      {running && (
        <div className="text-sm border border-neutral-800 rounded p-3 space-y-1">
          <div className="flex items-center justify-between">
            <span className="text-neutral-300">
              {job.automatic && <span className="text-neutral-500">Automatic · </span>}
              {job.phase === "models" && "Getting the face models (39 MB, once)"}
              {job.phase === "rendering" && `Reading frames in ${job.folder || job.scope}`}
              {job.phase === "scanning" && (job.closer
                ? `Looking closer at ${job.scope}`
                : `Looking for faces in ${job.folder || job.scope}`)}
              {job.phase === "grouping" && "Grouping the faces"}
            </span>
            <button onClick={stop} className="text-xs text-neutral-500 hover:text-white">Stop</button>
          </div>
          {job.total > 0 && (
            <>
              <div className="h-1 bg-neutral-800 rounded overflow-hidden">
                <div className="h-full bg-amber-500 transition-all"
                  style={{ width: `${Math.min(100, Math.round((job.done / job.total) * 100))}%` }} />
              </div>
              <p className="text-xs text-neutral-500">
                {job.done} / {job.total} in this folder
                {job.folders > 1 ? ` · folder ${job.folder_index} of ${job.folders}` : ""}
                {job.phase === "scanning" && !job.closer && ` · ${job.faces} faces so far`}
                {job.failed > 0 ? ` · ${job.failed} unreadable` : ""}
              </p>
            </>
          )}
        </div>
      )}

      {!running && job && job.status === "failed" && (
        <p className="text-sm text-red-300/90 bg-red-950/30 border border-red-900/50 rounded p-3">
          {job.error}
        </p>
      )}
      {!running && job && job.status === "completed" && (
        <p className="text-xs text-neutral-500">
          Last scan: {job.scanned} frames looked at, {job.faces} faces, {job.people} people
          {job.skipped > 0 ? ` · ${job.skipped} already done` : ""}
          {job.dropped > 0 ? ` · ${job.dropped} stale faces cleared` : ""}
        </p>
      )}
      {note && <p className="text-xs text-neutral-500">{note}</p>}

      {people.length === 0 && !running && (
        <p className="text-sm text-neutral-500">
          No faces stored yet. A scan reads the grid rendering of each frame, at a
          size where one person's face is separable from another's, and keeps the
          result on this machine.
        </p>
      )}

      {singles.length > 0 && (
        <button onClick={() => setShowSingles((v) => !v)}
          className="text-xs text-neutral-500 hover:text-neutral-300">
          {showSingles ? "Hide" : "Show"} {singles.length} group
          {singles.length === 1 ? "" : "s"} seen in a single photograph
        </button>
      )}
      {unclearGroups.length > 0 && (
        <button onClick={() => setShowUnclear((v) => !v)}
          title="Groups made mostly of blurred, tiny or turned-away faces: often not one person"
          className="ml-4 text-xs text-neutral-500 hover:text-neutral-300">
          {showUnclear ? "Hide" : "Show"} {unclearGroups.length} unclear group
          {unclearGroups.length === 1 ? "" : "s"} (mostly blurred faces)
        </button>
      )}

      <div className="grid gap-2"
        style={{ gridTemplateColumns: "repeat(auto-fill, minmax(min(560px, 100%), 1fr))" }}>
        {shown.map((p) => (
          <PersonRow key={p.id} person={p} people={people}
            onChanged={load} onOpen={showPerson} open={openPerson?.id === p.id} />
        ))}
      </div>
      {listed.length > shown.length && (
        <button onClick={() => setLimit((n) => n + PAGE)}
          className="text-xs text-neutral-500 hover:text-neutral-300">
          Show {Math.min(PAGE, listed.length - shown.length)} more of {listed.length - shown.length}
        </button>
      )}

      {openPerson && (
        <PersonReview person={openPerson} people={people} onChanged={load}
          onClose={() => setOpenPerson(null)} />
      )}
    </div>
  );
}

// ── Activity Panel ──────────────────────────────────────────────────
//
// Everything working in the background, and how far the library has got. This
// was the thumbnail queue's page when thumbnails were the only background work;
// imports, measuring, faces, places and exports have joined since, and a page
// that showed one of them said "idle" while the machine was busy.

const FACE_PHASES = { models: "Loading models", rendering: "Preparing pictures", scanning: "Looking for faces", grouping: "Grouping people" };

function ago(seconds) {
  const s = Math.max(0, Date.now() / 1000 - seconds);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  return new Date(seconds * 1000).toLocaleDateString();
}

function Meter({ value, total, tone = "bg-blue-500" }) {
  const pct = total > 0 ? Math.min(100, Math.round((value / total) * 100)) : 0;
  return (
    <div className="w-full bg-neutral-800 rounded-full h-1.5">
      <div className={`${tone} h-1.5 rounded-full transition-all`} style={{ width: `${pct}%` }} />
    </div>
  );
}

// What the library is doing on its own, in the header on every tab. Automatic
// work (faces, smart search, camera data, places, measuring, thumbnails for
// folders not yet opened) steps aside while the app is in use and picks up a
// few seconds after; Pause holds it until Resume. Work started by hand runs on.
function BackgroundBadge({ onOpen }) {
  const [state, setState] = useState(null);
  const refresh = useCallback(async () => {
    try {
      const res = await fetch(`${API}/background`);
      if (res.ok) setState(await res.json());
    } catch { /* keep the last answer */ }
  }, []);
  useEffect(() => {
    let stop = false, timer = null;
    const tick = async () => {
      await refresh();
      if (!stop) timer = setTimeout(tick, 3000);
    };
    tick();
    return () => { stop = true; clearTimeout(timer); };
  }, [refresh]);

  if (!state) return null;
  const work = state.work || [];
  if (!work.length && !state.paused) return null;
  const toggle = async () => {
    try {
      const res = await fetch(`${API}/background/${state.paused ? "resume" : "pause"}`, { method: "POST" });
      if (res.ok) setState(await res.json());
    } catch { /* the next tick shows the truth */ }
  };
  const first = work[0];
  const what = first
    ? [first.kind, first.total > 0 ? `${first.done.toLocaleString()}/${first.total.toLocaleString()}` : "",
       work.length > 1 ? `+${work.length - 1} more` : ""].filter(Boolean).join(" · ")
    : "Background work";
  const how = state.paused ? "paused" : state.yielding ? "waiting while you browse" : "running";
  const tone = state.paused ? "bg-amber-400" : state.yielding ? "bg-neutral-500" : "bg-blue-400 animate-pulse";
  const detail = work.map((w) => `${w.kind}${w.where ? ` — ${w.where}` : ""}${w.automatic ? "" : " (started by hand: not paused)"}`).join("\n");
  return (
    <div className="ml-auto flex items-center gap-2 text-xs" data-testid="background-badge">
      <button onClick={onOpen} title={`${detail || "Nothing running"}\n\nAutomatic work steps aside while you use the app and resumes a few seconds after. Click for the Activity tab.`}
        className="flex items-center gap-1.5 text-neutral-400 hover:text-neutral-200 min-w-0">
        <span className={`w-2 h-2 rounded-full shrink-0 ${tone}`} />
        <span className="truncate max-w-[16rem]">{what}</span>
        <span className="text-neutral-600 whitespace-nowrap">{how}</span>
      </button>
      <button onClick={toggle} data-testid="background-toggle"
        title={state.paused ? "Let background work run again" : "Hold all automatic background work until you resume it"}
        className={`px-2 py-0.5 rounded border ${state.paused ? "border-amber-600 text-amber-300 hover:bg-amber-950" : "border-neutral-700 text-neutral-400 hover:text-neutral-200"}`}>
        {state.paused ? "Resume" : "Pause"}
      </button>
    </div>
  );
}

function BackgroundNote({ state, onChange }) {
  const toggle = async () => {
    await fetch(`${API}/background/${state.paused ? "resume" : "pause"}`, { method: "POST" }).catch(() => {});
    onChange();
  };
  return (
    <div className="flex items-center gap-3 text-xs text-neutral-500">
      <span className="flex-1">
        {state.paused
          ? "Automatic work is paused. Imports, exports and scans you start yourself still run."
          : state.yielding
            ? `Automatic work is waiting while the app is in use${state.waiting?.length ? ` (${state.waiting.join(", ")})` : ""}. It picks up ${state.quiet_seconds} s after you stop.`
            : "Automatic work runs when the app is idle and steps aside while you browse."}
      </span>
      <button onClick={toggle} data-testid="activity-background-toggle"
        className={`px-2 py-0.5 rounded border ${state.paused ? "border-amber-600 text-amber-300" : "border-neutral-700 text-neutral-400 hover:text-neutral-200"}`}>
        {state.paused ? "Resume" : "Pause all"}
      </button>
    </div>
  );
}

function ActivityPanel({ local }) {
  const [data, setData] = useState(null);
  const [failed, setFailed] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const res = await fetch(`${API}/activity`);
      if (!res.ok) throw new Error(res.status);
      setData(await res.json());
      setFailed(false);
    } catch {
      setFailed(true);
    }
  }, []);

  // The next request is sent two seconds after the last one answered, never on
  // a fixed beat: a slow answer must not have more requests piling up behind it.
  useEffect(() => {
    let stop = false, timer = null;
    const tick = async () => {
      await refresh();
      if (!stop) timer = setTimeout(tick, 2000);
    };
    tick();
    return () => { stop = true; clearTimeout(timer); };
  }, [refresh]);

  const action = async (jobId, act) => {
    await fetch(`${API}/thumbnails/queue/${jobId}/${act}`, { method: "POST" }).catch(() => {});
    refresh();
  };

  if (!data) {
    return <p className={`text-sm ${failed ? "text-red-400" : "text-neutral-500"}`}>
      {failed ? "The library computer did not answer. Trying again…" : "Checking…"}
    </p>;
  }

  const { coverage: cov, queue: status, thumbnail_jobs: jobs = [], imports = [], faces, places, exports = [] } = data;

  // What is running now, one row each. A worker that stopped, or a cache that
  // refused writes, is said in red: an empty list would read as "all done".
  const running = [];
  const dead = status.enabled && !status.alive;
  if (dead) {
    running.push({ key: "q", label: "Thumbnails", detail: "The background worker is not running.", bad: true });
  } else if (status.storage_error) {
    running.push({ key: "q", label: "Thumbnails", detail: `Paused: ${status.storage_error}`, bad: true });
  } else if (status.activity === "measuring") {
    running.push({ key: "q", label: "Measuring", detail: `${status.activity_folder} · ${Math.round(status.activity_seconds)}s` });
  } else if (status.activity === "thumbnails") {
    running.push({ key: "q", label: "Thumbnails", detail: status.activity_folder });
  }
  imports.filter((j) => j.status === "running").forEach((j) => running.push({
    key: `i${j.job_id}`, label: "Import",
    detail: `${IMPORT_STAGES[j.stage] || j.stage} ${j.current_file || ""}`,
    done: j.completed, total: j.total,
  }));
  if (faces?.status === "running") {
    running.push({
      key: "f", label: "Faces",
      detail: [FACE_PHASES[faces.phase] || faces.phase, faces.folder || faces.scope,
        faces.folders > 1 ? `folder ${faces.folder_index} of ${faces.folders}` : ""].filter(Boolean).join(" · "),
      done: faces.done, total: faces.total,
    });
  }
  if (places?.status === "running") {
    running.push({
      key: "p", label: "Places",
      detail: [places.folder, places.folders > 1 ? `folder ${places.folder_index} of ${places.folders}` : "",
        `${places.found} with a location`].filter(Boolean).join(" · "),
      done: places.done, total: places.total,
    });
  } else if (data.places_auto) {
    running.push({ key: "pa", label: "Places", detail: "Reading where newly added photos were taken" });
  }
  const smart = data.smart_search;
  if (smart?.status === "running") {
    running.push({ key: "s", label: "Smart search", detail: "Describing photos for searching by what they show",
      done: smart.done, total: smart.total });
  }
  const exif = data.exif;
  if (exif?.status === "running") {
    running.push({
      key: "x", label: "Camera data",
      detail: [exif.folder, `${(exif.filled || 0).toLocaleString()} filled in`].filter(Boolean).join(" · "),
      done: exif.done, total: exif.total,
    });
  }
  exports.filter((e) => e.status === "running").forEach((e) => running.push({
    key: `e${e.job_id}`, label: "Export",
    detail: `${local && e.owner !== "local" ? `${e.owner_name} · ` : ""}${e.current_file}`,
    done: e.completed + e.failed, total: e.total,
  }));

  const coverage = [
    { label: "Thumbnails", value: cov.thumbnails, note: "made, for browsing" },
    { label: "Measured", value: cov.measured, note: "for sharpness, burst and exposure hints" },
    { label: "Faces", value: cov.faces_scanned, note: "looked at for people" },
    { label: "Places", value: cov.places_read, note: `read for a location · ${cov.places_found.toLocaleString()} have one` },
    { label: "Camera data", value: cov.exif_read ?? 0, note: `read for camera, lens, exposure and date · ${(cov.exif_known ?? 0).toLocaleString()} have it` },
  ];

  const activeJobs = jobs.filter((j) => j.status !== "completed");
  const finishedJobs = jobs.length - activeJobs.length;

  return (
    <div className="space-y-6 max-w-3xl">
      <section className="space-y-2">
        <h3 className="text-sm font-medium text-neutral-300">Running now</h3>
        {data.background && (
          <BackgroundNote state={data.background} onChange={refresh} />
        )}
        {running.length === 0 && <p className="text-neutral-500 text-sm">Nothing is running. The app is idle.</p>}
        {running.map((r) => (
          <div key={r.key} className="border border-neutral-800 rounded p-3 text-sm space-y-1.5">
            <div className="flex items-baseline gap-3">
              <span className={`w-24 shrink-0 font-medium ${r.bad ? "text-red-400" : "text-blue-300"}`}>{r.label}</span>
              <span className={`flex-1 min-w-0 truncate text-xs ${r.bad ? "text-red-400" : "text-neutral-400"}`}>{r.detail}</span>
              {r.total > 0 && <span className="text-xs text-neutral-500 tabular-nums">{r.done.toLocaleString()} / {r.total.toLocaleString()}</span>}
            </div>
            {r.total > 0 && <Meter value={r.done} total={r.total} />}
          </div>
        ))}
        {status.measure_cursor && !dead && (
          <p className="text-xs text-neutral-600" title="How far the idle measurement sweep has got. If this does not move over several minutes, the sweep is not running.">
            Measurement sweep at {status.measure_cursor}
          </p>
        )}
      </section>

      <section className="space-y-2">
        <h3 className="text-sm font-medium text-neutral-300">
          Library <span className="text-neutral-500 font-normal">· {cov.photos.toLocaleString()} photos</span>
        </h3>
        <div className="border border-neutral-800 rounded p-3 space-y-3">
          {coverage.map((c) => {
            const shown = Math.min(c.value, cov.photos);
            const pct = cov.photos > 0 ? Math.floor((shown / cov.photos) * 100) : 0;
            return (
              <div key={c.label} className="space-y-1">
                <div className="flex items-baseline gap-3 text-sm">
                  <span className="w-24 shrink-0 text-neutral-300">{c.label}</span>
                  <span className="flex-1 min-w-0 truncate text-xs text-neutral-500">{c.note}</span>
                  <span className="text-xs text-neutral-400 tabular-nums">{shown.toLocaleString()} · {pct}%</span>
                </div>
                <Meter value={shown} total={cov.photos} tone={pct >= 100 ? "bg-green-600" : "bg-blue-500"} />
              </div>
            );
          })}
        </div>
      </section>

      <section className="space-y-2">
        <h3 className="text-sm font-medium text-neutral-300">
          Thumbnail jobs {finishedJobs > 0 && <span className="text-neutral-500 font-normal">· {finishedJobs} finished</span>}
        </h3>
        {activeJobs.length === 0 && <p className="text-neutral-500 text-sm">No folders waiting.</p>}
        {activeJobs.map((j) => (
          <div key={j.id} className="border border-neutral-800 rounded p-3 text-sm space-y-1">
            <div className="flex items-center justify-between gap-3">
              <span className="font-mono text-neutral-300 truncate">{j.folder}</span>
              <span className={`text-xs px-2 py-0.5 rounded capitalize ${
                j.status === "running" ? "bg-blue-900/50 text-blue-300" :
                j.status === "failed" ? "bg-red-900/50 text-red-300" :
                j.status === "paused" ? "bg-yellow-900/50 text-yellow-300" :
                "bg-neutral-800 text-neutral-400"
              }`}>{j.status}</span>
            </div>
            <Meter value={j.completed_items} total={j.total_items} />
            <div className="flex items-center justify-between text-xs text-neutral-500">
              <span>{j.completed_items}/{j.total_items}</span>
              <div className="flex gap-2">
                {j.status === "running" && <button onClick={() => action(j.id, "pause")} className="hover:text-yellow-400">Pause</button>}
                {j.status === "paused" && <button onClick={() => action(j.id, "resume")} className="hover:text-blue-400">Resume</button>}
                {(j.status === "running" || j.status === "paused" || j.status === "queued") &&
                  <button onClick={() => action(j.id, "cancel")} className="hover:text-red-400">Cancel</button>}
              </div>
            </div>
          </div>
        ))}
      </section>

      <section className="space-y-2">
        <h3 className="text-sm font-medium text-neutral-300">Recent exports</h3>
        {exports.length === 0 && <p className="text-neutral-500 text-sm">No exports since the app started.</p>}
        {exports.length > 0 && (
          <div className="border border-neutral-800 rounded divide-y divide-neutral-800 text-sm">
            {exports.map((e) => (
              <div key={e.job_id} className="px-3 py-2 flex items-baseline gap-3">
                <span className="w-20 shrink-0 text-xs text-neutral-500">{ago(e.started_at)}</span>
                {local && <span className="w-32 shrink-0 truncate text-neutral-300" title={e.owner_name}>{e.owner_name}</span>}
                <span className="flex-1 min-w-0 truncate text-xs text-neutral-400" title={e.folder || undefined}>
                  {e.kind === "download"
                    ? (e.expired ? "Download (expired)" : "Download to the device")
                    : e.folder}
                </span>
                <span className={`text-xs tabular-nums ${
                  e.status === "failed" ? "text-red-400" : e.status === "running" ? "text-blue-300" : "text-neutral-400"
                }`}>
                  {e.status === "failed" ? "Stopped" : `${e.completed}/${e.total}`}
                  {e.failed > 0 && <span className="text-red-400"> · {e.failed} failed</span>}
                </span>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

// ── Export Panel ─────────────────────────────────────────────────────

// One line describing a frame's edit, for the export panel. The develop view is
// where an edit is read in detail — eight sliders, a histogram and the original
// to compare against — so this only has to answer "what is about to be applied
// to this frame" without becoming a second set of controls.
const EDIT_LABELS = {
  exposure: "Exposure", temperature: "Warmth", tint: "Tint",
  highlights: "Highlights", shadows: "Shadows", whites: "Whites",
  blacks: "Blacks", contrast: "Contrast",
};

function describeEdit(settings) {
  const entries = Object.entries(settings || {});
  // The treatment is a switch, said first and in words.
  const parts = settings?.mono ? ["Black & white"] : [];
  entries
    .filter(([key, value]) => key !== "mono" && !Array.isArray(value) && Number(value) !== 0)
    .forEach(([key, value]) => {
      const shown = key === "exposure" ? Number(value).toFixed(2) : value;
      parts.push(`${EDIT_LABELS[key] || key} ${Number(value) > 0 ? "+" : ""}${shown}`);
    });
  // A tone curve is a list of points, said in words rather than as numbers.
  if (entries.some(([, value]) => Array.isArray(value))) parts.push("Tone curve");
  return parts.length
    ? parts.join(" · ")
    : "No edit — this frame exports as it was shot.";
}

// The basket's photos for a collage: a RAW+JPEG pair once (the JPEG, as the
// grid shows it), in the order they were added.
function basketPhotos(items) {
  const seen = new Map();
  for (const item of items) {
    const name = item.filename || item.filepath.split("/").pop();
    const key = `${item.folder || item.filepath.slice(0, item.filepath.lastIndexOf("/"))}/${name.replace(/\.[^.]+$/, "")}`;
    const jpg = /\.jpe?g$/i.test(name);
    if (!seen.has(key) || (jpg && !/\.jpe?g$/i.test(seen.get(key)))) seen.set(key, item.filepath);
  }
  return [...seen.values()];
}

// The export's settings as last used on this device: the credit, the size and
// the quality are the same event after event, so they are not typed again.
const EXPORT_SETTINGS_KEY = "export.settings";
const EXPORT_DEFAULTS = { photographer: "", maxRes: 2048, quality: 85, stripExif: true };
// The sizes people send photos at. Google Photos keeps 2048 px as it is, and it
// is plenty on a phone or a laptop; full size is for prints and archives.
const EXPORT_SIZES = [
  { value: 2048, label: "2048 px", note: "Google Photos, sharing and screens" },
  { value: 0, label: "Full size", note: "As large as the photo is: prints and archives" },
];

function ExportPanel({ basketItems, basketCount, onRefreshBasket, onClearBasket, onDevelop, onCollage }) {
  const kept = { ...EXPORT_DEFAULTS, ...(recall(EXPORT_SETTINGS_KEY) || {}) };
  const [photographer, setPhotographer] = useState(kept.photographer);
  const [maxRes, setMaxRes] = useState(kept.maxRes);
  const [quality, setQuality] = useState(kept.quality);
  const [stripExif, setStripExif] = useState(kept.stripExif);
  // Where the export goes, as the server says: a folder name under the
  // library's export folder on this computer, or a download on any other
  // device. Nothing here can name a path; see export/jobs.py.
  const [dest, setDest] = useState(null);
  const [folderName, setFolderName] = useState("");
  const [exportError, setExportError] = useState("");
  const [exporting, setExporting] = useState(false);
  const [progress, setProgress] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewFile, setPreviewFile] = useState(null);
  const [openError, setOpenError] = useState("");

  useEffect(() => {
    fetch(`${API}/export/default-output`).then(r => r.json()).then(setDest).catch(() => {});
  }, []);

  useEffect(() => {
    remember(EXPORT_SETTINGS_KEY, { photographer, maxRes, quality, stripExif });
  }, [photographer, maxRes, quality, stripExif]);

  // Auto-pick first basket item as preview file
  useEffect(() => {
    if (basketItems.length > 0 && !previewFile) {
      setPreviewFile(basketItems[0].filepath);
    }
  }, [basketItems, previewFile]);

  // The edit the previewed frame will be exported with: the photograph's own,
  // read from the basket payload. This panel only shows it; how a frame looks,
  // and making a set look alike (copy and paste), is the develop view's work.
  const previewEdit = useMemo(
    () => basketItems.find((i) => i.filepath === previewFile)?.adjustment || {},
    [basketItems, previewFile],
  );

  useEffect(() => {
    if (!previewFile) return;
    let stop = false;
    setPreviewLoading(true);
    fetch(`${API}/export/preview`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filepath: previewFile, size: 512, adjustment: previewEdit, resize_only: true }),
    })
      .then(r => { if (!r.ok) throw new Error(); return r.blob(); })
      .then(blob => {
        if (stop) return;
        const url = URL.createObjectURL(blob);
        setPreviewUrl(prev => { if (prev) URL.revokeObjectURL(prev); return url; });
      })
      .catch(() => { if (!stop) setPreviewUrl(null); })
      .finally(() => { if (!stop) setPreviewLoading(false); });
    return () => { stop = true; };
  }, [previewEdit, previewFile]);

  const runExport = async () => {
    setExporting(true);
    setProgress(null);
    setExportError("");
    setOpenError("");
    try {
      const res = await fetch(`${API}/export/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          folder: folderName,
          photographer,
          max_resolution: maxRes,
          quality,
          strip_exif: stripExif,
        }),
      });
      const data = await res.json();
      if (data.job_id) pollProgress(data.job_id);
      else {
        setExportError(data.detail || "The export could not start");
        setExporting(false);
      }
    } catch (e) {
      setExportError("The export could not start");
      setExporting(false);
    }
  };

  const pollProgress = (jobId) => {
    const iv = setInterval(async () => {
      try {
        const r = await fetch(`${API}/export/progress/${jobId}`);
        const p = await r.json();
        setProgress(p);
        if (p.status !== "running") {
          clearInterval(iv);
          setExporting(false);
        }
      } catch {
        clearInterval(iv);
        setExporting(false);
      }
    }, 500);
  };

  const openFolder = async () => {
    setOpenError("");
    try {
      const r = await fetch(`${API}/export/open/${progress.job_id}`, { method: "POST" });
      if (!r.ok) setOpenError((await r.json()).detail || "The folder could not be opened");
    } catch {
      setOpenError("The folder could not be opened");
    }
  };

  const customSize = !EXPORT_SIZES.some((z) => z.value === maxRes);

  return (
    <div className="space-y-6">
      {/* Basket summary */}
      <div className="border border-neutral-800 rounded-lg p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="font-medium">Export Basket <span className="text-neutral-500 text-sm ml-1">({basketCount} photos)</span></h3>
          <div className="flex gap-2">
            {/* The basket gathers photos from any folder and day, so it is where a
                collage of one person across the years is put together. */}
            {basketItems.length >= 2 && onCollage && (
              <button onClick={() => onCollage(basketPhotos(basketItems))} data-testid="basket-collage"
                title="These photos as one picture: a story, a square or a print (up to 9 at a time)"
                className="text-xs px-2 py-0.5 rounded bg-neutral-800 text-sky-200 hover:bg-neutral-700">▦ Collage</button>
            )}
            <button onClick={onRefreshBasket} className="text-xs text-neutral-400 hover:text-white">↻ Refresh</button>
            {basketCount > 0 && <button onClick={onClearBasket} className="text-xs text-red-400 hover:text-red-300">Clear all</button>}
          </div>
        </div>
        {basketItems.length === 0 ? (
          <p className="text-neutral-500 text-sm">No photos in basket. Filter a folder by rating, colour, or event tag and use “Add N to basket”, or add photos one at a time with the ⬡ button.</p>
        ) : (
          <div className="flex gap-4">
            <div className="flex-1 flex flex-wrap gap-1 max-h-40 overflow-y-auto content-start">
              {basketItems.map((item) => (
                <span key={item.filepath}
                  onClick={() => setPreviewFile(item.filepath)}
                  className={`text-xs px-2 py-1 rounded truncate max-w-[150px] cursor-pointer ${
                    previewFile === item.filepath ? "bg-orange-600/30 text-orange-300 ring-1 ring-orange-500/50" : "bg-neutral-800 text-neutral-300 hover:bg-neutral-700"
                  }`}>{item.adjustment && Object.keys(item.adjustment).length > 0 && <span className="text-amber-400" title="This photo has its own edit, and is written as developed">±</span>}{item.filename}{item.copies > 0 && <span className="ml-1 text-sky-400" title={`${item.copies} virtual cop${item.copies === 1 ? "y" : "ies"} of this photo — each is written as its own file on export (unless switched off in the develop view)`}>+{item.copies}</span>}</span>
              ))}
            </div>
            {/* The photo picked above, as the export will write it. */}
            {previewFile && (
              <div className="w-48 flex-shrink-0" data-testid="export-preview">
                <div className="relative aspect-[3/2] bg-neutral-900 rounded overflow-hidden border border-neutral-700">
                  {previewUrl && (
                    <img src={previewUrl} alt="As it will be exported"
                      className={`w-full h-full object-contain ${previewLoading ? "opacity-50" : ""}`} />
                  )}
                  {previewLoading && (
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="w-5 h-5 border-2 border-orange-500 border-t-transparent rounded-full animate-spin" />
                    </div>
                  )}
                </div>
                <div className="flex items-center justify-between mt-1 gap-2">
                  <p className="text-xs text-neutral-500 truncate" title={describeEdit(previewEdit)}>{previewFile.split("/").pop()}</p>
                  <button onClick={() => onDevelop?.(previewFile)}
                    title="Open this photo in the develop view (D)"
                    className="text-xs text-amber-400 hover:text-amber-300 shrink-0">Develop ✎</button>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Export settings */}
      <div className="border border-neutral-800 rounded-lg p-4 space-y-4">
        <h3 className="font-medium">Export Settings</h3>
        <p className="text-xs text-neutral-500 leading-relaxed" data-testid="export-note">
          Each photo is written as you developed it; one you have not developed is the
          camera's own picture. To make a set look alike, copy an edit and paste it onto
          the others in the Library or the develop view.
        </p>

        <div className="grid grid-cols-2 gap-4">
          {dest?.download ? (
            <div className="text-sm space-y-1">
              <span className="text-neutral-400">Save to</span>
              <p className="text-xs text-neutral-400 leading-relaxed">
                This device. The pictures are made on the library computer, then you download
                them{shareSupported ? " or share them to Photos, Google Photos and other apps" : ""}.
                JPEGs only; locations are removed. Up to {dest.limit} at a time.
              </p>
            </div>
          ) : (
            <label className="text-sm space-y-1">
              <span className="text-neutral-400">Folder</span>
              <div className="flex items-center bg-neutral-800 border border-neutral-700 rounded text-sm">
                <span className="pl-3 text-neutral-500 whitespace-nowrap truncate max-w-[55%]" title={dest?.path}>
                  {dest?.path ? `${dest.path}\\` : "export\\"}
                </span>
                <input type="text" value={folderName} onChange={(e) => setFolderName(e.target.value)}
                  placeholder="the event, e.g. Band-Review-2026"
                  title="A folder name inside the export folder (today's date and time when left empty). Exports never write anywhere else, and never over an existing file."
                  className="flex-1 min-w-0 bg-transparent px-1 py-1.5 outline-none" />
              </div>
            </label>
          )}
          <label className="text-sm space-y-1">
            <span className="text-neutral-400">Credit</span>
            <input type="text" value={photographer} onChange={(e) => setPhotographer(e.target.value)}
              placeholder="e.g. © Your Name 2026"
              title="Written into each photo as its Artist and Copyright. Kept for next time."
              className="w-full bg-neutral-800 border border-neutral-700 rounded px-3 py-1.5 text-sm" />
          </label>
          <div className="text-sm space-y-1">
            <span className="text-neutral-400">Size</span>
            <div className="flex flex-wrap items-center gap-1" data-testid="export-sizes">
              {EXPORT_SIZES.map((z) => (
                <button key={z.value} onClick={() => setMaxRes(z.value)} title={z.note}
                  className={`px-2 py-1 rounded border text-xs ${maxRes === z.value
                    ? "border-orange-500 text-orange-300 bg-orange-600/20" : "border-neutral-700 text-neutral-400 hover:text-white"}`}>
                  {z.label}
                </button>
              ))}
              <input type="number" value={customSize ? maxRes : ""} min={256} max={12000} step={64}
                onChange={(e) => setMaxRes(+e.target.value || 2048)} placeholder="other"
                title="Another longest side, in pixels"
                className={`w-20 bg-neutral-800 border rounded px-2 py-1 text-xs ${customSize ? "border-orange-500" : "border-neutral-700"}`} />
            </div>
          </div>
          <label className="text-sm space-y-1">
            <span className="text-neutral-400">JPEG quality</span>
            <input type="number" value={quality} min={1} max={100} onChange={(e) => setQuality(+e.target.value)}
              className="w-full bg-neutral-800 border border-neutral-700 rounded px-3 py-1.5 text-sm" />
          </label>
        </div>

        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" checked={stripExif || !!dest?.download} onChange={(e) => setStripExif(e.target.checked)}
            disabled={!!dest?.download}
            title={dest?.download ? "Pictures that leave the library computer always have their EXIF removed: no location, no camera serial number" : undefined}
            className="rounded border-neutral-600 disabled:opacity-60" />
          <span className="text-neutral-400">Keep only the date taken and the credit (no location, no camera details)</span>
        </label>
      </div>

      {/* Export button + progress */}
      <div className="flex items-center gap-4">
        <button onClick={runExport}
          disabled={exporting || basketCount === 0 || !dest}
          className="px-6 py-2.5 bg-orange-600 hover:bg-orange-500 disabled:bg-neutral-700 disabled:text-neutral-500 rounded text-sm font-medium transition-colors">
          {exporting ? "Exporting…" : `Export ${basketCount} photos`}
        </button>
        {progress && (
          <div className="flex-1 text-sm">
            <div className="w-full bg-neutral-800 rounded-full h-2 mb-1">
              <div className="bg-orange-500 h-2 rounded-full transition-all"
                style={{ width: `${progress.total > 0 ? Math.round(((progress.completed + progress.failed) / progress.total) * 100) : 0}%` }} />
            </div>
            <div className="flex justify-between text-xs text-neutral-500">
              <span>{progress.completed + progress.failed}/{progress.total} · {progress.current_file}</span>
              {progress.failed > 0 && <span className="text-red-400">{progress.failed} failed</span>}
              {progress.status === "completed" && <span className="text-green-400">Done!</span>}
              {progress.status === "failed" && <span className="text-red-400">Stopped: {progress.error}</span>}
            </div>
          </div>
        )}
      </div>
      {exportError && <p className="text-sm text-red-400">{exportError}</p>}
      {progress?.status === "completed" && progress.kind === "folder" && (
        <div className="text-sm text-neutral-400 space-y-1" data-testid="export-done">
          <p>
            Saved in <span className="font-mono text-neutral-300">{progress.folder}</span>
            <button onClick={openFolder} data-testid="export-open-folder"
              title="Show the photos in Explorer, to drag them into Google Photos or an album"
              className="ml-3 text-xs px-2 py-0.5 rounded border border-neutral-700 text-neutral-200 hover:text-white">Open folder</button>
          </p>
          <p className="text-xs text-neutral-500">
            To share them, drag the photos from the folder into an album at{" "}
            <a href="https://photos.google.com/albums" target="_blank" rel="noreferrer"
              className="underline hover:text-neutral-300">photos.google.com</a>. They keep the date they were
            taken, so the album is in the order they were shot.
          </p>
          {openError && <p className="text-xs text-red-400">{openError}</p>}
        </div>
      )}
      {progress?.status === "completed" && progress.kind === "download" && progress.files > 0 && (
        <ExportDownload job={progress} />
      )}
    </div>
  );
}

// Whether this browser can hand files to the system share sheet (Photos,
// Google Photos, AirDrop…). Only on HTTPS, and only some browsers: phones
// mostly yes, desktops mostly no.
const shareSupported = (() => {
  try {
    return typeof navigator !== "undefined" && !!navigator.canShare && window.isSecureContext &&
      navigator.canShare({ files: [new File([""], "x.jpg", { type: "image/jpeg" })] });
  } catch { return false; }
})();

// The pictures a paired device exported, waiting on the library computer for
// an hour. A ZIP for a computer; on a phone, the share sheet is the way into
// its photo library (and from there iCloud or Google Photos), so the files are
// fetched first — the sheet has to open straight from the tap.
const SHARE_MAX = 50;

function ExportDownload({ job }) {
  const base = `${API}/export/download/${job.job_id}`;
  const [names, setNames] = useState([]);
  const [files, setFiles] = useState(null);
  const [shareMsg, setShareMsg] = useState("");

  useEffect(() => {
    let stop = false;
    (async () => {
      try {
        const list = (await (await fetch(base)).json()).files || [];
        if (stop) return;
        setNames(list);
        if (!shareSupported || list.length > SHARE_MAX) return;
        const got = [];
        for (const name of list) {
          const blob = await (await fetch(`${base}/file/${encodeURIComponent(name)}`)).blob();
          got.push(new File([blob], name, { type: "image/jpeg" }));
          if (stop) return;
        }
        setFiles(got);
      } catch {}
    })();
    return () => { stop = true; };
  }, [base]);

  const share = async () => {
    setShareMsg("");
    try {
      await navigator.share({ files });
    } catch (e) {
      if (e?.name !== "AbortError") setShareMsg("This browser would not share them; download instead.");
    }
  };

  const one = names.length === 1;
  return (
    <div className="border border-neutral-800 rounded-lg p-4 space-y-2">
      <div className="flex flex-wrap items-center gap-3">
        <a href={one ? `${base}/file/${encodeURIComponent(names[0])}` : `${base}/photos.zip`}
          download
          className="px-4 py-2 bg-orange-600 hover:bg-orange-500 rounded text-sm font-medium">
          {one ? "Download photo" : `Download ${job.files} photos (ZIP)`}
        </a>
        {shareSupported && names.length > 0 && names.length <= SHARE_MAX && (
          <button onClick={share} disabled={!files}
            className="px-4 py-2 border border-neutral-600 hover:border-neutral-400 disabled:opacity-50 rounded text-sm">
            {files ? "Share… (Photos, Google Photos)" : "Preparing to share…"}
          </button>
        )}
      </div>
      <p className="text-xs text-neutral-500">
        {shareSupported
          ? (names.length > SHARE_MAX
            ? `Share takes up to ${SHARE_MAX} photos at a time; download the ZIP instead, or export fewer.`
            : "Share, then Save to Photos to put them in iCloud Photos, or pick Google Photos.")
          : "To put them in Google Photos or iCloud, upload the downloaded photos at photos.google.com or icloud.com/photos."}
        {" "}Available for an hour.
      </p>
      {shareMsg && <p className="text-xs text-red-400">{shareMsg}</p>}
    </div>
  );
}

// ── Main App ────────────────────────────────────────────────────────

const IMPORT_STAGES = {
  starting: "Starting…",
  indexing_library: "Indexing library…",
  checking: "Checking which photos are already backed up…",
  reading_exif: "Reading capture dates…",
  copying: "Copying to library…",
  done: "Finishing up…",
  indexing: "Adding the new photos to the library…",
};

// ── Other devices ───────────────────────────────────────────────────
//
// The app holds a family's photographs, so another device on the network sees
// nothing until it is paired: it asks, both screens show the same six digits,
// and someone at the computer that holds the library approves it (backend
// access.py). These are the three pieces of that: the screen a new device sees,
// the prompt on this computer, and the list of devices to revoke.

function PairScreen() {
  const [name, setName] = useState(() => {
    const ua = navigator.userAgent || "";
    return /iPhone/.test(ua) ? "iPhone" : /iPad/.test(ua) ? "iPad" : /Android/.test(ua) ? "Android"
      : /Mac/.test(ua) ? "Mac" : /Windows/.test(ua) ? "Windows PC" : "";
  });
  const [ask, setAsk] = useState(null);
  const [state, setState] = useState("");
  const [error, setError] = useState("");

  const request = async () => {
    setError(""); setState("");
    try {
      const res = await fetch(`${API}/auth/pair/request`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      const data = await res.json();
      if (!res.ok) { setError(data.detail || "Could not ask"); return; }
      setAsk(data); setState("waiting");
    } catch { setError("The computer that holds the library is not answering"); }
  };

  useEffect(() => {
    if (state !== "waiting" || !ask) return undefined;
    const timer = setInterval(async () => {
      try {
        const res = await fetch(`${API}/auth/pair/claim`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ id: ask.id, claim: ask.claim }),
        });
        const data = await res.json();
        if (data.status === "approved") window.location.reload();
        else if (data.status !== "waiting") { setState(data.status); setAsk(null); }
      } catch { /* try again */ }
    }, 2000);
    return () => clearInterval(timer);
  }, [state, ask]);

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100 flex items-center justify-center p-4">
      <div className="w-full max-w-sm space-y-4 border border-neutral-800 rounded-lg p-6">
        <h1 className="text-lg font-semibold">This device is not paired</h1>
        {state === "waiting" && ask ? (
          <>
            <p className="text-sm text-neutral-400">
              On the computer that holds the library, open Photool and approve this device.
              Check that it shows the same code:
            </p>
            <p className="text-4xl font-mono tracking-[0.3em] text-center text-amber-300">
              {ask.code.slice(0, 3)} {ask.code.slice(3)}
            </p>
            <p className="text-xs text-neutral-500">Waiting… the request lasts five minutes.</p>
          </>
        ) : (
          <>
            <p className="text-sm text-neutral-400">
              Photool only shows the album to devices approved on the computer that holds it.
            </p>
            {state === "denied" && <p className="text-sm text-red-300">The request was declined.</p>}
            {state === "expired" && <p className="text-sm text-amber-300">The request expired. Ask again.</p>}
            <label className="block text-xs text-neutral-500">Name this device
              <input value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Living-room iPad"
                className="mt-1 w-full bg-neutral-900 border border-neutral-700 rounded px-2 py-1.5 text-sm text-neutral-100" />
            </label>
            <button onClick={request} disabled={!name.trim()}
              className="w-full px-3 py-2 rounded bg-amber-600/80 hover:bg-amber-600 text-white text-sm disabled:opacity-40">
              Ask to join
            </button>
          </>
        )}
        {error && <p className="text-sm text-red-300">{error}</p>}
      </div>
    </div>
  );
}

// On this computer: a device asking to join, with its code to compare.
function PairingRequests() {
  const [requests, setRequests] = useState([]);
  const load = useCallback(async () => {
    try {
      const data = await fetch(`${API}/auth/pending`).then((r) => r.json());
      setRequests(data.requests || []);
    } catch { /* quiet */ }
  }, []);
  useEffect(() => {
    load();
    const timer = setInterval(load, 3000);
    return () => clearInterval(timer);
  }, [load]);
  const answer = async (id, verb) => {
    try { await fetch(`${API}/auth/pair/${id}/${verb}`, { method: "POST" }); } catch {}
    load();
  };
  if (!requests.length) return null;
  return (
    <div className="bg-amber-950/50 border-b border-amber-900 px-6 py-2 space-y-1">
      {requests.map((r) => (
        <div key={r.id} className="flex items-center gap-3 text-sm flex-wrap">
          <span className="text-amber-200">
            <b>{r.name}</b> ({r.address}) wants to see the album. Its screen should show
          </span>
          <span className="font-mono text-lg tracking-widest text-amber-300">{r.code.slice(0, 3)} {r.code.slice(3)}</span>
          <button onClick={() => answer(r.id, "approve")}
            className="px-2 py-0.5 rounded bg-emerald-700 hover:bg-emerald-600 text-white">Approve</button>
          <button onClick={() => answer(r.id, "deny")}
            className="px-2 py-0.5 rounded bg-neutral-800 hover:bg-neutral-700 text-neutral-200">Decline</button>
        </div>
      ))}
    </div>
  );
}

// On this computer: every paired device, and the button that shuts it out.
function DevicesPanel({ lanAccess }) {
  const [devices, setDevices] = useState([]);
  const load = useCallback(async () => {
    try {
      const data = await fetch(`${API}/auth/devices`).then((r) => r.json());
      setDevices(data.devices || []);
    } catch { setDevices([]); }
  }, []);
  useEffect(() => { load(); }, [load]);
  const revoke = async (id) => {
    try { await fetch(`${API}/auth/devices/${id}`, { method: "DELETE" }); } catch {}
    load();
  };
  const when = (t) => (t ? new Date(t * 1000).toLocaleString() : "—");
  return (
    <div className="space-y-4">
      <h2 className="text-xl font-medium">Devices</h2>
      <p className="text-sm text-neutral-400 max-w-2xl">
        {lanAccess
          ? "Other devices on the network can ask to see the album. Each one shows a six-digit code; approve it here only when the same code is on its screen."
          : "Network access is off (lan_access in config.yaml), so only this computer — and devices reaching it through Tailscale — can ask to join."}
      </p>
      {devices.length === 0 ? (
        <p className="text-sm text-neutral-500">No device has been paired.</p>
      ) : (
        <div className="border border-neutral-800 rounded divide-y divide-neutral-800">
          {devices.map((d) => (
            <div key={d.id} className="flex items-center gap-3 px-3 py-2 text-sm">
              <div className="flex-1 min-w-0">
                <p className="text-neutral-200">{d.name}</p>
                <p className="text-xs text-neutral-500 truncate" title={d.user_agent}>
                  paired {when(d.created_at)} · last seen {when(d.last_seen)} from {d.last_address || "?"}
                </p>
              </div>
              <button onClick={() => revoke(d.id)}
                className="px-2 py-1 rounded border border-neutral-700 text-neutral-300 hover:text-red-300 hover:border-red-800">
                Revoke
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// Who is looking decides what is drawn: the album, or the pairing screen.
function App() {
  const [access, setAccess] = useState(null);
  useEffect(() => {
    let stop = false;
    const check = async () => {
      try {
        const res = await fetch(`${API}/auth/status`);
        const data = res.ok ? await res.json() : { local: false, paired: false };
        // Shut out: what this device remembered of the library goes too.
        if (!data.paired) forgetAll();
        if (!stop) setAccess(data);
      } catch {
        // An older backend without pairing: carry on as before.
        if (!stop) setAccess({ local: true, paired: true });
      }
    };
    check();
    window.addEventListener("focus", check);
    return () => { stop = true; window.removeEventListener("focus", check); };
  }, []);
  if (access === null) return <div className="min-h-screen bg-neutral-950" />;
  if (!access.paired) return <PairScreen />;
  return <PhotoolApp access={access} />;
}

function PhotoolApp({ access }) {
  // Another device has no SD card to import from: it opens on the album.
  const [tab, setTab] = useState(() => {
    const asked = parseRoute(window.location.hash);
    if (asked && (asked.tab !== "devices" || access.local)) return asked.tab;
    return access.local ? "import" : "library";
  });
  const [thumbSize, setThumbSize] = useStoredState("photool.thumbSize", "M");
  const [sdCards, setSdCards] = useState([]);
  // As last seen here, until the server answers: see remembered.js.
  const [folders, setFolders] = useState(() => recall("folders") || []);
  const [importing, setImporting] = useState(false);
  const [importProgress, setImportProgress] = useState(null);
  const [lastResult, setLastResult] = useState(null);
  const [selectedFolder, setSelectedFolder] = useState(null);
  // The Library's folder cards, by folder: what the header calls the open folder.
  const [folderCards, setFolderCards] = useState(
    () => Object.fromEntries((recall("cards")?.cards || []).map((c) => [c.folder, c])));
  // The folder just left, so the overview can open where its card is.
  const [returnTo, setReturnTo] = useState(null);
  const [photos, setPhotos] = useState([]);
  const [loadingPhotos, setLoadingPhotos] = useState(false);
  const [cullIndex, setCullIndex] = useState(null);
  // Photos picked by hand in the grid, by filepath, and the last one clicked
  // (the start of a Shift-click run).
  const [picked, setPicked] = useState(() => new Set());
  // A collage being made from the picked photos (their filepaths, in grid order).
  const [collageOf, setCollageOf] = useState(null);
  const [picking, setPicking] = useState(false);
  const [pickBusy, setPickBusy] = useState(false);
  const [pickMessage, setPickMessage] = useState("");
  // What the last paste onto a selection replaced ({ filepath: edit }), for Undo.
  const [pasteUndo, setPasteUndo] = useState(null);
  const clipboard = useCopiedEdit();
  const pickAnchorRef = useRef(null);
  // Bumped each time the develop view closes over the cull view, so the cull
  // view moves to the frame develop ended on even when that is the index the
  // parent already holds.
  const [cullSync, setCullSync] = useState(0);
  // The frame the develop view is open on, or null. Held as a path rather than
  // an index so moving between frames inside the view cannot be invalidated by
  // the grid re-sorting underneath it.
  const [developFile, setDevelopFile] = useState(null);
  // A video is never developed: every way into the develop view comes through here.
  const openDevelop = useCallback((filepath) => {
    if (!isVideo(filepath)) setDevelopFile(filepath);
  }, []);
  // A slideshow over the screen: { photos, start, title }, or null.
  const [slideshow, setSlideshow] = useState(null);
  const playSlides = useCallback((list, start = 0, title = "") => {
    if (list && list.length) setSlideshow({ photos: list, start, title });
  }, []);
  const [folderStats, setFolderStats] = useState(null);
  const [filter, setFilter] = useState({ flag: null, minRating: 0, color: null, tag: null, unreviewed: false, edited: false, pairMode: "paired",
    camera: null, matches: null, hidden: [], showDemoted: false });
  const phone = usePhone();
  const [moreOpen, setMoreOpen] = useState(false);
  const [filtersOpen, setFiltersOpen] = useState(false);
  // How many filters narrow the grid, said on the phone's folded Filters button.
  const activeFilters = [filter.flag != null, filter.minRating > 0, filter.color != null, !!filter.tag,
    filter.unreviewed, filter.edited, filter.camera != null, !!filter.matches, filter.showDemoted].filter(Boolean).length;
  // The open folder's cameras, for the chips; and whether a search typed in a
  // folder is of that folder (the default) or of the whole library.
  const [cameras, setCameras] = useState([]);
  // While demoted cameras are left out: how many pairs that is, and the
  // review counts of the rest (shownCounts).
  const [shown, setShown] = useState(null);
  const filterRef = useRef(filter);
  filterRef.current = filter;
  const [searchInFolder, setSearchInFolder] = useState(true);
  const [folderSearch, setFolderSearch] = useState(null);
  // The shoots this device finds less important: [{folder, camera}], camera
  // "*" for a whole folder.
  const [demoted, setDemotedState] = useState(() => recall("demoted") || []);
  const demotedRef = useRef(demoted);
  demotedRef.current = demoted;
  const setDemoted = useCallback((shoots) => { setDemotedState(shoots); remember("demoted", shoots); }, []);
  const loadDemoted = useCallback(async () => {
    try {
      const res = await fetch(`${API}/library/demoted`);
      if (res.ok) setDemoted((await res.json()).shoots || []);
    } catch { /* nothing demoted, as far as this page knows */ }
  }, [setDemoted]);
  const setShootDemoted = useCallback(async (folder, camera, on) => {
    try {
      const res = await fetch(`${API}/library/demoted`, {
        method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ folder, camera, demoted: on }),
      });
      if (res.ok) setDemoted((await res.json()).shoots || []);
    } catch { /* the button stays as it was */ }
  }, [setDemoted]);
  // Where the user last stopped in the open folder, and the refs behind the
  // throttled write that keeps it up to date.
  const [resumeAt, setResumeAt] = useState(null);
  const savedPositionRef = useRef({ folder: null, path: null });
  const positionTimerRef = useRef(null);
  const indexedFoldersRef = useRef(new Set());
  const pendingEditsRef = useRef(new Map());
  const [tags, setTags] = useState([]);
  const [bulkBusy, setBulkBusy] = useState(false);
  const [bulkMessage, setBulkMessage] = useState("");
  // Burst groups for the open folder, straight from /catalog/bursts. Each group
  // is {size, keep, members:[{filepath, filename, flag}]}.
  const [bursts, setBursts] = useState([]);
  const [burstCoverage, setBurstCoverage] = useState(null);
  // Quality hints for the open folder, from /analysis/hints: {total, measured,
  // soft, blown, median_sharpness, hints:{filepath:{soft_frame,blown_highlights}}}.
  // Warnings only — nothing here decides anything.
  const [folderHints, setFolderHints] = useState(null);
  // An agent's waiting proposals for the open folder, {filepath: proposal}
  // (backend catalog/proposals.py), and the frames the last "Accept" took, so
  // it can be undone.
  const [proposals, setProposals] = useState({});
  const [proposalMessage, setProposalMessage] = useState("");
  const [proposalBusy, setProposalBusy] = useState(false);
  const [lastAccepted, setLastAccepted] = useState([]);
  // Every frame of the open folder, before any filter. A filter change is applied
  // to this in memory, so the grid answers on the keystroke and a filter never
  // costs a request.
  const basePhotosRef = useRef([]);
  // Bursts and hints are the only part of a filter change that needs the server.
  // Each refresh takes a sequence number: the newest one is the only one allowed
  // to draw, and the request it supersedes is cancelled instead of finished.
  const derivedSeqRef = useRef(0);
  const derivedAbortRef = useRef(null);
  const derivedTimerRef = useRef(null);
  const [measuring, setMeasuring] = useState(false);
  const [basketSet, setBasketSet] = useState(new Set());
  const [basketItems, setBasketItems] = useState([]);
  const [basketCount, setBasketCount] = useState(0);
  const [storageWarning, setStorageWarning] = useState(null);
  // exiftool is missing: the info panel and the camera's look quietly lose most of
  // what they show, so the page says why, until it is closed for this session.
  const [noExiftool, setNoExiftool] = useState(false);
  const [exiftoolNoted, setExiftoolNoted] = useState(false);
  // A library search: the text in the box, the answer to the last query that
  // ran ({query, understood, total, shown}), and why the last one could not run.
  // While there is an answer the grid shows its photos instead of a folder's.
  const [searchText, setSearchText] = useState("");
  const [searchResult, setSearchResult] = useState(null);
  const [searchError, setSearchError] = useState("");
  const searchSeqRef = useRef(0);
  // Describe: plain words say what the pictures show (smart search). Kept per
  // browser, as a preference.
  const [describeSearch, setDescribeSearch] = useState(() => {
    try { return localStorage.getItem("photool.describe") === "1"; } catch { return false; }
  });
  const [smartSearch, setSmartSearch] = useState(null);
  const toggleDescribe = useCallback(() => {
    setDescribeSearch((on) => {
      try { localStorage.setItem("photool.describe", on ? "0" : "1"); } catch { /* a preference only */ }
      return !on;
    });
  }, []);
  // How far smart search is, while it matters: asked again every few seconds
  // while the models are fetched or the library is being described.
  useEffect(() => {
    if (!describeSearch) return;
    let cancelled = false;
    let timer = null;
    const ask = async () => {
      try {
        const res = await fetch(`${API}/library/smart-search`);
        if (!res.ok) return;
        const data = await res.json();
        if (cancelled) return;
        setSmartSearch(data);
        const busy = data.setup?.status === "downloading" || (data.available && data.described < data.photos);
        if (busy) timer = setTimeout(ask, 4000);
      } catch { /* asked again when Describe is next turned on */ }
    };
    ask();
    return () => { cancelled = true; clearTimeout(timer); };
  }, [describeSearch, smartSearch?.setup?.status]); // eslint-disable-line react-hooks/exhaustive-deps
  const setupSmartSearch = useCallback(async () => {
    try {
      const res = await fetch(`${API}/library/smart-search/setup`, { method: "POST" });
      const data = await res.json();
      setSmartSearch((s) => ({ ...(s || {}), setup: data.setup }));
    } catch { /* the status line says nothing changed */ }
  }, []);

  const scanCards = useCallback(async () => {
    try {
      const res = await fetch(`${API}/sd-cards`);
      setSdCards(await res.json());
    } catch { console.error("Backend not reachable"); }
  }, []);

  // The catalog and thumbnail cache usually live on a secondary volume that can
  // go read-only without the app noticing: reads keep working, writes vanish.
  const checkStorage = useCallback(async () => {
    try {
      const res = await fetch(`${API}/health`);
      const data = await res.json();
      setStorageWarning(data.status === "degraded" ? (data.storage || null) : null);
      setNoExiftool(data.tools?.exiftool === false);
    } catch { console.error("Backend not reachable"); }
  }, []);

  const loadFolders = useCallback(async () => {
    try {
      const fsRes = await fetch(`${API}/library/folders`);
      const fsData = await fsRes.json();
      const allFolders = (fsData.folders || []).map((f) => ({ ...f, picks: 0, rejects: 0, decided: 0 }));
      try {
        const catRes = await fetch(`${API}/catalog/folders`);
        const catData = await catRes.json();
        if (catData.folders?.length > 0) {
          const catMap = {};
          for (const cf of catData.folders) catMap[cf.folder] = cf;
          for (const f of allFolders) {
            if (catMap[f.path]) {
              f.picks = catMap[f.path].picks;
              f.rejects = catMap[f.path].rejects;
              f.unique_images = catMap[f.path].unique_images;
              f.decided = catMap[f.path].decided || 0;
            }
          }
        }
      } catch {}
      setFolders(allFolders);
      remember("folders", allFolders);
    } catch { console.error("Failed to load folders"); }
  }, []);

  const loadTags = useCallback(async () => {
    try {
      const res = await fetch(`${API}/tags`);
      const data = await res.json();
      setTags(data.tags || []);
    } catch {}
  }, []);

  // Burst grouping reads the measurements the thumbnail queue already collects,
  // so this is one folder query plus in-memory clustering. It takes the grid's
  // filters so a burst is always a set of frames the user can actually see.
  const loadBursts = useCallback(async (folderPath, f, seq, signal) => {
    try {
      const res = await fetch(`${API}/catalog/bursts/${folderPath}?${photoQuery(f)}`, { signal });
      if (!res.ok) {
        if (seq === derivedSeqRef.current) { setBursts([]); setBurstCoverage(null); }
        return;
      }
      const data = await res.json();
      if (seq !== derivedSeqRef.current) return;
      setBursts(data.groups || []);
      setBurstCoverage({
        measured: data.measured || 0,
        unmeasured: data.unmeasured || 0,
        total: data.total || 0,
      });
    } catch (err) {
      // Cancelled on purpose by a newer request: nothing to report.
      if (err && err.name === "AbortError") return;
      setBursts([]);
      setBurstCoverage(null);
    }
  }, []);

  // Quality hints come from the measurements the thumbnail queue already
  // collects, and are worked out again on every read rather than stored: what
  // counts as soft is a reading of the numbers, and a reading should be free to
  // change. They are warnings, never decisions — a badge points the eye at a
  // frame, and the judgement stays the user's.
  const loadHints = useCallback(async (folderPath, f, seq, signal) => {
    try {
      const res = await fetch(`${API}/analysis/hints/${folderPath}?${photoQuery(f)}`, { signal });
      if (!res.ok) {
        if (seq === derivedSeqRef.current) setFolderHints(null);
        return;
      }
      const data = await res.json();
      if (seq !== derivedSeqRef.current) return;
      setFolderHints(data);
    } catch (err) {
      if (err && err.name === "AbortError") return;
      setFolderHints(null);
    }
  }, []);

  // Proposals are the folder's, not the filter's: a suggestion for a frame the
  // filter hides is still waiting.
  const loadProposals = useCallback(async (folderPath, seq, signal) => {
    try {
      const res = await fetch(`${API}/commands/cull.proposals`, {
        method: "POST", headers: { "Content-Type": "application/json" }, signal,
        body: JSON.stringify({ folder: folderPath, status: "pending" }),
      });
      const data = res.ok ? await res.json() : null;
      if (seq !== undefined && seq !== derivedSeqRef.current) return;
      setProposals(Object.fromEntries((data?.proposals || []).map((p) => [p.filepath, p])));
    } catch (err) {
      if (err && err.name === "AbortError") return;
      setProposals({});
    }
  }, []);

  // The grid is filtered in memory. Every rule lives in _clientFilter, which is
  // the same code that draws the folder before it has been indexed, so a filter
  // change and a folder open cannot disagree about what the selection is.
  const applyFilter = useCallback((f) => {
    const all = _applyPendingEdits(basePhotosRef.current, pendingEditsRef.current);
    setPhotos(_clientFilter(all, f));
    setCameras(folderCameras(basePhotosRef.current));
    setShown(f.hidden?.length ? shownCounts(all, f.hidden) : null);
  }, []);

  // Bursts and quality hints are the only part of a filter change that needs the
  // server, because they are worked out from every measured frame in the folder.
  // Clicking through several filters in a row should cost one folder query, not
  // one per click: the timer holds the request back, and the sequence number
  // drops any answer that a newer request has already replaced.
  const refreshDerived = useCallback((folderPath, f) => {
    const seq = ++derivedSeqRef.current;
    if (derivedAbortRef.current) derivedAbortRef.current.abort();
    const controller = new AbortController();
    derivedAbortRef.current = controller;
    return Promise.all([
      loadBursts(folderPath, f, seq, controller.signal),
      loadHints(folderPath, f, seq, controller.signal),
      loadProposals(folderPath, seq, controller.signal),
    ]);
  }, [loadBursts, loadHints, loadProposals]);

  // Grouping and quality warnings only exist for frames that have been measured.
  // The queue works through the library in the gaps between thumbnail jobs, which
  // is the right default but no help when the folder in front of you is next.
  const measureFolder = useCallback(async (folderPath, f) => {
    setMeasuring(true);
    try {
      await fetch(`${API}/analysis/run/${folderPath}`, { method: "POST" });
      await refreshDerived(folderPath, f);
    } catch { /* the folder reports itself as unmeasured again */ }
    finally { setMeasuring(false); }
  }, [refreshDerived]);

  // Re-read the current folder from the catalog. The query is deliberately
  // unfiltered: the folder is held in memory once and the filter is applied to
  // that, so changing a filter never goes back to the server. allowEmpty is only
  // true after indexing has finished — before that an empty catalog would blank a
  // folder that is still being populated.
  //
  // The filter applied is the one on screen when the answer arrives, not the
  // one asked with: indexing a folder takes seconds, and a camera or a filter
  // chosen meanwhile must not be undone by the redraw.
  const fetchPhotos = useCallback(async (folderPath, _f, allowEmpty) => {
    const res = await fetch(`${API}/catalog/photos/${folderPath}`);
    const data = await res.json();
    if (data.photos && (allowEmpty || data.photos.length > 0)) {
      basePhotosRef.current = _applyPendingEdits(data.photos, pendingEditsRef.current);
      applyFilter(filterRef.current);
    }
  }, [applyFilter]);

  // The folders' cards, read again: after indexing puts a new folder in the
  // catalog, and while waiting for that (see the effect further down).
  const refreshCards = useCallback(async () => {
    try {
      const data = await fetch(`${API}/library/cards`).then((r) => r.json());
      const fresh = Object.fromEntries((data.cards || []).map((c) => [c.folder, c]));
      // A note saved while the folder was still indexing stays shown until its
      // card arrives with it.
      setFolderCards((prev) => {
        const next = { ...fresh };
        for (const [folder, card] of Object.entries(prev)) {
          if (!next[folder] && card?.pending) next[folder] = card;
        }
        return next;
      });
      return fresh;
    } catch { return null; }
  }, []);

  const openFolder = useCallback(
    async (folderPath) => {
      const f = { ...filter, camera: null, matches: null, showDemoted: false,
        hidden: hiddenCameras(demotedRef.current, folderPath, null, false) };
      setFilter(f);
      setFolderSearch(null);
      setSearchResult(null);
      setSelectedFolder(folderPath);
      setLoadingPhotos(true);
      setCullIndex(null);
      // The previous folder's bursts would otherwise badge this folder's frames.
      setBursts([]);
      setBurstCoverage(null);
      setFolderHints(null);
      setResumeAt(null);
      // Anything still in flight belongs to the folder just left. Retire it
      // before this one starts asking, so an answer cannot arrive late and badge
      // the wrong frames.
      derivedSeqRef.current += 1;
      if (derivedAbortRef.current) derivedAbortRef.current.abort();
      basePhotosRef.current = [];
      savedPositionRef.current = { folder: null, path: null };
      try {
        // Step 1: Show files immediately from filesystem (fast, no EXIF needed).
        // Saved ratings come from the catalog in the same round trip — a local DB
        // read — so a folder never flashes back to zero stars while it indexes.
        const [filesData, catalogData, stateData] = await Promise.all([
          fetch(`${API}/library/files/${folderPath}`).then((r) => r.json()),
          fetch(`${API}/catalog/photos/${folderPath}`).then((r) => r.json()).catch(() => ({})),
          fetch(`${API}/catalog/folder-state/${folderPath}`).then((r) => r.json()).catch(() => ({})),
        ]);
        // Where the last visit stopped. Offered as a button rather than jumped
        // to: opening a folder should not drop the user somewhere unexpected.
        if (stateData.last_filepath) setResumeAt(stateData.last_filepath);
        const saved = {};
        for (const p of catalogData.photos || []) saved[p.filepath] = p;
        const rawFiles = (filesData.files || []).map((f) => {
          const filepath = `${folderPath}/${f.name}`;
          const prev = saved[filepath] || {};
          return {
            filepath,
            filename: f.name,
            folder: folderPath,
            extension: f.extension,
            base_name: f.name.replace(/\.[^.]+$/, ""),
            file_size: Math.round(f.size_mb * 1024 * 1024),
            rating: prev.rating || 0,
            flag: prev.flag || 0,
            color_label: prev.color_label || 0,
            tags: prev.tags || [],
            camera: prev.camera || "", lens: prev.lens || "", focal_length: prev.focal_length || "",
            aperture: prev.aperture || "", shutter_speed: prev.shutter_speed || "", iso: prev.iso || "",
            capture_date: prev.capture_date || "", dimensions: prev.dimensions || "",
          };
        });

        // Apply pair_mode client-side for instant display
        basePhotosRef.current = _applyPendingEdits(rawFiles, pendingEditsRef.current);
        applyFilter(f);
        setLoadingPhotos(false);

        // Step 2: Index in background (EXIF extraction, catalog population)
        // Don't await — let it run while user browses
        const indexPromise = fetch(`${API}/catalog/index/${folderPath}`, { method: "POST" });

        // Step 3: Once indexed, reload with full catalog data (ratings, EXIF)
        indexPromise.then(async () => {
          indexedFoldersRef.current.add(folderPath);
          refreshCards();
          await fetchPhotos(folderPath, f, true);
          const statsRes = await fetch(`${API}/catalog/stats/${folderPath}`);
          setFolderStats(await statsRes.json());
          // After indexing, because both are only as good as the measurements
          // the folder has. An unmeasured folder simply reports nothing.
          refreshDerived(folderPath, filterRef.current);
        }).catch(() => {});

      } catch { basePhotosRef.current = []; setPhotos([]); setLoadingPhotos(false); }
    },
    [filter, fetchPhotos, applyFilter, refreshDerived, refreshCards]
  );

  // Leaving a folder for the folder list, or for a search across all of them:
  // nothing that was worked out for the open folder may badge other frames.
  const leaveFolder = useCallback(() => {
    setSelectedFolder(null);
    setPhotos([]);
    setCullIndex(null);
    setBursts([]);
    setBurstCoverage(null);
    setFolderHints(null);
    setResumeAt(null);
    derivedSeqRef.current += 1;
    if (derivedAbortRef.current) derivedAbortRef.current.abort();
    basePhotosRef.current = [];
  }, []);

  // Done with a folder: back to the overview, where it was left.
  const overviewScrollRef = useRef(null);
  const backToFolders = useCallback(() => {
    setReturnTo(selectedFolder);
    leaveFolder();
    loadFolders();
  }, [selectedFolder, leaveFolder, loadFolders]);
  const openFromOverview = useCallback((folder) => {
    overviewScrollRef.current = window.scrollY;
    openFolder(folder);
    window.scrollTo(0, 0);
  }, [openFolder]);

  // ── The browser's Back and Forward ──
  //
  // Where the app is — the tab, the open folder, the photo viewer — lives in the
  // address (#/library/2026/2026-07-04), one history entry per step in. Back
  // steps out of the viewer, then out of the folder, then to the last tab; a
  // reload or a bookmark opens the same folder. Leaving by Escape or "All
  // folders" goes back through history too, so Back never returns to a folder
  // that was just closed.
  const routeRef = useRef(null);
  const loadBasketRef = useRef(null);       // defined further down
  const poppingRef = useRef(null);
  const expectPopRef = useRef(false);
  const liveRef = useRef({});
  liveRef.current = { tab, selectedFolder, cullIndex };

  useEffect(() => {
    try { window.history.scrollRestoration = "manual"; } catch {}
  }, []);

  useEffect(() => {
    const route = {
      tab,
      folder: tab === "library" ? selectedFolder : null,
      viewer: tab === "library" && !!selectedFolder && cullIndex !== null,
    };
    const prev = routeRef.current;
    routeRef.current = route;
    if (!prev) {
      window.history.replaceState({ photool: route, depth: 0 }, "", routeHash(route));
      return;
    }
    if (sameRoute(prev, route)) return;
    if (poppingRef.current) {
      const target = poppingRef.current;
      if (sameRoute(target, route)) poppingRef.current = null;
      return;
    }
    const st = window.history.state;
    const outOfViewer = prev.viewer && !route.viewer && prev.folder === route.folder && prev.tab === route.tab;
    const outOfFolder = !prev.viewer && prev.folder && !route.folder && prev.tab === route.tab;
    if ((outOfViewer || outOfFolder) && st && st.photool && sameRoute(st.photool, prev) && st.depth > 0) {
      expectPopRef.current = true;
      window.history.back();
      return;
    }
    window.history.pushState({ photool: route, depth: ((st && st.depth) || 0) + 1 }, "", routeHash(route));
  }, [tab, selectedFolder, cullIndex !== null]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const onPop = (e) => {
      if (expectPopRef.current) { expectPopRef.current = false; return; }
      const asked = (e.state && e.state.photool) || parseRoute(window.location.hash)
        || { tab: access.local ? "import" : "library", folder: null, viewer: false };
      const live = liveRef.current;
      // The viewer cannot be reopened at the photo it showed; Forward into it
      // lands on its folder instead.
      const target = { ...asked, viewer: asked.viewer && live.cullIndex !== null };
      if (asked.viewer && !target.viewer) {
        window.history.replaceState({ ...(e.state || {}), photool: target }, "", routeHash(target));
      }
      poppingRef.current = target;
      if (target.tab !== live.tab) {
        setTab(target.tab);
        if (target.tab === "library") loadFolders();
        if (target.tab === "export") loadBasketRef.current?.();
      }
      if (target.tab === "library") {
        if (target.folder && target.folder !== live.selectedFolder) {
          if (!live.selectedFolder) overviewScrollRef.current = window.scrollY;
          openFolder(target.folder);
          window.scrollTo(0, 0);
        } else if (!target.folder && live.selectedFolder) {
          setReturnTo(live.selectedFolder);
          leaveFolder();
          loadFolders();
        }
      }
      if (!target.viewer && live.cullIndex !== null) setCullIndex(null);
      // Nothing changes when the address already matches what is shown.
      const now = { tab: live.tab, folder: live.tab === "library" ? live.selectedFolder : null,
        viewer: live.tab === "library" && !!live.selectedFolder && live.cullIndex !== null };
      if (sameRoute(now, target)) poppingRef.current = null;
    };
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, [openFolder, leaveFolder, loadFolders, access.local]);

  // A reload or a bookmark of a folder opens it. The address as the page was
  // loaded with: by the time this runs, the effect above has already written
  // the address of the folder list over it.
  const loadedRoute = useRef(null);
  if (loadedRoute.current === null) loadedRoute.current = parseRoute(window.location.hash) || false;
  useEffect(() => {
    const asked = loadedRoute.current;
    if (asked && asked.tab === "library" && asked.folder) openFolder(asked.folder);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // A folder opened from a search or the map may not have its card yet; a
  // folder seen for the first time gets one once indexing has put it in the
  // catalog. Indexing a big folder of RAWs reads every file's EXIF and can take
  // minutes, so one look a few seconds later often missed it and the card never
  // came: openFolder asks again the moment indexing finishes, and this keeps
  // looking, more slowly, while it has not.
  useEffect(() => {
    if (!selectedFolder || (folderCards[selectedFolder] && !folderCards[selectedFolder].pending)) return;
    let cancelled = false, timer = null, wait = 3000;
    const look = async () => {
      const fresh = await refreshCards();
      if (cancelled || fresh?.[selectedFolder]) return;
      timer = setTimeout(look, wait);
      wait = Math.min(wait * 2, 30000);
    };
    look();
    return () => { cancelled = true; clearTimeout(timer); };
  }, [selectedFolder]); // eslint-disable-line react-hooks/exhaustive-deps

  // A search answers for the whole library, so it takes the grid over from any
  // open folder. Each run takes a sequence number so a slow answer to an older
  // query cannot replace a newer one. A query that cannot be read keeps the
  // grid as it was and says which term is wrong.
  const runSearch = useCallback(async (query) => {
    const q = (query || "").trim();
    if (!q) return;
    const seq = ++searchSeqRef.current;
    const inFolder = selectedFolder && searchInFolder ? selectedFolder : "";
    try {
      const res = await fetch(`${API}/catalog/search?q=${encodeURIComponent(q)}${describeSearch ? "&describe=1" : ""}`
        + (inFolder ? `&folder=${encodeURIComponent(inFolder)}&limit=5000` : ""));
      const data = await res.json();
      if (seq !== searchSeqRef.current) return;
      if (!res.ok) { setSearchError(data.detail || "Search failed"); return; }
      if (inFolder) {
        setSearchError("");
        setCullIndex(null);
        setFilter((f) => ({ ...f, matches: { query: q, bases: new Set(data.photos.map((p) => p.base_name)) } }));
        setFolderSearch({ query: q, understood: data.understood, total: data.total, shown: data.photos.length,
          unclear: data.unclear || [], inFolder: true });
        return;
      }
      leaveFolder();
      setSearchError("");
      setPhotos(_applyPendingEdits(data.photos, pendingEditsRef.current));
      setSearchResult({ query: q, understood: data.understood, total: data.total, shown: data.photos.length,
        unclear: data.unclear || [], demoted: data.demoted || 0 });
    } catch {
      if (seq === searchSeqRef.current) setSearchError("Backend not reachable");
    }
  }, [leaveFolder, describeSearch, selectedFolder, searchInFolder]);

  const clearSearch = useCallback(() => {
    searchSeqRef.current += 1;
    setSearchText("");
    setSearchError("");
    if (searchResult) { setPhotos([]); setCullIndex(null); }
    setSearchResult(null);
    if (folderSearch) {
      setFolderSearch(null);
      setCullIndex(null);
      setFilter((f) => ({ ...f, matches: null }));
    }
  }, [searchResult, folderSearch]);

  // A filter change redraws the grid from the frames already in memory, so it is
  // instant, and only the derived views go back to the server — held back for a
  // beat so a run of clicks is one query for the last one, not a queue of
  // queries for all of them.
  useEffect(() => {
    if (!selectedFolder) return;
    const hidden = hiddenCameras(demoted, selectedFolder, filter.camera, filter.showDemoted);
    if (hidden.join("|") !== (filter.hidden || []).join("|")) setFilter((f) => ({ ...f, hidden }));
  }, [demoted, selectedFolder, filter.camera, filter.showDemoted]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!selectedFolder) return;
    applyFilter(filter);
    clearTimeout(derivedTimerRef.current);
    derivedTimerRef.current = setTimeout(() => refreshDerived(selectedFolder, filter), 250);
    return () => clearTimeout(derivedTimerRef.current);
  }, [filter, applyFilter, refreshDerived]);
  useEffect(() => { scanCards(); loadFolders(); checkStorage(); loadTags(); loadDemoted(); }, [scanCards, loadFolders, checkStorage, loadTags, loadDemoted]);

  // Remember where the user stopped reviewing, so reopening a folder can offer to
  // pick up there. Throttled because arrow keys move faster than the network
  // needs to, and this is browsing position, not cull data: a lost write costs
  // nothing worse than starting again from the top.
  useEffect(() => {
    if (!selectedFolder || cullIndex === null) return;
    const photo = photos[cullIndex];
    if (!photo) return;
    const last = savedPositionRef.current;
    if (last.folder === selectedFolder && last.path === photo.filepath) return;
    if (positionTimerRef.current) clearTimeout(positionTimerRef.current);
    positionTimerRef.current = setTimeout(() => {
      savedPositionRef.current = { folder: selectedFolder, path: photo.filepath };
      fetch(`${API}/catalog/folder-state/${selectedFolder}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ last_filepath: photo.filepath }),
      }).catch(() => {});
    }, 700);
    return () => clearTimeout(positionTimerRef.current);
  }, [cullIndex, photos, selectedFolder]);

  // Where a saved position lands in the list that is actually on screen. The
  // saved frame may have been decided and filtered out since it was written; the
  // list is sorted, so the next frame after it is where the review stands.
  // -1 means the position does not apply to this view, and no chip is offered.
  const resumeIndex = useCallback((filepath) => {
    const exact = photos.findIndex((p) => p.filepath === filepath);
    if (exact >= 0) return exact;
    return photos.findIndex((p) => p.filepath > filepath);
  }, [photos]);

  const clearResumePoint = useCallback(() => {
    setResumeAt(null);
    if (!selectedFolder) return;
    savedPositionRef.current = { folder: null, path: null };
    fetch(`${API}/catalog/folder-state/${selectedFolder}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ last_filepath: null }),
    }).catch(() => {});
  }, [selectedFolder]);

  // Closing the develop view lands on the frame it ended on. When it was opened
  // from the cull view, that view is still underneath, and leaving it on the
  // frame it started from would put the person back somewhere they have left.
  const closeDevelop = useCallback(() => {
    if (developFile !== null && cullIndex !== null) {
      const at = photos.findIndex((p) => p.filepath === developFile);
      if (at >= 0) setCullIndex(at);
      setCullSync((n) => n + 1);
    }
    setDevelopFile(null);
  }, [developFile, cullIndex, photos]);

  // A frame edited in the develop view keeps its marker in the grid, and the
  // marker survives a folder refetch for the same reason a rating does: the
  // list that comes back can predate the write.
  const handleEditSaved = useCallback((filepath, settings) => {
    if (!filepath) return;
    const stored = settings || {};
    const edits = pendingEditsRef.current.get(filepath) || {};
    edits.edit = stored;
    edits.edited = Object.keys(stored).length > 0;
    pendingEditsRef.current.set(filepath, edits);
    setPhotos((prev) => prev.map((p) => (p.filepath === filepath ? { ...p, ...edits } : p)));
  }, []);

  const handleRate = useCallback(
    async (filepath, field, value) => {
      // Remember the edit locally as well: a folder that is still indexing
      // refetches its list when indexing finishes, and that snapshot can predate
      // this write, which would make the star go dark again a second later.
      const edits = pendingEditsRef.current.get(filepath) || {};
      edits[field] = value;
      pendingEditsRef.current.set(filepath, edits);
      setPhotos((prev) => prev.map((p) => p.filepath === filepath ? { ...p, [field]: value } : p));
      try {
        const body = { filepath };
        body[field] = value;
        await fetch(`${API}/catalog/rate`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        if (selectedFolder) {
          const statsRes = await fetch(`${API}/catalog/stats/${selectedFolder}`);
          setFolderStats(await statsRes.json());
        }
      } catch (err) { console.error("Failed to save rating:", err); }
    },
    [selectedFolder]
  );

  const loadBasket = useCallback(async () => {
    try {
      const res = await fetch(`${API}/export/basket`);
      const data = await res.json();
      const items = data.items || [];
      setBasketItems(items);
      setBasketCount(items.length);
      setBasketSet(new Set(items.map((i) => i.filepath)));
    } catch {}
  }, []);
  loadBasketRef.current = loadBasket;

  const toggleBasket = useCallback(async (photo) => {
    const inBasket = basketSet.has(photo.filepath);
    if (inBasket) {
      await fetch(`${API}/export/basket/${encodeURIComponent(photo.filepath)}`, { method: "DELETE" }).catch(() => {});
    } else {
      await fetch(`${API}/export/basket/add`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filepath: photo.filepath, folder: photo.folder, filename: photo.filename }),
      }).catch(() => {});
    }
    loadBasket();
  }, [basketSet, loadBasket]);

  // A new folder or search starts with nothing picked.
  useEffect(() => {
    setPicked(new Set()); setPicking(false); setPickMessage(""); setPasteUndo(null); pickAnchorRef.current = null;
  }, [selectedFolder, searchResult]);

  const pickPhoto = useCallback((photo, index, e) => {
    setPickMessage("");
    // Read the anchor now: the updater may run after it has moved to this photo.
    const from = pickAnchorRef.current;
    setPicked((prev) => {
      const next = new Set(prev);
      if (e && e.shiftKey && from != null) {
        const [a, b] = [from, index].sort((x, y) => x - y);
        for (let i = a; i <= b; i++) if (photos[i]) next.add(photos[i].filepath);
      } else if (next.has(photo.filepath)) next.delete(photo.filepath);
      else next.add(photo.filepath);
      return next;
    });
    pickAnchorRef.current = index;
  }, [photos]);

  const clearPicked = useCallback(() => {
    setPicked(new Set()); setPicking(false); setPickMessage(""); setPasteUndo(null); pickAnchorRef.current = null;
  }, []);

  const editPicked = useCallback(async (change) => {
    const filepaths = photos.filter((p) => picked.has(p.filepath)).map((p) => p.filepath);
    if (!filepaths.length) return;
    setPickBusy(true);
    setPickMessage("");
    try {
      const res = await fetch(`${API}/catalog/batch`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filepaths, ...change }),
      });
      const out = await res.json();
      if (!res.ok) { setPickMessage(out.detail || "Not saved"); return; }
      const fields = ["rating", "flag", "color_label"].filter((f) => change[f] !== undefined);
      const add = change.add_tags || [];
      const remove = (change.remove_tags || []).map((t) => t.toLowerCase());
      const wanted = new Set(filepaths);
      for (const fp of filepaths) {
        const edits = pendingEditsRef.current.get(fp) || {};
        for (const f of fields) edits[f] = change[f];
        pendingEditsRef.current.set(fp, edits);
      }
      setPhotos((prev) => prev.map((p) => {
        if (!wanted.has(p.filepath)) return p;
        const next = { ...p };
        for (const f of fields) next[f] = change[f];
        if (add.length || remove.length) {
          const tagsNow = (p.tags || []).filter((t) => !remove.includes(t.toLowerCase()));
          for (const t of add) if (!tagsNow.includes(t)) tagsNow.push(t);
          next.tags = tagsNow;
          const edits = pendingEditsRef.current.get(p.filepath) || {};
          edits.tags = tagsNow;
          pendingEditsRef.current.set(p.filepath, edits);
        }
        return next;
      }));
      const n = out.photos;
      let msg = `${n} photo${n === 1 ? "" : "s"} updated`;
      if (out.faces_removed) msg += ` · ${out.faces_removed} face${out.faces_removed === 1 ? "" : "s"} taken out of that person`;
      setPickMessage(msg);
      if (add.length || remove.length) loadTags();
      if (selectedFolder) refreshFolderStats(selectedFolder);
    } catch { setPickMessage("The backend is not reachable"); } finally { setPickBusy(false); }
  }, [photos, picked, selectedFolder, loadTags]); // eslint-disable-line react-hooks/exhaustive-deps

  const basketPicked = useCallback(async () => {
    const items = photos.filter((p) => picked.has(p.filepath))
      .map((p) => ({ filepath: p.filepath, folder: p.folder || selectedFolder, filename: p.filename }));
    if (!items.length) return;
    setPickBusy(true);
    try {
      await fetch(`${API}/export/basket/add-batch`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ items }),
      });
      setPickMessage(`${items.length} photo${items.length === 1 ? "" : "s"} added to the basket`);
      loadBasket();
    } catch { setPickMessage("The backend is not reachable"); } finally { setPickBusy(false); }
  }, [photos, picked, selectedFolder, loadBasket]);

  // The edit copied in the develop view, onto every selected photo: the parts
  // its Copy checklist ticked, each photo keeping the rest of its own edit.
  const PASTE_ASKS_ABOVE = 20;
  const pastePicked = useCallback(async () => {
    const copied = copiedEdit();
    const filepaths = photos.filter((p) => picked.has(p.filepath)).map((p) => p.filepath);
    if (!copied || !filepaths.length) return;
    const mode = pasteAs();
    const how = PASTE_AS.find((p) => p.id === mode);
    // Undo lasts only while the selection does, so a paste onto many asks first.
    if (filepaths.length > PASTE_ASKS_ABOVE && !window.confirm(
      `Paste ${copied.sections.join(" + ")} from ${copied.source.split("/").pop()} onto ${filepaths.length} photos`
      + (mode === "copied" ? "?" : `, ${how.label.toLowerCase()}?`) + "\n\n"
      + "Each keeps the rest of its own edit. Undo is offered until you leave this selection.")) return;
    setPickBusy(true);
    setPickMessage("");
    setPasteUndo(null);
    try {
      const res = await fetch(`${API}/photos/edit/batch`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...copied.settings, filepaths, keys: copied.keys, source: copied.source, paste_as: mode }),
      });
      const out = await res.json();
      if (!res.ok) { setPickMessage(out.detail || "Not pasted"); return; }
      for (const [fp, stored] of Object.entries(out.edits || {})) handleEditSaved(fp, stored);
      const n = out.copied.length;
      const kept = out.kept_framing?.length || 0;
      const adjusted = pasteSummary(mode, out.notes);
      // The frames whose light itself changed, by name: a lighting cue may be a
      // mood worth keeping, and these are the ones to look at.
      const relit = lightChanged(out.notes).map((fp) => fp.split("/").pop());
      const named = relit.length ? ` (${relit.slice(0, 4).join(", ")}${relit.length > 4 ? ", …" : ""})` : "";
      setPickMessage(`Pasted ${copied.sections.join(" + ")} onto ${n} photo${n === 1 ? "" : "s"}`
        + (adjusted ? ` · ${adjusted}${named}` : "")
        + (kept ? ` · ${kept} turned the other way kept their own crop and spots` : ""));
      setPasteUndo(out.previous || null);
    } catch { setPickMessage("The backend is not reachable"); } finally { setPickBusy(false); }
  }, [photos, picked, handleEditSaved]);

  const undoPaste = useCallback(async () => {
    if (!pasteUndo) return;
    setPickBusy(true);
    try {
      const res = await fetch(`${API}/photos/edit/restore`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ edits: pasteUndo }),
      });
      if (!res.ok) { setPickMessage("Could not take the paste back"); return; }
      for (const [fp, stored] of Object.entries(pasteUndo)) handleEditSaved(fp, stored);
      const n = Object.keys(pasteUndo).length;
      setPasteUndo(null);
      setPickMessage(`Paste taken back from ${n} photo${n === 1 ? "" : "s"}`);
    } catch { setPickMessage("The backend is not reachable"); } finally { setPickBusy(false); }
  }, [pasteUndo, handleEditSaved]);

  // Ctrl/Cmd+A picks every photo shown; Escape lets go of them.
  useEffect(() => {
    const onKey = (e) => {
      if (tab !== "library" || cullIndex !== null || developFile !== null) return;
      if (e.target.closest?.("input, select, textarea")) return;
      // The viewer closes itself on Escape and may already be gone by the time
      // the key reaches here, so ask where the key came from, not what is open.
      if (e.target.closest?.("[data-modal]")) return;
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "a" && photos.length) {
        e.preventDefault();
        setPicked(new Set(photos.map((p) => p.filepath)));
      } else if (e.key === "Escape") {
        // A dialog open over the folder (its editor) takes its own Escape.
        if (document.querySelector("[data-modal]")) return;
        if (picked.size || picking) clearPicked();
        else if (selectedFolder) backToFolders();
        else if (searchResult) clearSearch();
      } else if (e.key.toLowerCase() === "e" && !e.ctrlKey && !e.metaKey && !e.altKey && selectedFolder
        && !document.querySelector("[data-modal]")) {
        document.querySelector('button[aria-label="Edit folder"]')?.click();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [tab, cullIndex, developFile, photos, picked, picking, clearPicked, selectedFolder, searchResult, backToFolders, clearSearch]);

  const clearBasket = useCallback(async () => {
    await fetch(`${API}/export/basket/clear`, { method: "DELETE" }).catch(() => {});
    loadBasket();
  }, [loadBasket]);

  // Bulk: copy the whole current selection into the basket in one call. The
  // server runs the same query the grid uses, so the basket gets exactly what
  // is on screen without shipping thousands of paths up front.
  const addFilteredToBasket = useCallback(async () => {
    if (!selectedFolder) return;
    setBulkBusy(true);
    setBulkMessage("");
    try {
      const res = await fetch(`${API}/export/basket/add-filtered`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ folder: selectedFolder, ...filterPayload(filter) }),
      });
      const data = await res.json();
      setBulkMessage(
        data.ok
          ? `Added ${data.added} to the export basket (${data.matched} matched).`
          : "Could not add the selection."
      );
      loadBasket();
    } catch (err) {
      console.error("Failed to add selection to basket:", err);
      setBulkMessage("Could not add the selection.");
    } finally { setBulkBusy(false); }
  }, [selectedFolder, filter, loadBasket]);

  // Bulk: tag or untag the same selection. Each match writes a keyword into
  // its sidecar, so this takes a moment on a full folder.
  const applyTagToSelection = useCallback(async (tag, remove) => {
    if (!selectedFolder) return;
    setBulkBusy(true);
    setBulkMessage("");
    try {
      const res = await fetch(`${API}/catalog/tags/apply`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ folder: selectedFolder, tag, remove, ...filterPayload(filter) }),
      });
      const data = await res.json();
      const n = remove ? data.untagged ?? 0 : data.tagged ?? 0;
      setBulkMessage(
        remove ? `Removed "${tag}" from ${n} photos.` : `Tagged ${n} photos "${tag}".`
      );
      await Promise.all([loadTags(), fetchPhotos(selectedFolder, filter, true)]);
    } catch (err) {
      console.error("Failed to tag selection:", err);
      setBulkMessage("Tagging failed.");
    } finally { setBulkBusy(false); }
  }, [selectedFolder, filter, fetchPhotos, loadTags]);

  // filepath -> the burst it belongs to, for the cull strip, and filepath ->
  // group size, for the small badge on each grid tile.
  const burstByPath = useMemo(() => {
    const map = new Map();
    for (const group of bursts) {
      for (const member of group.members || []) map.set(member.filepath, group);
    }
    return map;
  }, [bursts]);

  const burstSizes = useMemo(() => {
    const map = new Map();
    for (const group of bursts) {
      for (const member of group.members || []) map.set(member.filepath, group.size);
    }
    return map;
  }, [bursts]);

  const refreshFolderStats = useCallback(async (folderPath) => {
    try {
      const res = await fetch(`${API}/catalog/stats/${folderPath}`);
      setFolderStats(await res.json());
    } catch {}
  }, []);

  // Shift+R in the cull view: keep one frame, reject the rest of the burst.
  const rejectBurstRest = useCallback(async (keep, others) => {
    if (!others || others.length === 0) return;
    setPhotos((prev) => prev.map((p) => (others.includes(p.filepath) ? { ...p, flag: -1 } : p)));
    setBursts((prev) => prev.filter((g) => !(g.members || []).some((m) => others.includes(m.filepath))));
    try {
      const res = await fetch(`${API}/catalog/bursts/reject-rest`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ keep, others }),
      });
      const data = await res.json();
      if (res.ok) {
        setBulkMessage(`Rejected ${data.rejected.length} frame${data.rejected.length === 1 ? "" : "s"} of that burst.`);
        if (selectedFolder) refreshFolderStats(selectedFolder);
      } else {
        setBulkMessage(data.detail || "Could not reject the burst.");
      }
    } catch (err) {
      console.error("Failed to reject burst:", err);
      setBulkMessage("Could not reject the burst.");
    }
  }, [selectedFolder, refreshFolderStats]);

  // Whole folder: keep the sharpest frame of every burst, reject the rest.
  // The measurement hints, with "eyes closed?" added where a burst's faces have
  // been looked at (backend analysis/eyes.py).
  const hintsWithEyes = useMemo(() => {
    const closed = bursts.flatMap((g) => g.closed || []);
    if (!closed.length) return folderHints ? folderHints.hints : null;
    const out = { ...(folderHints?.hints || {}) };
    for (const fp of closed) out[fp] = { ...(out[fp] || {}), eyes_closed: true };
    return out;
  }, [bursts, folderHints]);

  const checkBurstFaces = useCallback(async () => {
    if (!selectedFolder) return;
    setBulkBusy(true);
    setBulkMessage(`Looking at the faces in ${bursts.reduce((n, g) => n + g.size, 0)} burst frames…`);
    try {
      const res = await fetch(`${API}/catalog/bursts/check-faces`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ folder: selectedFolder, ...filterPayload(filter) }),
      });
      const data = await res.json();
      if (!res.ok) { setBulkMessage(data.detail || "Could not look at the faces"); return; }
      setBursts(data.groups || []);
      const closed = (data.groups || []).reduce((n, g) => n + (g.closed || []).length, 0);
      setBulkMessage(`Faces checked: each burst now keeps the frame with its people at their best`
        + (closed ? `; ${closed} frame${closed === 1 ? "" : "s"} where someone's eyes may be closed are marked "eyes?".` : "."));
    } catch {
      setBulkMessage("Backend not reachable");
    } finally {
      setBulkBusy(false);
    }
  }, [selectedFolder, filter, bursts]);

  // Accept, dismiss or undo an agent's proposals: every waiting one in the
  // folder, or one frame from the photo view. One frame accepted by hand
  // replaces the marks it had, because that is what the person just chose;
  // "Accept all" keeps any mark the person has already set.
  const actOnProposals = useCallback(async (action, filepaths = null) => {
    if (!selectedFolder) return;
    const scope = filepaths ? { filepaths } : { folder: selectedFolder };
    const one = !!filepaths && filepaths.length === 1;
    const taking = action === "accept" ? (filepaths || Object.keys(proposals)) : [];
    setProposalBusy(true);
    try {
      const res = await fetch(`${API}/commands/cull.${action}`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(action === "accept" ? { ...scope, overwrite: one } : scope),
      });
      const data = await res.json();
      if (!res.ok) { setProposalMessage(data.detail || `Could not ${action}`); return; }
      if (action === "accept") {
        setLastAccepted(taking);
        if (!one) {
          setProposalMessage(`Accepted ${data.accepted}: ${data.changed} mark${data.changed === 1 ? "" : "s"} set`
            + (data.kept ? `, ${data.kept} of your own kept.` : "."));
        }
        // What the frame looks like now: the proposal's marks, kept locally as a
        // rating is, so a folder refetch that predates the write cannot undo it.
        if (one) {
          const p = proposals[filepaths[0]] || {};
          const fields = Object.fromEntries(["rating", "flag", "color_label"]
            .filter((k) => p[k] !== null && p[k] !== undefined).map((k) => [k, p[k]]));
          const edits = { ...(pendingEditsRef.current.get(filepaths[0]) || {}), ...fields };
          pendingEditsRef.current.set(filepaths[0], edits);
          setPhotos((prev) => prev.map((x) => (x.filepath === filepaths[0] ? { ...x, ...fields } : x)));
        }
      } else if (action === "dismiss") {
        if (!one) setProposalMessage(`Dismissed ${data.dismissed}.`);
      } else if (action === "undo") {
        setLastAccepted([]);
        setProposalMessage(`Undone: ${data.restored} mark${data.restored === 1 ? "" : "s"} put back`
          + (data.kept ? `, ${data.kept} you changed since left as they are.` : "."));
      }
      if (!one) {
        // Marks changed on the server for many frames at once: start from what it holds.
        for (const fp of [...taking, ...(action === "undo" ? lastAccepted : [])]) pendingEditsRef.current.delete(fp);
        await fetchPhotos(selectedFolder, filter, true);
        refreshFolderStats(selectedFolder);
      }
      await loadProposals(selectedFolder);
    } catch {
      setProposalMessage("Backend not reachable");
    } finally {
      setProposalBusy(false);
    }
  }, [selectedFolder, filter, proposals, lastAccepted, fetchPhotos, refreshFolderStats, loadProposals]);

  // A message or an undo is about the folder it was given in.
  useEffect(() => { setProposalMessage(""); setLastAccepted([]); }, [selectedFolder]);

  const suggestBest = useCallback(async (count) => {
    if (!selectedFolder) return;
    setBulkBusy(true);
    setBulkMessage("");
    try {
      const res = await fetch(`${API}/catalog/best`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ folder: selectedFolder, count, ...filterPayload(filter) }),
      });
      const data = await res.json();
      if (!res.ok) { setBulkMessage(data.detail || "Could not suggest"); return; }
      setPicked(new Set(data.filepaths || []));
      setPicking(true);
      setBulkMessage(`Selected the ${data.filepaths.length} best of ${data.considered} moments — add them to the basket, flag or tag them from the selection bar.`
        + (bursts.length && !bursts.every((g) => g.faces_checked) ? " ☺ Check faces first to let closed eyes count." : ""));
    } catch {
      setBulkMessage("Backend not reachable");
    } finally {
      setBulkBusy(false);
    }
  }, [selectedFolder, filter, bursts]);

  const rejectAllBursts = useCallback(async () => {
    if (!selectedFolder) return;
    // Exactly what the server is about to touch — the keeper and any frame that
    // is already rejected are not in this count.
    const doomed = bursts.reduce(
      (sum, g) => sum + (g.members || []).filter((m) => m.flag !== -1 && m.filepath !== g.keep).length,
      0
    );
    if (doomed === 0) return;
    if (!window.confirm(
      `Reject ${doomed} near-identical frame${doomed === 1 ? "" : "s"} in this folder, keeping the ${bursts.every((g) => g.faces_checked) ? "best" : "sharpest"} of each burst?\n\n` +
      `Rejected photos are dimmed and left out of the basket. Press U on one to undo it.`
    )) return;
    setBulkBusy(true);
    setBulkMessage("");
    try {
      const res = await fetch(`${API}/catalog/bursts/reject-duplicates`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ folder: selectedFolder, ...filterPayload(filter) }),
      });
      const data = await res.json();
      if (res.ok) {
        setBulkMessage(`Kept the sharpest frame of ${data.groups} burst${data.groups === 1 ? "" : "s"}, rejected ${data.rejected}.`);
      } else {
        setBulkMessage(data.detail || "Could not clean up duplicates.");
      }
      await Promise.all([
        fetchPhotos(selectedFolder, filter, true),
        refreshDerived(selectedFolder, filter),
      ]);
      refreshFolderStats(selectedFolder);
    } catch (err) {
      console.error("Failed to clean up duplicates:", err);
      setBulkMessage("Could not clean up duplicates.");
    } finally { setBulkBusy(false); }
  }, [selectedFolder, filter, bursts, fetchPhotos, refreshDerived, refreshFolderStats]);

  // Single photo, from the cull view. A tag is an edit like any other: it is
  // remembered locally too, so changing a filter cannot show the folder as it was
  // before the keyword was written.
  const handleSetTags = useCallback(async (filepath, nextTags) => {
    const edits = pendingEditsRef.current.get(filepath) || {};
    edits.tags = nextTags;
    pendingEditsRef.current.set(filepath, edits);
    setPhotos((prev) => prev.map((p) => p.filepath === filepath ? { ...p, tags: nextTags } : p));
    try {
      const res = await fetch(`${API}/catalog/tags`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filepath, tags: nextTags }),
      });
      const data = await res.json();
      if (data.ok) {
        edits.tags = data.tags;
        setPhotos((prev) => prev.map((p) => p.filepath === filepath ? { ...p, tags: data.tags } : p));
        loadTags();
      }
    } catch (err) { console.error("Failed to save tags:", err); }
  }, [loadTags]);

  // Naming a face in the viewer puts the person's name on the photo as a tag.
  const refreshAfterFaces = useCallback(() => {
    if (selectedFolder) fetchPhotos(selectedFolder, filter, true);
    loadTags();
  }, [selectedFolder, filter, fetchPhotos, loadTags]);

  useEffect(() => { loadBasket(); }, [loadBasket]);

  const runImport = async (mountPoint) => {
    setImporting(true); setLastResult(null); setImportProgress(null);
    try {
      const res = await fetch(`${API}/import`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mount_point: mountPoint, verify_checksums: true }),
      });
      const job = await res.json();
      if (!job.job_id) {
        setLastResult({ total_files: 0, copied: 0, skipped: 0, failed: 0, elapsed_seconds: 0 });
        return;
      }

      // The import runs server-side; poll so the UI shows which photos it is
      // working on instead of appearing to hang on a full card.
      let final = null;
      while (true) {
        const p = await fetch(`${API}/import/progress/${job.job_id}`).then((r) => r.json());
        setImportProgress(p);
        if (p.status !== "running") { final = p; break; }
        await new Promise((r) => setTimeout(r, 1000));
      }

      if (final.status === "failed") {
        setLastResult({ error: final.error || "Import failed" });
      } else {
        setLastResult(final.result);
      }
      loadFolders();
    } catch (err) { setLastResult({ error: err.message }); }
    finally { setImporting(false); }
  };

  // A saved position only earns a button if it lands somewhere in the current
  // view. A frame that has been deleted, or that the filters hide, does not; the
  // next cull session writes a new position over it.
  const resumeTarget = resumeAt ? resumeIndex(resumeAt) : -1;

  // The warnings for one frame, if this folder has been measured at all.
  const hintFor = (filepath) => (filepath && hintsWithEyes ? hintsWithEyes[filepath] ?? null : null);

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100">
      {slideshow && (
        <Slideshow photos={slideshow.photos} start={slideshow.start} title={slideshow.title}
          onClose={() => setSlideshow(null)} />
      )}
      {developFile && (
        // Keyed on the frame, so moving with ←/→ starts clean and recovers from
        // a frame that failed rather than staying stuck on its error.
        <PanelBoundary key={developFile} onClose={closeDevelop}>
          <DevelopView
            filepath={developFile}
            photos={photos.filter((p) => !isVideo(p)).map((p) => ({
              filepath: p.filepath, filename: p.filename, folder: p.folder,
            }))}
            onClose={closeDevelop}
            onNavigate={(filepath) => setDevelopFile(filepath)}
            onSaved={handleEditSaved}
          />
        </PanelBoundary>
      )}

      {cullIndex !== null && (selectedFolder || searchResult) && (
        <CullView photos={photos} folderPath={selectedFolder} startIndex={cullIndex}
          onClose={() => setCullIndex(null)} onRate={handleRate}
          basketSet={basketSet} onToggleBasket={toggleBasket}
          allTags={tags} onSetTags={handleSetTags}
          burst={burstByPath.get(photos[cullIndex]?.filepath) ?? null}
          hint={hintFor(photos[cullIndex]?.filepath)}
          proposal={proposals[photos[cullIndex]?.filepath] ?? null}
          onProposal={(action, photo) => actOnProposals(action, [photo.filepath])}
          onDevelop={(photo) => openDevelop(photo.filepath)}
          developing={developFile !== null}
          syncToken={cullSync}
          onFacesChanged={refreshAfterFaces}
          onRejectBurst={rejectBurstRest} />
      )}

      <header className={`border-b border-neutral-800 px-4 lg:px-6 flex flex-wrap items-center gap-x-6 gap-y-2 ${phone ? "py-2" : "py-3"}`}
        style={phone ? { paddingTop: "max(0.5rem, env(safe-area-inset-top))" } : undefined}>
        <h1 className="text-lg font-semibold tracking-tight whitespace-nowrap">Photool</h1>
        <nav className={phone ? "hidden" : "flex flex-wrap gap-1"}>
          {["import", "library", "people", "activity", "export", ...(access.local ? ["devices"] : [])].map((t) => (
            <button key={t} onClick={() => { setTab(t); if (t === "library") loadFolders(); if (t === "export") loadBasket(); }}
              className={`px-3 py-1.5 rounded text-sm capitalize transition-colors ${
                tab === t ? "bg-neutral-800 text-white" : "text-neutral-500 hover:text-neutral-300"
              }`}>{t === "export" ? `Export${basketCount > 0 ? ` (${basketCount})` : ""}` : t}</button>
          ))}
        </nav>
        <BackgroundBadge onOpen={() => setTab("activity")} />
        {!access.local && access.device && (
          <span className="text-xs text-neutral-500">{access.device}</span>
        )}
      </header>

      {access.local && <PairingRequests />}

      {/* On a phone the tabs are at the bottom, under the thumb. Import, Activity
          and Devices are things for the computer, under More. */}
      {phone && cullIndex === null && developFile === null && !(picked.size > 0 || picking) && (
        <nav className="fixed inset-x-0 bottom-0 z-40 bg-neutral-950/95 backdrop-blur border-t border-neutral-800 flex"
          style={{ paddingBottom: "env(safe-area-inset-bottom)" }} data-testid="tab-bar">
          {[["library", "Library", "▦"], ["people", "People", "☺"],
            ["export", basketCount > 0 ? `Export ${basketCount}` : "Export", "⬡"]].map(([t, label, icon]) => (
            <button key={t} onClick={() => { setMoreOpen(false); setTab(t); if (t === "library") loadFolders(); if (t === "export") loadBasket(); }}
              className={`flex-1 h-14 flex flex-col items-center justify-center text-[11px] ${tab === t ? "text-white" : "text-neutral-500"}`}>
              <span className="text-lg leading-none">{icon}</span>{label}
            </button>
          ))}
          <button onClick={() => setMoreOpen((o) => !o)} data-testid="tab-more"
            className={`flex-1 h-14 flex flex-col items-center justify-center text-[11px] ${
              ["import", "activity", "devices"].includes(tab) || moreOpen ? "text-white" : "text-neutral-500"}`}>
            <span className="text-lg leading-none">⋯</span>More
          </button>
          {moreOpen && (
            <div className="absolute bottom-full right-2 mb-2 w-44 rounded-lg border border-neutral-800 bg-neutral-900 py-1 shadow-xl">
              {["import", "activity", ...(access.local ? ["devices"] : [])].map((t) => (
                <button key={t} onClick={() => { setMoreOpen(false); setTab(t); }}
                  className={`block w-full text-left px-4 py-3 text-sm capitalize ${tab === t ? "text-white" : "text-neutral-300"}`}>{t}</button>
              ))}
            </div>
          )}
        </nav>
      )}

      {storageWarning && (
        <div className="bg-red-950/60 border-b border-red-900 px-6 py-3 text-sm">
          <p className="font-semibold text-red-300">
            Storage is read-only — ratings, imports and thumbnails are not being saved.
          </p>
          {Object.values(storageWarning)
            .filter((b) => b && typeof b === "object" && !b.writable)
            .map((b) => (
              <p key={b.path} className="text-red-400/80 mt-1">
                {b.label}: {b.path} — {b.error}
              </p>
            ))}
          <p className="text-neutral-400 mt-1">
            Fix the drive, or point db_path / thumbnail_cache in backend/config.yaml at a writable location.
          </p>
        </div>
      )}

      {noExiftool && !exiftoolNoted && (
        <div className="bg-amber-950/50 border-b border-amber-900 px-6 py-2 text-sm flex items-start gap-3"
             data-testid="no-exiftool">
          <p className="text-amber-200/90 flex-1">
            ExifTool isn't installed, so capture settings, lens data and the camera's Picture Style can't be
            read. Install it (macOS: <code>brew install exiftool</code>, Windows:{" "}
            <code>winget install OliverBetz.ExifTool</code>) and restart Photool.
          </p>
          <button onClick={() => setExiftoolNoted(true)} className="text-amber-300/70 hover:text-amber-200"
                  title="Hide until the next visit">✕</button>
        </div>
      )}

      {/* Library and People use the whole width of the window; the form-like
          tabs keep a reading width so a line of text does not run 3000px. */}
      <main className={`mx-auto lg:px-6 ${phone ? "px-2 pt-2 pb-24" : "px-4 py-4"} ${
        tab === "library" || tab === "people" ? "w-full" : "max-w-6xl"}`}>
        {tab === "import" && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-medium">Import from a camera card</h2>
              <button onClick={scanCards} className="text-sm text-neutral-400 hover:text-white">↻ Rescan</button>
            </div>
            {sdCards.length === 0 ? (
              <div className="border border-dashed border-neutral-700 rounded-lg p-12 text-center text-neutral-500">
                <p className="text-lg mb-2">No camera cards detected</p>
                <p className="text-sm">Insert a card (or plug in a camera, GoPro or DJI Pocket as a drive) and click Rescan.</p>
              </div>
            ) : (
              <div className="space-y-3">
                {sdCards.map((card) => (
                  <div key={card.mount_point} className="border border-neutral-800 rounded-lg p-4 flex items-center justify-between">
                    <div>
                      <p className="font-medium">
                        {card.mount_point}
                        {card.device && <span className="ml-2 text-neutral-400 font-normal">{card.device}</span>}
                      </p>
                      <p className="text-sm text-neutral-400">
                        {card.total_files - (card.videos || 0)} photo files
                        {card.videos > 0 && ` · ${card.videos} videos`} · {card.total_size_mb} MB
                      </p>
                    </div>
                    <button onClick={() => runImport(card.mount_point)} disabled={importing}
                      className="px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-neutral-700 rounded text-sm font-medium">
                      {importing && importProgress
                        ? `Importing… ${importProgress.completed}/${importProgress.total}`
                        : importing ? "Importing…" : "Import All"}
                    </button>
                  </div>
                ))}
              </div>
            )}
            {importing && importProgress && (() => {
              const total = importProgress.total || 1;
              const pct = Math.min(100, (importProgress.completed / total) * 100);
              return (
                <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-4 text-sm space-y-2">
                  <div className="flex items-center justify-between gap-4">
                    <span className="text-neutral-200">
                      {IMPORT_STAGES[importProgress.stage] || importProgress.stage}
                    </span>
                    <span className="text-neutral-500 font-mono shrink-0">
                      {importProgress.completed}/{importProgress.total} · {importProgress.elapsed_seconds}s
                    </span>
                  </div>
                  <div className="h-1.5 bg-neutral-800 rounded overflow-hidden">
                    <div className="h-full bg-blue-600 transition-all duration-500"
                      style={{ width: `${pct}%` }} />
                  </div>
                  {importProgress.current_file && (
                    <p className="text-xs text-neutral-500 font-mono truncate">
                      {importProgress.current_file}
                    </p>
                  )}
                </div>
              );
            })()}
            {lastResult && !lastResult.error && (
              <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-4 text-sm space-y-1">
                <p className="text-green-400">Import complete</p>
                <p>Copied: {lastResult.copied} · Skipped: {lastResult.skipped} · Failed: {lastResult.failed}</p>
                <p className="text-neutral-500">{lastResult.elapsed_seconds}s</p>
              </div>
            )}
          </div>
        )}

        {tab === "library" && (
          <div className="space-y-4">
            {!selectedFolder && (
              <div className="flex items-center gap-3">
                <h2 className="text-xl font-medium">Library</h2>
                {searchResult && photos.length > 0 && (
                  <button onClick={() => (picking || picked.size ? clearPicked() : setPicking(true))}
                    className={`order-last px-2 py-0.5 rounded border text-xs ${picking || picked.size
                      ? "border-sky-500 text-sky-300" : "border-neutral-800 text-neutral-400 hover:text-white"}`}
                    title="Pick photos to rate, flag, tag or basket together (or Ctrl/Cmd-click, Shift-click)">
                    {picking || picked.size ? "Done" : "Select"}
                  </button>
                )}
              </div>
            )}

            {selectedFolder && (
              <FolderAbout folder={selectedFolder} card={folderCards[selectedFolder] || null}
                onCards={setFolderCards}
                before={
                  <button onClick={backToFolders} title="Back to all folders (Esc)"
                    aria-label="All folders"
                    className="shrink-0 px-2.5 py-1 rounded border border-neutral-700 text-sm text-neutral-300 hover:text-white hover:border-neutral-500">
                    ←<span className="hidden sm:inline"> All folders</span></button>
                }
                after={cameras.length < 2 && (
                  // One camera: less important is about the folder, so it sits with the folder.
                  <ShootMenu folder={selectedFolder} cameras={cameras} demoted={demoted} onSet={setShootDemoted} />
                )} />
            )}

            {/* In a folder the search sits in the folder's panel, below. */}
            {!selectedFolder && (
              <SearchBar value={searchText} onChange={setSearchText}
                onSubmit={runSearch} onClear={clearSearch}
                result={searchResult} error={searchError}
                describe={describeSearch} onDescribe={toggleDescribe}
                smart={smartSearch} onSetup={setupSmartSearch} />
            )}

            {searchResult && (
              photos.length === 0 ? (
                <p className="text-neutral-500">No photos match this search.</p>
              ) : (
                <>
                <button onClick={() => playSlides(photos, 0, searchResult.query)}
                  className="px-2.5 py-1 rounded bg-neutral-800 text-sky-200 hover:bg-neutral-700 text-sm"
                  title="Show these photos full screen, one at a time">▶ Slideshow</button>
                <ImageGrid photos={photos} folderPath={null}
                  onSelect={(photo) => {
                    const idx = photos.findIndex((p) => p.filepath === photo.filepath);
                    setCullIndex(idx >= 0 ? idx : 0);
                  }}
                  onDevelop={(photo) => openDevelop(photo.filepath)}
                  onRate={handleRate}
                  basketSet={basketSet} onToggleBasket={toggleBasket}
                  selected={picked} selecting={picking} onPick={pickPhoto} />
                </>
              )
            )}

            {!selectedFolder && !searchResult && (
              folders.length === 0 ? (
                <p className="text-neutral-500">No photos in library. Import from SD card first.</p>
              ) : (
                <FolderOverview folders={folders} onOpen={openFromOverview} onCards={setFolderCards}
                  onPlay={playSlides} demoted={demoted}
                  returnTo={returnTo} restoreScroll={overviewScrollRef.current} />
              )
            )}

            {selectedFolder && (
              <>
                {/* One panel for the folder's controls: narrow the grid (search,
                    camera, filters, with how far the review has got), then act
                    on what is left in view. */}
                <section className="rounded-lg border border-neutral-800 bg-neutral-900/40" data-testid="folder-panel">
                  <div className="p-3">
                    <div className="flex items-start gap-x-3 gap-y-2 flex-wrap">
                      <div className="w-full lg:w-[26rem] lg:shrink-0">
                        <SearchBar value={searchText} onChange={setSearchText}
                          onSubmit={runSearch} onClear={clearSearch}
                          result={searchResult || folderSearch} error={searchError}
                          describe={describeSearch} onDescribe={toggleDescribe}
                          smart={smartSearch} onSetup={setupSmartSearch}
                          folder={selectedFolder} inFolder={searchInFolder} onInFolder={setSearchInFolder} />
                      </div>
                      {phone && !searchResult && (
                        <button onClick={() => setFiltersOpen((o) => !o)} data-testid="phone-filters"
                          className={`h-9 px-3 rounded border text-sm ${activeFilters
                            ? "border-sky-700 text-sky-300" : "border-neutral-700 text-neutral-300"}`}>
                          Filters{activeFilters ? ` · ${activeFilters}` : ""} {filtersOpen ? "▴" : "▾"}
                        </button>
                      )}
                      {phone && !searchResult && !filtersOpen && (
                        <button onClick={() => { setCullIndex(null); setFilter((f) => ({ ...f, unreviewed: !f.unreviewed })); }}
                          data-testid="phone-undecided"
                          className={`h-9 px-3 rounded border text-sm ${filter.unreviewed
                            ? "border-sky-700 text-sky-300 bg-sky-950/40" : "border-neutral-700 text-neutral-300"}`}>
                          Undecided only
                        </button>
                      )}
                      {!searchResult && (!phone || filtersOpen) && <FilterBar filter={filter} onFilterChange={setFilter} tags={tags} />}
                      {!searchResult && (!phone || filtersOpen) && (cameras.length > 1 || filter.showDemoted || shown?.hidden > 0) && (
                        <div className="flex items-center gap-1.5 flex-wrap self-center" data-testid="camera-group">
                          <CameraChips cameras={cameras} value={filter.camera}
                            demoted={new Set(demoted.filter((d) => d.folder === selectedFolder).map((d) => d.camera))}
                            onChange={(camera) => { setCullIndex(null); setFilter((f) => ({ ...f, camera })); }} />
                          {cameras.length > 1 && (
                            <ShootMenu folder={selectedFolder} cameras={cameras} demoted={demoted} onSet={setShootDemoted} />
                          )}
                          <HiddenShots
                            cameras={demoted.filter((d) => d.folder === selectedFolder && d.camera !== "*"
                              && d.camera !== filter.camera).map((d) => d.camera)}
                            count={shown?.hidden || 0} showing={filter.showDemoted}
                            onShow={(on) => { setCullIndex(null); setFilter((f) => ({ ...f, showDemoted: on })); }} />
                        </div>
                      )}
                    </div>
                  </div>
                  {/* Acting on what is in view, how far the review has got, and
                      how big the thumbnails are. */}
                  <div className="border-t border-neutral-800 px-3 py-2 flex items-center gap-x-4 gap-y-2 flex-wrap">
                    {(!phone || filtersOpen) && <FolderActions
                      count={photos.length}
                      tags={tags}
                      busy={bulkBusy}
                      message={bulkMessage}
                      onAddToBasket={addFilteredToBasket}
                      onSlideshow={() => playSlides(photos, 0, folderCards[selectedFolder]?.title || selectedFolder)}
                      onCheckFaces={checkBurstFaces}
                      onSuggestBest={suggestBest}
                      facesChecked={bursts.length > 0 && bursts.every((g) => g.faces_checked)}
                      onApplyTag={applyTagToSelection}
                      bursts={bursts}
                      burstCoverage={burstCoverage}
                      onRejectDuplicates={rejectAllBursts}
                      quality={folderHints}
                      measuring={measuring}
                      onMeasure={() => measureFolder(selectedFolder, filter)}
                      selecting={picking || picked.size > 0}
                      onSelect={() => (picking || picked.size ? clearPicked() : setPicking(true))}
                    />}
                    <div className="flex items-center gap-4 sm:ml-auto flex-wrap">
                      <ReviewProgress
                        stats={shown && folderStats ? { ...folderStats, ...shown } : folderStats}
                        resume={resumeTarget >= 0 ? resumeAt : null}
                        onResume={() => setCullIndex(resumeTarget)}
                        onClearResume={clearResumePoint} />
                      <div className={`${phone ? "hidden" : "flex"} items-center gap-1 text-xs text-neutral-500`} title="Thumbnail size">
                        {Object.keys(THUMB_SIZES).map((k) => (
                          <button key={k} onClick={() => setThumbSize(k)}
                            className={`px-1.5 py-0.5 rounded ${thumbSize === k
                              ? "bg-neutral-800 text-white" : "hover:text-neutral-300"}`}>{k}</button>
                        ))}
                      </div>
                    </div>
                  </div>
                </section>
                <ProposalBar proposals={proposals} busy={proposalBusy} message={proposalMessage}
                  canUndo={lastAccepted.length > 0}
                  onAccept={() => actOnProposals("accept")}
                  onDismiss={() => actOnProposals("dismiss")}
                  onUndo={() => actOnProposals("undo", lastAccepted)}
                  onSelect={() => { setPicked(new Set(Object.keys(proposals))); setPicking(true); }} />
                {loadingPhotos ? (
                  <p className="text-neutral-500">Loading…</p>
                ) : photos.length === 0 ? (
                  <p className="text-neutral-500">
                    {filter.unreviewed
                      ? "Nothing left to decide in this folder — every frame carries a rating, flag, or colour label."
                      : "No photos match the current filter."}
                  </p>
                ) : (
                  <ImageGrid photos={photos} folderPath={selectedFolder}
                    onSelect={(photo) => {
                      const idx = photos.findIndex((p) => p.filepath === photo.filepath);
                      setCullIndex(idx >= 0 ? idx : 0);
                    }}
                    onDevelop={(photo) => openDevelop(photo.filepath)}
                    onRate={handleRate}
                    basketSet={basketSet} onToggleBasket={toggleBasket}
                    burstSizes={burstSizes}
                    cellSize={phone ? PHONE_CELL : THUMB_SIZES[thumbSize] || THUMB_SIZES.M}
                    hintByPath={hintsWithEyes}
                    proposalByPath={proposals}
                    selected={picked} selecting={picking} onPick={pickPhoto} />
                )}
              </>
            )}
          </div>
        )}

        {tab === "library" && (selectedFolder || searchResult) && (
          <SelectionBar photos={photos} selected={picked} tags={tags} busy={pickBusy}
            message={pickMessage} selecting={picking} onEdit={editPicked} onBasket={basketPicked}
            onCollage={() => setCollageOf(photos.filter((p) => picked.has(p.filepath)).map((p) => p.filepath))}
            onSelectAll={() => setPicked(new Set(photos.map((p) => p.filepath)))} onClear={clearPicked}
            copied={clipboard} onPaste={pastePicked} onUndoPaste={pasteUndo ? undoPaste : null} />
        )}
        {tab === "library" && (picked.size > 0 || picking) && <div className="h-28" />}
        {collageOf && <CollageMaker filepaths={collageOf} onClose={() => setCollageOf(null)} />}

        {tab === "devices" && access.local && <DevicesPanel lanAccess={access.lan_access} />}

        {tab === "activity" && (
          <div className="space-y-4">
            <h2 className="text-xl font-medium">Activity</h2>
            <ActivityPanel local={access.local} />
          </div>
        )}

        {tab === "people" && <PeoplePanel folder={selectedFolder} />}

        {tab === "export" && (
          <div className="space-y-4">
            <h2 className="text-xl font-medium">Export</h2>
            <ExportPanel basketItems={basketItems} basketCount={basketCount}
              onRefreshBasket={loadBasket} onClearBasket={clearBasket}
              onDevelop={(filepath) => openDevelop(filepath)} onCollage={setCollageOf} />
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
