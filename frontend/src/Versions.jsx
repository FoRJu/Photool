import { useEffect, useState, useCallback } from "react";

// ── A photo's versions, side by side ────────────────────────────────
// The photograph's own edit and each of its virtual copies, drawn as they look.
// One of them is the cover: the version the grid, the slideshow, On this day
// and the year's best show for this photo (backend catalog/service.py, cover).
// The keeper is often a copy, the black-and-white or the crop made second, and
// choosing it here brings it out from behind the photo's own edit. Exporting
// is not changed by it.

const API = "/api";

function versionUrl(filepath, version, size = "preview", key = "") {
  const cut = filepath.lastIndexOf("/");
  return `${API}/thumb/${filepath.slice(0, cut)}/${encodeURIComponent(filepath.slice(cut + 1))}`
    + `?size=${size}&ver=${version}${key ? `&v=${key}` : ""}`;
}

export default function VersionsSheet({ photo, onClose, onOpen, onCoverChanged }) {
  const [copies, setCopies] = useState(null);
  const [cover, setCover] = useState(photo.cover || 0);
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState("");
  // Opened at this moment: the thumbnails are drawn afresh, never an old copy.
  const [stamp] = useState(() => Date.now().toString(36));

  useEffect(() => {
    let gone = false;
    fetch(`${API}/photos/versions?filepath=${encodeURIComponent(photo.filepath)}`)
      .then((r) => (r.ok ? r.json() : { copies: [], cover: 0 }))
      .then((d) => { if (!gone) { setCopies(d.copies || []); setCover(d.cover || 0); } })
      .catch(() => { if (!gone) setCopies([]); });
    return () => { gone = true; };
  }, [photo.filepath]);

  useEffect(() => {
    const key = (e) => { if (e.key === "Escape") { e.stopPropagation(); onClose(); } };
    window.addEventListener("keydown", key, true);
    return () => window.removeEventListener("keydown", key, true);
  }, [onClose]);

  const choose = useCallback(async (version) => {
    setBusy(true);
    setNote("");
    try {
      const res = await fetch(`${API}/photos/versions/cover`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filepath: photo.filepath, version }),
      });
      if (!res.ok) throw new Error("cover");
      const data = await res.json();
      setCover(version);
      const found = (data.copies || []).find((c) => c.version === version);
      onCoverChanged?.(photo.filepath, {
        cover: version,
        cover_edit: version ? (found?.settings || {}) : {},
        copies: (data.copies || []).length,
      });
    } catch {
      setNote("Could not change which version is shown");
    } finally {
      setBusy(false);
    }
  }, [photo.filepath, onCoverChanged]);

  const versions = copies === null ? [] : [
    { version: 0, name: "Main", export: true },
    ...copies.map((c) => ({ ...c, name: c.name || `Copy ${c.version}` })),
  ];

  return (
    <div className="fixed inset-0 z-[80] bg-black/80 flex items-center justify-center p-4" data-modal
      onClick={onClose} data-testid="versions-sheet">
      <div className="bg-neutral-950 border border-neutral-800 rounded-lg max-w-5xl w-full max-h-[90vh] overflow-y-auto p-4"
        onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center gap-3 mb-3">
          <h2 className="text-sm text-neutral-200 truncate">
            Versions of <span className="text-white">{photo.filename}</span>
          </h2>
          <span className="text-[11px] text-neutral-500 hidden sm:inline">
            ★ is the one shown while browsing. Every version marked for export is still written.
          </span>
          <button onClick={onClose} className="ml-auto text-neutral-400 hover:text-white text-lg leading-none"
            aria-label="Close">✕</button>
        </div>
        {copies === null ? (
          <div className="text-xs text-neutral-500 py-8 text-center">Loading versions…</div>
        ) : (
          <div className="grid gap-3" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(min(220px, 42vw), 1fr))" }}>
            {versions.map((v) => {
              const isCover = v.version === cover;
              return (
                <div key={v.version}
                  className={`rounded overflow-hidden border ${isCover ? "border-amber-400" : "border-neutral-800"}`}>
                  <button className="block w-full aspect-square bg-neutral-900" title="Open this version in Develop"
                    onClick={() => onOpen?.(photo.filepath, v.version)}>
                    <img src={versionUrl(photo.filepath, v.version, "preview", stamp)} alt={v.name}
                      loading="lazy" className="w-full h-full object-contain" />
                  </button>
                  <div className="flex items-center gap-2 px-2 py-1.5 text-[11px]">
                    <span className={`truncate ${isCover ? "text-amber-300" : "text-neutral-300"}`}>
                      {isCover && "★ "}{v.name}
                    </span>
                    {!v.export && <span className="text-neutral-600" title="Not written on export">no export</span>}
                    {isCover ? (
                      <span className="ml-auto text-amber-400/80">shown</span>
                    ) : (
                      <button disabled={busy} onClick={() => choose(v.version)} data-testid={`cover-${v.version}`}
                        className="ml-auto px-2 py-0.5 rounded border border-neutral-700 text-neutral-300 hover:text-white hover:border-amber-500 disabled:opacity-40"
                        title="Show this version for the photo in the grid, slideshows, On this day and Year in review">
                        ★ Show this
                      </button>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
        {note && <div className="mt-2 text-xs text-red-400">{note}</div>}
      </div>
    </div>
  );
}
