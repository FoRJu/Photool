// The folder view's Look menu: one of your looks on every selected photo, or
// each camera's own look on the photos of that camera. Nothing is put on until
// a look is chosen here.
import { useCallback, useEffect, useState } from "react";

const API = "/api";
export const EACH_CAMERAS = "__each_camera__";

export default function LookOnSelection({ picked, disabled, onLook }) {
  const [presets, setPresets] = useState([]);
  const load = useCallback(async () => {
    try {
      const res = await fetch(`${API}/export/develop/presets`);
      if (res.ok) setPresets((await res.json()).presets || []);
    } catch { /* the menu is a convenience */ }
  }, []);
  useEffect(() => { load(); }, [load]);
  if (!presets.length) return null;

  const cameras = new Set(presets.map((p) => p.camera).filter(Boolean));
  const offerEach = (picked || []).some((p) => p.camera && cameras.has(p.camera));
  return (
    <select value="" disabled={disabled} onFocus={load} data-testid="look-on-selection"
      onChange={(e) => { const choice = e.target.value; if (choice) onLook(choice, presets); }}
      title="Put one of your looks on the selected photos: it sets the sections the look carries, and each photo keeps the rest of its edit"
      className="bg-neutral-800 border border-neutral-700 rounded px-1 py-1 text-xs text-neutral-300 disabled:opacity-40">
      <option value="">Look…</option>
      {offerEach && <option value={EACH_CAMERAS}>★ Each camera's own look</option>}
      {presets.map((p) => <option key={p.name} value={p.name}>{p.name}</option>)}
    </select>
  );
}
