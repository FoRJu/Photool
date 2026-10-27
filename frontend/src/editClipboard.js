// ── The copied edit ─────────────────────────────────────────────────
//
// Copied in the develop view, pasted there frame by frame or in the grid onto a
// whole selection. It lives here, outside both, because the develop view is
// created afresh for every frame (it is keyed on the file, so a frame that fails
// does not strand the next one): a clipboard kept inside it would be emptied by
// the very act of moving to the frame it was copied for.
//
// What is held: { settings, keys, source, landscape, sections }. ``settings`` is
// the frame's whole edit; ``keys`` are the parts of it that a paste carries (the
// Copy checklist's sections), the rest of each target's edit staying its own.
// ``landscape`` is the shape of the frame it came from, so a crop is not put on
// a frame turned the other way. ``sections`` are the names, for saying what a
// paste will do.

import { useEffect, useState } from "react";

let copied = null;
const listeners = new Set();

export function copiedEdit() {
  return copied;
}

export function copyEdit(edit) {
  copied = edit;
  for (const listener of listeners) listener(copied);
}

// The copied edit, kept up to date in a component that shows it.
export function useCopiedEdit() {
  const [value, setValue] = useState(copied);
  useEffect(() => {
    listeners.add(setValue);
    setValue(copied);
    return () => { listeners.delete(setValue); };
  }, []);
  return value;
}

// Which of the Copy checklist's sections are ticked, kept on this device: a
// tripod session ticks Level once and every copy after carries it.
const CHOICE_KEY = "photool.develop.copy-sections";

export function copyChoice(fallback) {
  try {
    const kept = JSON.parse(localStorage.getItem(CHOICE_KEY) || "null");
    if (Array.isArray(kept)) return kept;
  } catch { /* nothing kept, or storage is off */ }
  return fallback;
}

export function keepCopyChoice(ids) {
  try { localStorage.setItem(CHOICE_KEY, JSON.stringify(ids)); } catch { /* this visit only */ }
}

// ── How a paste adjusts each frame ───────────────────────────────────
//
// Copy decides what is carried; this decides what happens to each frame it is
// put on (backend export/paste_match.py). Kept on this device and shared by the
// develop view's Paste, To burst and the grid's Paste edit, so choosing it in
// one place is choosing it everywhere.
export const PASTE_AS = [
  { id: "copied", label: "As copied", short: "Paste as copied",
    title: "The copied edit exactly as it is: for frames shot in the same light at the same exposure" },
  { id: "keep_light", label: "Keep each frame's exposure & white balance", short: "Keep own exposure & WB",
    title: "Everything else as copied: for frames whose exposure and white balance are already right" },
  { id: "match_exposure", label: "Match light to the copied frame", short: "Match light",
    title: "Each frame's exposure and white balance brought to the copied frame's: by its capture settings while the light stayed the same, checked against the people and things both frames show (same faces, the backdrop, instruments), at any angle or focal length. Where those say the light itself changed, they set it, and the note says so" },
  { id: "match_look", label: "Match the look to the copied frame", short: "Match the look",
    title: "Each frame's tone and colour matched to the copied frame as developed. Same shoot only: Match copies a picture's content as well as its look, so a frame of another scene is pushed towards that scene" },
];
const PASTE_AS_KEY = "photool.develop.paste-as";
const pasteListeners = new Set();

export function pasteAs() {
  try {
    const kept = localStorage.getItem(PASTE_AS_KEY);
    if (PASTE_AS.some((p) => p.id === kept)) return kept;
  } catch { /* nothing kept, or storage is off */ }
  return "copied";
}

export function keepPasteAs(id) {
  try { localStorage.setItem(PASTE_AS_KEY, id); } catch { /* this visit only */ }
  for (const listener of pasteListeners) listener(id);
}

// The paste choice, kept up to date in a component that shows it.
export function usePasteAs() {
  const [value, setValue] = useState(pasteAs);
  useEffect(() => {
    pasteListeners.add(setValue);
    setValue(pasteAs());
    return () => { pasteListeners.delete(setValue); };
  }, []);
  return value;
}

// What a paste onto many did to its frames, in a few words, from the backend's
// notes ({filepath: note}).
// The frames a paste corrected because the light itself changed: a lighting cue
// may be a mood worth keeping, so these are the ones to look at.
export function lightChanged(notes) {
  return Object.entries(notes || {})
    .filter(([, n]) => !n.startsWith("light left as pasted") && n.includes("the light changed"))
    .map(([fp]) => fp);
}
export function pasteSummary(mode, notes) {
  const all = Object.values(notes || {});
  if (mode === "keep_light") return "each kept its own exposure and white balance";
  if (mode === "match_exposure") {
    const left = all.filter((n) => n.startsWith("light left as pasted"));
    const moved = all.length - left.length;
    const changed = all.filter((n) => !n.startsWith("light left as pasted") && n.includes("the light changed")).length;
    const parts = [`light matched on ${moved}`];
    if (changed) parts.push(`${changed} where the light itself changed: check those keep the mood you want`);
    if (left.length) parts.push(`${left.length} left as pasted`);
    return parts.join(" · ");
  }
  if (mode === "match_look") {
    const failed = all.filter((n) => n.startsWith("look not matched")).length;
    const elsewhere = all.filter((n) => n.includes("another shoot")).length;
    const parts = [`look matched on ${all.length - failed}`];
    if (elsewhere) parts.push(`${elsewhere} from another shoot, check those`);
    if (failed) parts.push(`${failed} could not be read`);
    return parts.join(" · ");
  }
  return "";
}
