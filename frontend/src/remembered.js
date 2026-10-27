// ── What the Library's front page last showed, kept on this device ─────
//
// The folder list, the cards, On this day and the years are worked out from the
// whole library, a second or more each, and the page asks for all of them every
// time it is shown: after every folder, over Wi-Fi from a phone. They change
// rarely, so the page draws what it showed last at once and brings it up to
// date when the answers arrive.
//
// Kept in this tab's memory and in localStorage, so a reload draws at once too.
// It holds titles, places and names, never a picture: the pictures stay behind
// the pairing gate, in the browser's own cache. A device that is no longer
// paired forgets it all (forgetAll, from the pairing screen).

const PREFIX = "photool.remembered.";
const memory = new Map();

export function recall(key) {
  if (memory.has(key)) return memory.get(key);
  try {
    const text = localStorage.getItem(PREFIX + key);
    if (text) {
      const value = JSON.parse(text);
      memory.set(key, value);
      return value;
    }
  } catch { /* nothing kept, or storage is off */ }
  return undefined;
}

export function remember(key, value) {
  memory.set(key, value);
  try { localStorage.setItem(PREFIX + key, JSON.stringify(value)); } catch { /* kept for this tab only */ }
}

export function forgetAll() {
  memory.clear();
  try {
    for (const key of Object.keys(localStorage)) if (key.startsWith(PREFIX)) localStorage.removeItem(key);
  } catch { /* nothing to forget */ }
}
