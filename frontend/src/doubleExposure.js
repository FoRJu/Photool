// A double exposure is a copy of one photograph with another laid over it, so the
// photograph's own edit stays as it was. The grid makes one from two selected photographs
// (DoubleExposureDialog.jsx), and the develop view makes the same pair the other way round.
//
// The second comes as shot (no version: its raw light, toned by this photo's edit) or as
// one of its versions, finished by that version's own edit, of which it keeps a copy
// (backend adjust/double.py). A second chosen without saying which is its cover: the
// version the grid shows for it.
const API = "/api";

async function asked(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || "The server would not do that");
  }
  return res.json();
}

// A thumbnail of one version of a photo; ``key`` changes when that version's edit does.
export function versionThumb(filepath, version, size = "grid", key = "") {
  const cut = filepath.lastIndexOf("/");
  return `${API}/thumb/${filepath.slice(0, cut)}/${encodeURIComponent(filepath.slice(cut + 1))}`
    + `?size=${size}&ver=${version}${key ? `&v=${encodeURIComponent(key)}` : ""}`;
}

// The same text for the same edit, however it was written: keys in order, and true, 1 and
// "1" alike, as the server stores a switch as 1 and a slider may hold its number as text.
function stable(value) {
  if (Array.isArray(value)) return `[${value.map(stable).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.keys(value).sort().map((k) => `${JSON.stringify(k)}:${stable(value[k])}`).join(",")}}`;
  }
  if (typeof value === "boolean") return value ? "1" : "0";
  if (typeof value === "string" && value.trim() !== "" && Number.isFinite(Number(value))) return String(Number(value));
  return JSON.stringify(value);
}

function withoutSecond(settings) {
  const { second, ...rest } = settings || {};
  return rest;
}

// A short stable key for an edit, for a thumbnail's address.
export function editKey(settings) {
  const text = stable(withoutSecond(settings));
  let h = 5381;
  for (let i = 0; i < text.length; i += 1) h = ((h * 33) ^ text.charCodeAt(i)) >>> 0;
  return h.toString(36);
}

// Whether two edits are the same, a second exposure of their own aside.
export function sameEdit(a, b) {
  return stable(withoutSecond(a)) === stable(withoutSecond(b));
}

// Every version of a photo, its own edit first, as [{version, name, settings}], and its cover.
export async function versionsOf(filepath) {
  const [own, listed] = await Promise.all([
    asked(`${API}/photos/edit?filepath=${encodeURIComponent(filepath)}`),
    asked(`${API}/photos/versions?filepath=${encodeURIComponent(filepath)}`),
  ]);
  return {
    cover: listed.cover || 0,
    versions: [
      { version: 0, name: "Main", settings: own.settings || {} },
      ...(listed.copies || []).map((c) => ({ version: c.version, name: c.name || `Copy ${c.version}`, settings: c.settings || {} })),
    ],
  };
}

// Which photo a second is and which of its versions; ``version`` null is the photo as shot.
export function secondOf(path, version = null, settings = {}) {
  return version === null || version === undefined
    ? { path }
    : { path, version, edit: withoutSecond(settings) };
}

// The same from a grid row, which knows its cover and that cover's edit.
export function secondOfRow(photo) {
  const cover = photo.cover || 0;
  return secondOf(photo.filepath, cover, cover ? photo.cover_edit : photo.edit);
}

// A new copy of ``base``'s version ``fromVersion`` with ``second`` laid over it
// ({path, version?, edit?} and how it is laid on: backend adjust/double.py). The copy's
// version number.
export async function makeDoubleExposure(base, fromVersion, second) {
  const json = { "Content-Type": "application/json" };
  const made = await asked(`${API}/photos/versions`, {
    method: "POST", headers: json,
    body: JSON.stringify({ filepath: base, from_version: fromVersion || 0, name: "Double exposure" }),
  });
  const edit = await asked(`${API}/photos/edit?filepath=${encodeURIComponent(base)}&version=${made.version}`);
  await asked(`${API}/photos/edit`, {
    method: "PUT", headers: json,
    body: JSON.stringify({ filepath: base, version: made.version, ...edit.settings, second }),
  });
  return made.version;
}
