// A double exposure is a copy of one photograph with another laid over it, so the
// photograph's own edit stays as it was. The grid makes one from two selected photographs,
// and the develop view makes the same pair the other way round.
const API = "/api";

async function asked(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || "The server would not do that");
  }
  return res.json();
}

// A new copy of ``base`` from its own edit, with ``other`` as its second exposure. ``how`` is
// anything else about the second (backend adjust/double.py). The copy's version number.
export async function makeDoubleExposure(base, other, how = {}) {
  const json = { "Content-Type": "application/json" };
  const made = await asked(`${API}/photos/versions`, {
    method: "POST", headers: json,
    body: JSON.stringify({ filepath: base, from_version: 0, name: "Double exposure" }),
  });
  const edit = await asked(`${API}/photos/edit?filepath=${encodeURIComponent(base)}&version=${made.version}`);
  await asked(`${API}/photos/edit`, {
    method: "PUT", headers: json,
    body: JSON.stringify({ filepath: base, version: made.version, ...edit.settings, second: { ...how, path: other } }),
  });
  return made.version;
}
