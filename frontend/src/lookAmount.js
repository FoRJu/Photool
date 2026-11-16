// How much of a look: its settings made weaker or stronger by one amount, 0..150
// with 100 the look as saved. Plain functions, so Node can test them.
//
// Strengths are scaled; choices and places are kept as they are at any amount
// (black and white, a style's name, a white balance, a toning hue, a grain's
// size, noise reduction, sharpening, lens corrections).

const SCALED = new Set([
  "exposure", "highlights", "lights", "darks", "shadows", "whites", "blacks", "compress",
  "contrast", "clarity", "vibrance", "saturation", "fade", "shadow_sat", "highlight_sat",
  "vignette", "grain", "halation", "bloom", "flat", "temperature", "tint",
]);
const SCALED_PREFIXES = ["hue_", "sat_", "lum_", "bw_"];
export const LOOK_CURVE_KEYS = ["curve", "curve_r", "curve_g", "curve_b"];

export const isScaled = (key) => SCALED.has(key) || SCALED_PREFIXES.some((p) => key.startsWith(p));

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const isIdentity = (points) => !Array.isArray(points) || points.length < 2 || points.every(([x, y]) => x === y);

function scaleCurve(points, amount) {
  const f = Math.min(amount, 100) / 100;
  return points.map(([x, y]) => [x, clamp(Math.round(x + (y - x) * f), 0, 255)]);
}

/** A look's ``values`` at ``amount`` (0..150). ``limits[key]`` is ``[min, max]``,
 * ``[-100, 100]`` when absent. A value that comes to nothing is left out. */
export function scaleLook(values, amount, limits = {}) {
  const a = clamp(Number(amount) || 0, 0, 150);
  const out = {};
  for (const [key, value] of Object.entries(values || {})) {
    let next = value;
    if (a === 100) {
      next = value;
    } else if (LOOK_CURVE_KEYS.includes(key)) {
      next = Array.isArray(value) ? scaleCurve(value, a) : value;
    } else if (key === "style_fade") {
      continue;   // below, with the style
    } else if (isScaled(key) && typeof value === "number") {
      const [lo, hi] = limits[key] || [-100, 100];
      const raw = clamp((value * a) / 100, lo, hi);
      next = key === "exposure" ? Math.round(raw * 20) / 20 : Math.round(raw);
      if (Object.is(next, -0)) next = 0;
    }
    if (LOOK_CURVE_KEYS.includes(key) ? isIdentity(next) : next === 0 || next == null || next === "") continue;
    out[key] = next;
  }
  if (a !== 100 && values?.style) {
    const fade = Number(values.style_fade) || 0;
    const next = Math.round(100 - ((100 - fade) * Math.min(a, 100)) / 100);
    if (next !== 0) out.style_fade = next;
  } else if (a !== 100 && "style_fade" in (values || {}) && Number(values.style_fade) !== 0) {
    out.style_fade = values.style_fade;   // no style: left alone
  }
  return out;
}

function sameValue(key, a, b) {
  if (LOOK_CURVE_KEYS.includes(key)) {
    if (isIdentity(a) && isIdentity(b)) return true;
    return JSON.stringify(a) === JSON.stringify(b);
  }
  if (typeof a === "string" || typeof b === "string") return (a || "") === (b || "");
  return Math.abs((Number(a) || 0) - (Number(b) || 0)) < 1e-9;
}

/** Whether ``settings`` still read, on every key the look owns, what the look
 * at ``amount`` gives over ``zero``. */
export function lookIsLive(settings, keys, values, amount, zero = {}, limits = {}) {
  const scaled = scaleLook(values, amount, limits);
  return keys.every((k) => sameValue(k, settings?.[k], k in scaled ? scaled[k] : zero[k]));
}

// A mask's controls and their ranges (backend adjust/masks.py). Its kind, its
// place (a range, a depth band, the areas of people it names) and whether it is
// inverted are choices, kept at any amount.
const MASK_LIMITS = {
  exposure: [-4, 4], saturation: [-200, 200], blur: [0, 100], smooth: [0, 100], even: [0, 100],
  contrast: [-100, 100], highlights: [-100, 100], lights: [-100, 100], darks: [-100, 100],
  shadows: [-100, 100], temperature: [-100, 100], clarity: [-100, 100], texture: [-100, 100],
};
export const MASK_CONTROL_KEYS = Object.keys(MASK_LIMITS);

/** A look's ``masks`` at ``amount`` (0..150): every control scaled and clamped
 * to its range, exposure in 0.05 steps. A control that comes to nothing is left
 * out; the mask itself is kept, so it stays in the list to be turned up again. */
export function scaleMasks(masks, amount) {
  const a = clamp(Number(amount) || 0, 0, 150);
  return (masks || []).map((m) => {
    if (a === 100) return { ...m };
    const out = { ...m };
    for (const [key, [lo, hi]] of Object.entries(MASK_LIMITS)) {
      if (typeof m[key] !== "number") continue;
      const raw = clamp((m[key] * a) / 100, lo, hi);
      let next = key === "exposure" ? Math.round(raw * 20) / 20 : Math.round(raw);
      if (Object.is(next, -0)) next = 0;
      if (next === 0) delete out[key]; else out[key] = next;
    }
    return out;
  });
}

/** Whether the masks put on by the look named ``name`` are still the look's
 * ``masks`` at ``amount``: as many, of the same kinds, each control as given. */
export function lookMasksLive(masks, name, lookMasks, amount) {
  const mine = (masks || []).filter((m) => m.look === name);
  const want = scaleMasks(lookMasks, amount);
  if (mine.length !== want.length) return false;
  return mine.every((m, i) => m.kind === want[i].kind && !!m.invert === !!want[i].invert
    && MASK_CONTROL_KEYS.every((k) => Math.abs((Number(m[k]) || 0) - (Number(want[i][k]) || 0)) < 1e-9));
}
