// ── A vectorscope: where the picture's colours fall ─────────────────────
//
// Plain functions, no React and no DOM, so Node can test them. Every pixel is
// placed by its colour alone (BT.709 Cb and Cr of the display-encoded values):
// a neutral grey of any brightness at the centre, a more saturated colour
// further out. On the scope Cb runs right and Cr runs up.

// [cb, cr] for 0..255 display-encoded values, each -1..1: a fully saturated
// primary lies within the unit square.
export function chromaOf(r, g, b) {
  const y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  const cb = (b - y) / 1.8556;
  const cr = (r - y) / 1.5748;
  return [cb / 127.5, cr / 127.5];
}

// The six colours of a 75% chart: where a colour chart's patches fall.
const LIT = 191;
export const TARGETS = [
  { label: "R", rgb: [LIT, 0, 0] },
  { label: "Mg", rgb: [LIT, 0, LIT] },
  { label: "B", rgb: [0, 0, LIT] },
  { label: "Cy", rgb: [0, LIT, LIT] },
  { label: "G", rgb: [0, LIT, 0] },
  { label: "Yl", rgb: [LIT, LIT, 0] },
].map(({ label, rgb }) => ({ label, at: chromaOf(...rgb) }));

// Degrees counter-clockwise from +Cb: the line skin of every tone falls along.
export const SKIN_ANGLE = 123;

// How much the middle is magnified.
export const GAINS = [1, 2];

// Where a chroma lands in a size x size grid at a gain, or null if it leaves
// the square.
export function cellOf(cb, cr, size, gain = 1) {
  const x = cb * gain;
  const y = cr * gain;
  if (x < -1 || x > 1 || y < -1 || y > 1) return null;
  return [Math.round(((x + 1) / 2) * (size - 1)), Math.round(((1 - y) / 2) * (size - 1))];
}

// Count an RGBA buffer (as getImageData gives) into a size x size grid.
// Transparent pixels are skipped; at a gain above 1 what leaves the square is
// dropped, not piled on the edge.
export function scopeOf(data, size = 128, gain = 1) {
  const counts = new Uint32Array(size * size);
  let peak = 0;
  for (let i = 0; i + 3 < data.length; i += 4) {
    if (data[i + 3] === 0) continue;
    const [cb, cr] = chromaOf(data[i], data[i + 1], data[i + 2]);
    const cell = cellOf(cb, cr, size, gain);
    if (!cell) continue;
    const k = cell[1] * size + cell[0];
    const n = ++counts[k];
    if (n > peak) peak = n;
  }
  return { size, gain, counts, peak };
}
