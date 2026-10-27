// ── Where a slideshow picture pulls back from ──────────────────────
//
// The slideshow's "Pull back" opens each picture from its main person: the
// biggest face the catalog knows for that photograph. That framing is held for
// a moment, then a calm zoom out settles on the whole scene.
//
// The opening zoom is set by how small that person is in the picture. A face
// that already fills the frame barely moves — there is little scene to reveal —
// while someone in a wide shot opens wide enough to show the scene around them.
// "Calm" takes three quarters of that opening zoom, so the move is gentle.
//
// The faces are the catalog's, as `GET /api/faces/photo/{filepath}` gives them:
// box fractions of the finished picture, `shown` when the picture is cropped or
// levelled (`shown` None means the crop left the face out). Without a face on
// the picture the pull comes gently from the middle of the scene.

const MIN_SIZE = 0.004;                       // smaller than this is a speck, not a face
const OPEN = 0.30;                            // the main face takes about this share of the frame
const FROM_MIN = 1.15;                        // a face already filling the frame hardly moves
const FROM_MAX = 2.2;                         // a small face opens wide enough to show the scene
const CALM = 0.75;                            // Calm: three quarters of that opening zoom
const MIDDLE = 1.35;                          // no face: a gentle pull from the middle
const FALLBACK = { x: 0.5, y: 0.5, from: 1 + (MIDDLE - 1) * CALM };

const clamp = (v, low, high) => Math.min(high, Math.max(low, v));

// One face's box on the picture being shown, or null when it is not on it.
function boxOf(face) {
  const box = face && (face.shown === undefined ? face : face.shown);
  if (!box || !(box.w > MIN_SIZE) || !(box.h > MIN_SIZE)) return null;
  if (![box.x, box.y, box.w, box.h].every((v) => typeof v === "number" && Number.isFinite(v))) return null;
  return { x: box.x, y: box.y, w: box.w, h: box.h };
}

/** The point a slide opens from and the scale it opens at: {x, y, from} in
 * picture fractions (x, y) and a calm zoom that eases to 1. The middle of the
 * picture, gently zoomed, when there is no face. */
export function pullFocus(faces) {
  const boxes = (faces || []).map(boxOf).filter(Boolean);
  if (!boxes.length) return { ...FALLBACK };
  // The main person: the biggest face on the picture.
  const main = boxes.reduce((a, b) => (b.w * b.h > a.w * a.h ? b : a));
  const size = Math.max(main.w, main.h);
  const open = clamp(OPEN / Math.max(0.12, size), FROM_MIN, FROM_MAX);
  return {
    x: clamp(main.x + main.w / 2, 0.05, 0.95),
    y: clamp(main.y + main.h / 2, 0.05, 0.95),
    // Four places, so the CSS value the app writes is the round one.
    from: Math.round((1 + (open - 1) * CALM) * 1e4) / 1e4,
  };
}
