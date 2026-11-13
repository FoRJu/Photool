# Brief: a vectorscope beside the histogram

One session, one branch (`vectorscope`), one PR. Frontend only: no backend
change, no new package. Read only what this brief names. Do not read
`frontend/src/DevelopView.jsx` whole (6,400 lines): grep the anchors given.

## Why

The histogram says how bright the picture is, not what colour it is. With white
balance now in Kelvin and tint (`frontend/src/WhiteBalance.jsx`), a person needs
to see a cast: whether the greys sit in the middle, which way the picture
leans, and whether skin falls where skin does. A vectorscope shows exactly
that, and it is a measurement of the photograph, not a change to it.

## What changes for the person

- Above the histogram, two small buttons: **Histogram** and **Scope**. The
  choice is kept on this device.
- **Scope** draws a round vectorscope of the edited picture: every pixel placed
  by its colour, neutral in the centre, more saturated further out. Six small
  boxes mark red, magenta, blue, cyan, green and yellow, and a thin line marks
  where skin falls.
- A **2×** button magnifies the middle, for judging a white balance.
- Pointing at the picture rings that pixel's colour on the scope.
- The pixel readout under it, and everything else in the panel, is unchanged.

## The maths

New file `frontend/src/vectorscope.js`: plain functions, no React, no DOM, so
Node can test it.

- `chromaOf(r, g, b)` for display-encoded 0..255 values, BT.709:
  - `y = 0.2126 r + 0.7152 g + 0.0722 b`
  - `cb = (b - y) / 1.8556`, `cr = (r - y) / 1.5748`, each in -127.5..127.5
  - returned as `[cb / 127.5, cr / 127.5]`, so each is -1..1 and a fully
    saturated primary lies within the unit square.
- On the scope `cb` runs right and `cr` runs **up** (canvas y is flipped).
- `TARGETS`: the six colours at 75% (`191` for a lit channel, `0` otherwise),
  each `{ label: "R" | "Mg" | "B" | "Cy" | "G" | "Yl", at: chromaOf(...) }`.
  Work them out with `chromaOf`; do not type the coordinates in.
- `SKIN_ANGLE = 123` degrees, counter-clockwise from the `+cb` axis: the line
  from the centre that skin of every tone falls along.
- `scopeOf(data, size = 128)`: `data` is an RGBA `Uint8ClampedArray` (as
  `getImageData` gives). Returns `{ size, counts: Uint32Array(size * size),
  peak }`. A pixel goes to column `round((cb + 1) / 2 * (size - 1))` and row
  `round((1 - cr) / 2 * (size - 1))`. Skip pixels whose alpha is 0.
- `GAINS = [1, 2]`: at gain 2 the chroma is doubled before it is placed, and
  what leaves the square is dropped (not piled on the edge).
  `scopeOf(data, size, gain)`.

## The component

New file `frontend/src/Vectorscope.jsx`, default export
`Vectorscope({ scope, gain, probe, className })`:

- A `<canvas>` of `scope.size` squared, `data-testid="vectorscope"`, stretched
  to a square box (`aspect-square`, at most about 220 px wide, centred), on
  `bg-neutral-950 rounded border border-neutral-800` like the histogram.
- Each cell is painted in **its own colour**: the colour that sits at that
  `cb`, `cr` with a mid brightness. Its strength follows the count on a log
  scale (`log(1 + count) / log(1 + peak)`), so a few saturated pixels still
  show beside a large grey mass. Empty cells stay black.
- Over it, an SVG (as `Histogram` uses) with: a circle at the edge, a small
  cross at the centre, the six target boxes with their one or two letters, and
  the skin line. Thin, `#525252`-ish, `vectorEffect="non-scaling-stroke"`. At
  gain 2 the targets and the circle move outward with the data (those that
  leave the box are not drawn); the skin line and the cross stay.
- `probe` is `[r, g, b]` or null: a small white ring at that colour's place.
- It draws nothing itself from a bitmap: `scope` is handed in.
- Export a helper `scopeOfBitmap(bitmap, gain)`: draws the bitmap 256 px wide
  to a canvas, reads it, and calls `scopeOf`. Copy the first eight lines of
  `histogramOf` in `DevelopView.jsx` (grep `function histogramOf`) for how.

## Wiring in `DevelopView.jsx`

Follow how the clipping map is done; it is the same shape of problem (a thing
worked out from each preview's bitmap, only while it is shown).

- grep `showClipRef` (3 places, ~3653 and ~3925) and `lastBitmap` (~3648): add
  `scopeMode` state (`"histogram" | "scope"`, starting from
  `recall("develop.scope")`, see `frontend/src/remembered.js`), a ref that
  mirrors it, and `scopeGain`. Where each preview arrives (grep
  `const stats = histogramOf(result.bitmap)`, ~3921), also set the scope when
  the mode is `"scope"`. When the mode or the gain changes, work it out once
  from `lastBitmap.current`.
- The section header: grep `uppercase tracking-wide text-neutral-500">Histogram`
  (~6173). The word becomes the two buttons (`data-testid="scope-histogram"`,
  `data-testid="scope-vectorscope"`), with the **2×** button
  (`data-testid="scope-gain"`) shown only in scope mode. The legend on the
  right is the histogram's own: in scope mode it reads `neutral in the centre ·
  line: skin`.
- Below it, `<Histogram …>` or `<Vectorscope …>`. The probe is already in
  state: grep `setProbe(` (~4068); pass `probe?.rgb`.
- The compact top bar (grep `className="h-7"`, ~6090) keeps the histogram.
- On a short screen (grep `const short = useShortScreen()`), the scope is not
  offered: the buttons are not drawn and the histogram shows as now.

Do not move or restructure anything else in the file.

## Tests

No Python changes, so no Python tests to run.

1. New `frontend/tests/vectorscope.test.mjs`, run with
   `node --test frontend/tests/` (Node's own runner; add nothing to
   `package.json` but a `"test": "node --test tests/"` script):
   - a grey of any brightness is at `[0, 0]`
   - pure red, green and blue each lie inside the unit square, and red's `cr`
     is positive, blue's `cb` is positive
   - the six targets go round in the order R, Mg, B, Cy, G, Yl by angle
     (clockwise on the scope as drawn)
   - `SKIN_ANGLE` lies between the angles of Yl and R
   - a typical skin colour, `(224, 172, 150)`, is within 15 degrees of the
     skin line
   - `scopeOf` on a buffer of one grey and one red pixel: two cells counted,
     the grey at the centre cell; `peak` is 1; a transparent pixel is skipped
   - at gain 2 a colour's distance from the centre cell doubles, and a
     saturated red is dropped
2. `npm run build` in `frontend` succeeds.

## By eye, in Chrome

Say in the PR which of these were done:

- a grey-card or black-and-white frame is a dot in the centre
- dragging Temp moves the cloud along blue to amber, Tint across it
- a portrait's skin lies along the line
- the six boxes are where a colour chart's patches fall
- the ring follows the pointer
- the choice survives a reload
- 390 px wide: the scope fits the panel and the buttons are at least 32 px high

## Out of scope

The original's scope beside the edit's, a waveform or RGB parade, a scope in
the compact top bar, anything in the backend, any refactor of
`DevelopView.jsx`.

## When done

Say in the PR what was not verified by a test and was checked by eye instead,
and anything in this brief that turned out wrong.
