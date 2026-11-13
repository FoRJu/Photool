# Brief: white balance in Kelvin and tint, with a grey picker

One session, one branch (`white-balance-kelvin`), one PR. Backend first, with
tests; then the panel. Read only what this brief names. Do not read
`frontend/src/DevelopView.jsx` whole (6,400 lines): grep the anchors given.

## Why

Warmth today is a ±25% red/blue gain and Tint a ±15% green gain on a decode the
camera has already balanced (`adjust/develop.py: adjustment`). There is no
Kelvin, no "as shot", no way to click something neutral. The tooltip itself
says it is not a colour temperature. People cannot tell what the sliders mean
or where correct is.

## What changes for the person

- RAW: a **Temp** slider in Kelvin starting at the camera's as-shot value and a
  **Tint** slider on Lightroom's scale, an **As shot** readout, presets, and a
  **grey picker**: click something that should be neutral and both sliders are
  solved.
- JPEG/HEIC/TIFF: the same controls shown relative (-100..100), since the
  file's own Kelvin is not known; the picker works the same.
- Every existing edit renders exactly as it did.

## Settings (the edit model)

Two new keys, absolute:

| key | range | meaning |
|---|---|---|
| `wb_kelvin` | 2000..50000, int | absent = as shot. Present = custom white balance |
| `wb_tint` | -150..150, int | read only when `wb_kelvin` is present; absent then means 0 |

- `normalize` drops `wb_tint` when there is no `wb_kelvin`. The panel always
  writes both together.
- A custom value within 1 mired and 1 tint of as shot is the identity (no pass
  over the pixels).
- Higher Kelvin makes the picture warmer, positive tint more magenta
  (Lightroom's convention).
- The old `temperature` and `tint` keys stay, unchanged, everywhere: masks,
  set matching, `paste_match`, old sidecars. They are a relative trim applied
  after the new white balance.

## Colour maths

New module `backend/adjust/whitebalance.py`. Nothing else holds this maths.

1. **Kelvin, tint ↔ xy.** Port the DNG SDK's `dng_temperature` (Robertson's
   isotherm table in CIE 1960 uv, tint = offset from the locus × -3000). This
   gives Lightroom's numbers. `xy_of(kelvin, tint)` and `kelvin_tint_of(x, y)`.
2. **What a file knows** — `of_source(path) -> WhiteBalance | None`, cached by
   path and mtime, reading metadata only (`rawpy.imread`, no `postprocess`):
   - `cam_xyz`: `raw.rgb_xyz_matrix[:3]` (XYZ to camera)
   - `gains`: `raw.camera_whitebalance[:3]`
   - `srgb_from_cam`: `export.engine._srgb_from_camera(raw)`
   - as-shot neutral `n0 = 1 / gains` (green = 1); as-shot xy from
     `inv(cam_xyz) @ n0`; so `as_shot = (kelvin, tint)`.
   - Any of these missing, or not a RAW: the **nominal** case below.
3. **The re-balance, RAW.** The frame's pixels are
   `srgb_from_cam @ (balanced camera values)`. For a chosen white, its camera
   neutral is `n1 = cam_xyz @ XYZ(xy)` (green = 1). Then
   `W = S · diag(n0 / n1) · S⁻¹`, with `S = srgb_from_cam`. This is the same
   picture a fresh decode at the new white balance would give.
4. **The re-balance, nominal** (no matrix, or not RAW). As shot is taken to be
   D65. `W = M_xyz→rgb · Bradford(white chosen → D65) · M_rgb→xyz` in linear
   sRGB.
5. **The picker.** Given a linear sample `p` of the frame as decoded (before
   any edit): RAW: `n1 ∝ n0 * (S⁻¹ @ p)`, then xy from `inv(cam_xyz) @ n1`,
   then Kelvin and tint. Nominal: xy of `M_rgb→xyz @ p`. Refuse a sample with a
   channel clipped or the patch near black.

## Backend work

- `adjust/whitebalance.py`: the above, plus `matrix_for(wb, kelvin, tint)`.
- `export/engine.py: frame_of` (line ~1971): attach it lazily, as `frame.lens`
  is: `frame.wb = lambda: whitebalance.of_source(source)`. Add the field to
  `LinearFrame` (`adjust/render.py` ~340). Frames cut from another frame carry
  `anchor_of`; read the white balance from `anchor_of or frame`. Grep
  `LinearFrame(` (11 places) and make sure none loses it.
- `adjust/render.py: _light_and_tone` (~1108): when `wb_kelvin` is in the edit,
  `rgb = clip(frame.rgb @ W.T, 0)` before the existing `* gains`. Add both keys
  to `_LIGHT_AND_TONE_KEYS` (~1044) so the cache sees them.
- `adjust/develop.py`: the two keys in the slider tuples and `normalize`
  (~431), with the pairing rule. Sidecar (~879–1060): write and read our own
  tags following the `crs:TemperatureShift` pattern (`crs:WBKelvin`,
  `crs:WBTint`). Do not write Lightroom's `crs:Temperature` in this PR.
- `api/routes.py`: fields on the adjustment model (~139). Two routes, modelled
  on `develop_depth_at` (~3488):
  - `POST /export/develop/white-balance` `{filepath}` →
    `{raw: bool, as_shot: {kelvin, tint}}`
  - `POST /export/develop/white-point` `{filepath, u, v, settings, size}` →
    `{kelvin, tint}`, median of a small patch of `frame.rgb`, the point taken
    through `adjust.geometry.source_point`; 422 with a sentence when refused.
- `export/paste_match.py`:
  - `LIGHT_KEYS` (line 41) gains both keys, so *keep light* keeps a frame's own.
  - *Match exposure*: the anchors already measure the frames' as-shot
    renderings against each other, so pasting an absolute Kelvin on top would
    correct twice. Give the target its own as shot moved by the copied frame's
    difference from *its* as shot (in mireds, and in tint).
  - *Copied*: the absolute values, as pasted.

## Panel work

New file `frontend/src/WhiteBalance.jsx`; `DevelopView.jsx` only mounts it and
lends it the picker. Anchors in `DevelopView.jsx`:

- the section definition: grep `id: "wb"` (line ~79) and `{ id: "wb", label`
  (~475, the Copy checklist: its keys become all four)
- the picker pattern to copy: grep `pickingFocus` and `focusPick.current`
  (~4701, ~5612, ~5644)
- `function Slider` (~2676) for the control's look

The component:

- Asks `/export/develop/white-balance` once per file.
- **Temp**: positioned evenly in mireds, shown in Kelvin, track painted blue to
  amber. **Tint**: track painted green to magenta. Untouched, both sit at as
  shot and are not marked as changed. Double-click or double-tap: back to as
  shot (removes both keys).
- Not RAW: Temp shown as -100..100 (1.2 mired a step about D65), Tint as
  -100..100. Stored as Kelvin and tint all the same.
- A line under the sliders: `As shot 5200 K · +8`, and the presets as small
  buttons — As shot; and for RAW Daylight 5500/+10, Cloudy 6500/+10, Shade
  7500/+10, Tungsten 2850/0, Fluorescent 3800/+21, Flash 5500/0.
- **Pick grey** button, `data-testid="wb-pick"`: the next click or tap on the
  picture calls `/white-point` and sets both. A refusal shows its sentence.
- An edit that carries old `temperature`/`tint`: one line, `Earlier warmth +12,
  tint -3 · Clear`. The old sliders are no longer shown here (the mask's own
  Temp slider is untouched).
- Works at phone width: buttons wrap, targets at least 32 px high.

No Auto white balance in this PR.

## Tests

New `tests/test_white_balance.py`; run it, `tests/test_adjust.py`,
`tests/test_render.py`, `tests/test_paste_match.py`,
`tests/test_develop_routes.py` and `tests/test_develop_view.py`, not the suite.
How to run: `docs/setup.md` line 219.

1. Kelvin, tint → xy → back, 2000..25000 K and tint ±100: within 1 mired and
   0.5 tint. Illuminant A (0.4476, 0.4074) reads 2856 ± 15 K, tint within ±3.
2. Custom equal to as shot: the render is identical, bit for bit, to no keys.
3. With a real `cam_xyz` hard-coded in the test: a grey lit by T1 and balanced
   for T0, then set to T1, comes out with R = G = B within 0.5%.
4. The picker's answer, applied, makes the picked patch neutral within 1%,
   RAW and nominal. (First written as 0.5%; on real frames a whole-number tint
   leaves up to 1%, and whole numbers are kept.)
5. Direction: higher Kelvin raises a grey's R/B; positive tint lowers its G.
6. An edit with only `temperature`/`tint` renders identically to `main`.
7. `normalize`: ranges clamped, `wb_tint` alone dropped.
8. Sidecar round trip of both keys.
9. Paste: *keep light* keeps the target's keys; *match exposure* moves the
   target's as shot by the copied difference.

## Out of scope

Auto white balance, Lightroom's own `crs:Temperature`, masks in Kelvin, the
vectorscope (next brief), any refactor of `DevelopView.jsx`.

## When done

Say in the PR what was not verified by a test and was checked by eye instead
(the slider tracks, the picker on a phone), and anything in this brief that
turned out wrong.
