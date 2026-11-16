# Brief: the mask controls that are missing

One session, one branch (`mask-controls`, from `main` once brief 1 has merged,
or from its branch if it has not: the PR is still opened against `main`), one
PR. Backend first, with tests; then the panel. Do not read
`frontend/src/DevelopView.jsx` whole: grep the anchors given.

Second of three briefs from the Develop review of 2026-11-16.

## Why

A mask has exposure, contrast, highlights, lights, darks, shadows, temperature,
saturation, clarity, blur and the three skin controls. With object, people-part
and depth masks able to reach almost anything, what a mask can *do* there is now
the limit:

1. **No tint.** Temperature alone cannot fix mixed light: a green cast from
   fluorescent tubes or a magenta stage wash on a face needs the other axis.
2. **Temperature stops short.** ±100 is a 25% gain on red and blue
   (`MAX_WB_GAIN`, `adjust/develop.py` ~80). A face lit by a window in a
   tungsten room is out by about a factor of two. The global control has
   `wb_kelvin` for that; a mask has nothing.
3. **No whites or blacks.** The global tone curve has them; `_MASK_TONE_KEYS`
   (`adjust/render.py` ~1065) leaves them out, so a mask cannot set a clean
   white on a shirt or a true black on a backdrop.
4. **No hue.** A colour-range or object mask can find a shirt but cannot change
   its colour, only its saturation.
5. **No sharpness.** Eyes and a product's label want more, skin less, and
   sharpening is global only.
6. **No amount.** A mask's whole effect cannot be eased with one control; each
   slider must be moved.

## What changes for the person

In a mask's controls (`MASK_SLIDERS`, `DevelopView.jsx` ~1038; backend
`adjust/masks.py` ~93):

| control | range | what it does |
| --- | --- | --- |
| Temp | -200..200 (was ±100) | ±100 as today; ±200 is twice the stops again |
| Tint | -200..200, new | green to magenta, as the global Tint; ±100 the global's reach |
| Whites | -100..100, new | as the global Whites, where the mask reaches |
| Blacks | -100..100, new | as the global Blacks |
| Hue | -180..180 degrees, new | turns the colours it reaches round the wheel |
| Sharpness | -100..100, new | more or less of the edit's sharpening there; +100 over none is the global at 100 |
| Amount | 0..200, default 100, new | every control of this mask at that share, shown first |

Order in the panel: Amount; Exposure, Contrast, Highlights, Lights, Darks,
Shadows, Whites, Blacks; Temp, Tint, Hue, Saturation; Clarity, Sharpness, Blur.

## How

- **Temp and tint** act where temperature does now (`_light_and_tone`,
  `render.py` ~1160): a per-pixel white-balance gain in log space, weight times
  the log gain. Work in log gain so ±200 is exactly twice ±100:
  `log2(gain(±100)) * value / 100`. Do not route ±200 through `point_of`, which
  clamps to ±100.
- **Whites and blacks**: add to `_MASK_TONE_KEYS`; `tone_curve` already reads
  them.
- **Hue**: on display-encoded colour beside the mask's saturation
  (`_mask_saturation` ~1219 and `_apply_look` ~738): a rotation about the grey
  axis by `weight * degrees`, luminance held. Build the map of degrees as the
  saturation map is built, and skip the work when no mask sets it. Skipped when
  `mono`.
- **Sharpness**: `adjust/sharpen.py` takes the finished colour and the settings.
  Give it an optional per-pixel amount map (the global `sharpen` plus each
  mask's `weight * sharpness`, on the 0..150 scale, floored at 0). Radius,
  detail and masking stay the global ones. A mask's +sharpness with the global
  at 0 sharpens only there; the develop pipeline must then run the sharpen stage
  even though `sharpen` is absent (`develop_frame` ~1340).
- **Amount**: not a render concept. `normalize_mask` keeps `amount` (0..200,
  dropped at 100), and one function, `masks.effective(mask)`, returns the mask
  with every control scaled and clamped and `amount` removed; `active`,
  `_mask_layers`, skin and defocus read masks through it. The sliders keep
  showing the values as set.
- The `_light_and_tone` cache key (~1136) already holds the masks; make sure hue
  and sharpness, which act later, are left out of it as saturation is.
- `backend/commands/develop.py` `_MASKS_MEANING`, the guide (`docs/guide.md`,
  grep "gradient"), and `lookAmount.js` `scaleMasks` if brief 1 has landed.

## Tests

`tests/test_masks.py` and `tests/test_adjust.py`: each new control moves only
where the mask reaches and by the stated amount; temp ±200 is twice the log gain
of ±100; hue 120 turns pure red to green at whole weight and leaves grey alone;
sharpness with the global at 0; amount 50 equals every control halved; amount 0
makes the mask inactive; an old edit (no new keys) renders bit for bit as before.

## Out of scope

Widening exposure, contrast and the rest past the global ranges (the user chose
to leave those). A curve per mask. Mask noise reduction. Combining two masks
with intersect or subtract: range, section, parts and erasing strokes cover it
today; raise it again only if a real case needs it.
