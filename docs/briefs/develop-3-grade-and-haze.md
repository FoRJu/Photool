# Brief: the styling tools that are missing, and seeing looks before choosing

One session, one branch (`grade-and-haze`, from `main` once briefs 1 and 2 have
merged, or from the later branch if not: the PR is opened against `main`), one
PR. Backend first, with tests; then the panel. Do not read
`frontend/src/DevelopView.jsx` whole: grep the anchors given.

Third of three briefs from the Develop review of 2026-11-16.

## Why

Four global tools every style leans on are missing or half there, and a look can
only be chosen by its name.

1. **Colour grading is two-way and fixed.** Split toning tints shadows and
   highlights with a fixed crossover and nothing for the mid-tones
   (`_look_of`, `adjust/render.py` ~751). Most film and cinema styles put their
   colour in the mid-tones and move the crossover.
2. **The vignette has one control.** It always starts 35% of the way out
   (`_vignette`, ~1012). A tight spotlight and a faint edge burn are the same
   shape at different strengths.
3. **No dehaze.** Haze, backlight flare and a flat distant view are common on
   travel and outdoor frames; contrast and clarity do not undo them because haze
   grows with distance, not with tone.
4. **No texture.** Clarity works at one large scale (`smooth_base`, ~292);
   sharpening at a pixel. Nothing in between brings out cloth, bark or stone, or
   softens it, without touching edges.
5. **Looks are chosen blind.** Styles have "see this frame through every style
   at once"; looks, built in or saved, are a row of names.

## What changes for the person

- **Fade & toning** becomes three tints and a balance: adds `midtone_hue`
  (0..360), `midtone_sat` (0..100) and `tone_balance` (-100..100: left gives the
  shadows' tint more of the range, right the highlights'). At balance 0 with no
  mid-tone tint, a picture renders exactly as before. Sidecar names follow
  Lightroom's where its scale is ours (`crs:ColorGradeMidtoneHue`,
  `crs:ColorGradeMidtoneSat`, `crs:SplitToningBalance`); see `_LOOK_CRS`,
  `adjust/develop.py` ~849.
- **Effects > Vignette** gains `vignette_size` (-100..100, 0 as today: negative
  reaches further in, positive keeps to the corners) and `vignette_feather`
  (-100..100, 0 as today: negative a harder edge, positive softer). Both need
  `vignette` (`HUE_OF`, `develop.py` ~241, is where "needs" is recorded).
- **Presence** gains **Dehaze** (`dehaze`, -100..100) and **Texture**
  (`texture`, -100..100).
  - Dehaze: the dark-channel estimate (He, Sun, Tang) on linear light at a small
    size, refined with a guided filter against the luminance and applied at full
    size, with the airlight taken from the haziest 0.1% of pixels. Positive
    removes haze, up to 85% of the estimate at 100; negative adds an even haze
    of the airlight's colour. numpy and cv2 only, no model. It runs before the
    light and tone, after the blurs (`develop_frame`, ~1300), and must work on a
    cropped window of a frame the same as on the whole (`frame.window`): take
    the airlight and the small-size estimate from the whole frame and keep them
    on the frame as other per-frame results are kept. New module
    `adjust/dehaze.py`.
  - Texture: a second, finer split of the detail the tone stage already has
    (`LinearFrame.layers`, ~411): detail between about 2 and 12 pixels at full
    size, scaled by the frame's `scale` so a preview shows what an export gives.
    Gain on that band only, weighted off in the deepest shadows so noise is not
    brought up.
  - Both join `SCALED` in `lookAmount.js`, the Copy section `presence`, and
    `_LIGHT_AND_TONE_KEYS`.
- **Looks > Browse**: a button beside the built-in row opens a sheet of small
  pictures of this frame, one for each built-in and saved look, each drawn with
  the look put on the frame's present edit by the same rule a click uses. Click
  one to use it. Reuse what the Style browser does (grep `See this frame through
  every style` ~3322 and follow it to its component and its route) rather than
  building a second mechanism; draw at most 320 px on the long side, a few at a
  time, and stop when the sheet closes. A look with masks may take a moment the
  first time: show the name on a grey tile until it arrives.

## Tests

- Grading: balance 0 and no mid-tone tint is bit for bit the old picture; a
  mid-tone tint moves mid grey and leaves black and white alone; balance ±100
  moves the crossover the right way.
- Vignette: size and feather at 0 are bit for bit the old picture; the centre is
  never changed.
- Dehaze: a synthetic scene with haze added by the model is brought back to
  within a stated error; a clean, contrasty frame changes little at +30; a crop
  window matches the same area of the whole frame.
- Texture: a mid-scale pattern gains contrast, a hard edge has no halo beyond a
  stated size, a flat patch is unchanged.
- Sidecar round trips for every new key; `develop.controls` lists them with a
  one-line meaning each.

## Out of scope, and why

- **Defringe / chromatic aberration**: measured at 0 px on this lens set.
- **Calibration (primaries)**: the camera look and styles own that ground.
- **Point colour**: a colour-range mask with Hue and Saturation (brief 2) is it.
- **Presets from other editors**: the scales differ enough that an imported
  preset would look wrong more often than right. Ask before building.
- **More export formats**: not wanted.
