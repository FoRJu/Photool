# Brief: looks that carry masks which find their own place

One session, one branch (`looks-with-masks`, from `main`), one PR against `main`.
Backend first, with tests; then the panel. Read only what this brief names. Do
not read `frontend/src/DevelopView.jsx` whole (6,700 lines): grep the anchors
given. Line numbers are near enough, not exact.

This is the first of three briefs from the Develop review of 2026-11-16
(`develop-2-mask-controls.md`, `develop-3-grade-and-haze.md`). Do only this one.

## Why

A look (Develop > Looks) is how a style is made once and put on other photos: in
Develop, on a grid selection, as a camera's own look. Today a look never carries
a mask. The save form says so ("never crop, spots or masks") and the server
strips them (`backend/api/routes.py` ~2956).

That rule was right when a mask was a gradient or a brush drawn on one frame. It
is no longer true of most masks. Subject, sky, people, faces (with their parts)
and depth masks are found afresh on every frame, and so is a brightness or
colour range. "Darken and soften the background, lift the faces, smooth the
skin, deepen the sky" is most of what makes a portrait or travel style, and it
is exactly the part a look cannot hold. Copy and Paste already carries masks to
other frames and they re-find their place there; looks are the gap.

## What changes for the person

- **Save this edit as a look** gains a section, **Masks that find their place**,
  listing this frame's masks that can travel, each with a tick: `Sky · Exposure
  -0.50, Highlights -30`. Subject, sky, people, faces and depth masks are ticked
  to begin with; linear and radial gradients are listed unticked (a sky gradient
  travels well, a radial drawn round one face does not: the person decides).
  Brush and object masks, and a mask set to one person ("Person 2"), are not
  listed. The line under the name becomes "What it carries (never crop or spots;
  brushes and objects stay with their photo)".
- Putting such a look on a photo adds its masks to the photo's own. The photo's
  own masks stay. Masks from a look put on earlier are replaced. In the mask
  list a look's mask says where it came from (`Sky 2 · from “Stage night”`), and
  is edited and removed like any other.
- **Amount** scales the look's masks with the rest of it.
- The grid's **Look…** on a selection, and **Each camera's own look**, carry
  them too.
- A new row of built-in looks under the present ones, **Built in · finds the
  subject, sky, people and distance on each photo**:
  - **Subject pop**: subject `exposure 0.25, clarity 10`; background (subject
    inverted) `exposure -0.35, saturation -10`
  - **Soft background**: background `blur 35, exposure -0.2`
  - **Sky drama**: sky `exposure -0.5, highlights -30, contrast 20, clarity 15,
    saturation 15`
  - **Portrait glow**: people with areas `face_skin, body_skin` `smooth 25, even
    20`; faces `exposure 0.2`; all but people `exposure -0.25`
  - **Depth fade**: depth `near 40, far 0, soft 30` with `exposure 0.25, contrast
    -25, saturation -20, temperature -15` (distance going pale and cool)

  These set only masks: the tone sliders are left alone, so one can sit on top
  of "Warm film". Render each on three or four sample frames and tune the
  numbers by eye if one is plainly too strong or does nothing; say in the PR
  which you changed.
- With no look carrying masks, every picture renders exactly as it did.

## The edit model

- A mask gains one optional key, `look`: the name of the look it came from, a
  string of at most 80 characters. `normalize_mask` (`backend/adjust/masks.py`
  ~225) keeps it; it has no effect on rendering. It is written to the sidecar
  with the mask, as every mask key is (`to_text`, ~401).
- New in `adjust/masks.py`: `portable(mask) -> Optional[dict]`, the mask as a
  look may carry it, or `None`:
  - `brush`, `object`: `None`. A mask with `person` set: `None`.
  - `strokes` and `section` are dropped (drawn on one frame). `range`, `invert`,
    `areas`, the depth band and every control are kept.
  - a mask that changes nothing (`active`, ~391) is `None`.
- Saving (`api/routes.py` ~2940): when `"masks"` is in `sections`, keep
  `[portable(m) for m in masks]` less the `None`s, in `settings["masks"]`; drop
  the `masks` section when none is left. Otherwise strip masks as today. The
  panel sends only the ticked masks. A look that is only masks is a look: "There
  is no edit to save" must not refuse it.
- Putting a look on (`catalog/service.py` ~1150, the merge that today replaces
  the masks when `masks` is carried): when the incoming masks are not empty and
  every one has `look` set, the result is the target's masks that have no `look`
  (object masks included), then the incoming. Anything else behaves as today (a
  paste of Gradients & brushes replaces all but object masks). More than
  `MAX_MASKS` (16): the target's own are kept first and the look's that do not
  fit are left out; say so in the note the route returns if there is a simple
  place for it, and in the develop view's status line.
- `develop.paste` over MCP (`backend/commands/develop.py`, grep `preset`) must
  put a preset on by the same rule. Update `_MASKS_MEANING` and the presets
  command's docstring with one sentence each.
- LUT download (`routes.py` ~2990): masks cannot be in a LUT. Add them to what
  `X-Lut-Left-Out` names.

## The panel

Anchors in `DevelopView.jsx`:

- `COPY_SECTIONS` / `LOOK_SECTIONS` (~544–570): looks gain the `masks` section
  (label "Masks that find their place"); `lookOfPreset` (~572) must return the
  masks apart from the slider values, since they merge rather than replace.
- `LOOKS` (~620): the built-ins. Add the new row as a second list (`SCENE_LOOKS`)
  whose entries are `{ name, note, masks: [...] }`.
- `applyLook` / `applyPreset` / `changeLookAmount` (~5183–5215) and the `Fold
  id="looks"` JSX (~6023–6095): the save form's tick list, the second built-in
  row, and the merge: drop masks with `look`, add the look's with `look: name`.
- `frontend/src/lookAmount.js`: add `scaleMasks(masks, amount)`: every mask
  control times amount/100, clamped to its range (exposure ±4 in 0.05 steps,
  saturation ±200, `blur`, `smooth`, `even` 0..100, the rest ±100); `near`,
  `far`, `soft`, `range`, `invert`, `areas` untouched. It has Node tests beside
  it; add to them.
- The mask list's label (`maskLabel`, ~1150): append ` · from “<look>”`.
- Grid: `App.jsx` ~5185 (`lookPicked`) sends a look to `/photos/edit/batch`.
  Send the masks tagged with the look's name and `masks` among the keys.

## Cost, said plainly in the UI

A look with a subject or people mask on 200 selected photos runs a model on each
the first time it is drawn. Do nothing eager: thumbnails redraw as they do after
any paste. Add to the Look… menu's title: "A look with masks finds them on each
photo the first time it is shown, which takes a moment per photo."

## Tests

- `tests/test_masks.py`: `portable` for every kind; `look` survives `normalize`
  and the sidecar round trip.
- `tests/test_develop_routes.py`: saving with and without the `masks` section;
  a masks-only look; the LUT header.
- `tests/test_catalog.py` or where the merge is tested (grep `_merge` /
  `Gradients`): look masks replace earlier look masks and keep own masks and
  object masks; a plain paste behaves as before; the cap of 16.
- `tests/test_commands.py`: `develop.paste` with a preset that carries masks.
- Node tests for `scaleMasks`.
- Drive the page once in headless Edge against a scratch server on :8010: save a
  look with a sky mask from one frame, put it on another in Develop and on a
  selection in the grid, drag Amount, and check the second frame's edit. Attach
  two screenshots to the PR only if they show no real person; otherwise describe.

## Out of scope

New mask controls (brief 2). Dehaze, texture, colour grading, vignette shape and
a look browser (brief 3). Lens blur in a look. Importing presets from other
editors.
