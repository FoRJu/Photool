# Brief: lens blur with a clean edge, from an outline

One session, one branch (`blur-outline`, from `main`), one PR. Backend first,
with tests; then the panel. Read only what this brief names. Do not read
`frontend/src/DevelopView.jsx` whole (6,400 lines): grep the anchors given.
Line numbers are near enough, not exact.

## Why

Lens blur (`backend/adjust/depth.py`) puts each pixel in a layer by the depth
map. The map is about 1000 px across and soft by ten or twenty of its pixels at
an edge, so on a 6000 px frame the band where it is wrong is a hundred pixels
wide. There a person's edge is given a distance halfway between theirs and the
wall's: the edge of the person goes soft, a ring of the wall beside them stays
sharp, and their colour is smeared into the blur. The guide owns up to it
("look closely round hair and other fine edges").

Segment Anything already draws a far finer outline of one thing, at up to
4096 px, and the app already keeps such outlines as pictures
(`backend/adjust/objects.py`). This PR lets the blur take its edge from one.

It changes nothing about what is in the photograph: no pixel is invented, the
blur is the same blur, cut where the thing really ends.

## What changes for the person

In the Lens blur group, under **Pick the focus on the picture**:

- **Outline what is sharp**: the next click on the picture outlines the thing
  clicked (SAM 2.1), sets the focus to its distance, and from then on the blur
  ends at that outline. A second use adds another thing to the outline.
- A line says what is held, `Outlined: 2 things`, with a small grey picture of
  the outline, and **Clear**.
- The first time, SAM 2.1 is fetched (330 MB) with a note saying so, as the
  depth model is.
- With no outline, every picture renders exactly as it did.

## The edit model

One new key, `blur_outline`, shaped as an object mask's choice is:

```
"blur_outline": {"matte": "<hex id>", "parts": [ … ], "model": "sam2.1"}
```

- `parts` are object-mask parts (`adjust/masks.py: _normalize_parts`, ~178);
  `matte` names a kept picture (`objects.save` / `objects.load`), checked as an
  object mask's is (`masks.py` ~253). No valid `matte`: the key is dropped.
- `normalize` (`adjust/develop.py` ~441) drops it when there is no `lens_blur`.
- It belongs to its frame: add it to `OWN_KEYS` (`develop.py:196`), which keeps
  it out of presets (`api/routes.py` ~2917) and out of what a batch carries
  (`catalog/service.py` ~1055; read those ten lines to be sure that is what the
  set does). In the panel it is **not** added to the Copy checklist's `blur`
  keys, so a paste leaves each frame its own.
- Sidecar: compact JSON under `crs:PhotoolBlurOutline`, written and read the
  way `_SECOND_CRS` is (`develop.py` ~880, ~941, ~1064, and the tuple at ~1091).

## The maths

New module `backend/adjust/outline.py`. numpy and cv2 only, no model.

**1. Two depths instead of one** — `sides(depth, matte) -> (inside, outside)`
or `None`.

`depth` is the whole frame's map (`depth.whole_map`, 0 far to 1 near); `matte`
the outline, 0..1, any size. Bring the matte to the map's size (`INTER_AREA`).

- *Sure inside*: `matte > 0.5`, eroded by `r = max(3, round(longest / 50))`
  pixels of the map (an ellipse). *Sure outside*: `matte < 0.5`, eroded the
  same. The band between is where the map cannot be trusted.
- `inside` is the map where it is sure inside and, everywhere else, carried out
  from there: `blur(depth · sure) / blur(sure)` with a Gaussian, filling what is
  still empty at sigma 2, 4, 8, … until none is. `outside` likewise from the
  sure outside.
- No sure inside at all (a thin thing): use `matte > 0.5` uneroded. No pixel
  over 0.5, or none under: `None` (there is no edge to take).

A prototype of exactly this, on a step softened by sigma 3 at 400 px, gave
`inside` ≥ 0.997 and `outside` ≤ 0.003 right up to the edge.

**2. Each pixel shared between the two** — `depth.lens_blur` gains
`edge=None`, a pair `(alpha, inside)` at the picture's size; `depth` is then the
outside. The one changed line is the layer's share:

```
mine = max(1 - |position - i|, 0)                       # as now
mine = (1 - alpha) * mine + alpha * max(1 - |position_inside - i|, 0)   # with an edge
```

So an edge pixel that is half person is half in the person's layer and half in
the wall's, not wholly in a layer between them. Everything after that line is
untouched. With `edge=None` the result is bit for bit what it is today. In the
prototype, with the edge: the subject's edge columns changed by 0.005 (0.38
without), the background beside it was as blurred as far from it, and none of
the subject's colour reached the blur (0.24 without).

**3. Which of SAM's three answers** — `pick(logits, scores, depth, at) -> int`.

One click is often several things (the shirt, the person, the person and the
floor). `logits` is `(3, h, w)` from `objects.predict`, `depth` the map brought
to that size, `at` the map's value at the click. A candidate *holds together*
when at least 80% of its pixels (`logit > 0`) are within 0.15 of `at`. Take the
largest that holds together; if none does, `objects.chosen(scores, None)`. The
answer goes in the part as `pick`, so the outline kept is the one chosen
(`objects._part_matte` reads it).

## Backend wiring

Before anything, `grep -rn blur_sharp backend frontend/src` and make sure every
place that hands the three blur keys on hands `blur_outline` on too.

- `adjust/depth.py`:
  - `blurred_pixels(…, layers=LAYERS, outline=None)`: when `outline` names a
    matte that `objects.load` finds, get the sides once per whole frame (keep
    them in `whole._mattes[("depth_sides", matte_id)]`, `whole` as `whole_map`
    takes it), then `segment.sample` the matte and both sides to the pixels and
    window being rendered, and pass them to `lens_blur`. A matte that is not
    here (an edit from another computer), or `sides` returning `None`: as with
    no outline.
  - `blurred(frame, amount, focus=0, sharp=0, outline=None)`: the matte id
    joins the kept key (~418).
- `adjust/render.py` ~1279 and `export/engine.py` ~2371 and ~2439–2442 (the
  loupe's patch): pop and pass `blur_outline`.
- `api/routes.py`, modelled on `develop_depth_at` (~3544) and
  `develop_object_commit` (~3879):
  `POST /export/develop/blur-outline`
  `{filepath, u, v, settings, parts: [], + the FrameGeometry fields}` →
  `{outline: {matte, parts, model}, depth}`.
  - frame: `_mask_frame(config, filepath, objects.PICTURE_SIZE, req)`; the point
    through `adjust.geometry.source_point`, as `depth-at` does.
  - the new part is `{"view": [0, 0, 1, 1], "points": [[u, v, 1]]}`, with `pick`
    from `outline.pick`; `objects.predict(objects.picture(frame), part,
    "sam2.1", _object_key(req, size))`.
  - `parts` is the outline so far (normalized) with the new part added;
    `objects.selection` over all of them, `objects.save`.
  - `depth` is `depth.depth_at(frame, u, v)`, for the Focus slider.
  - SAM 2.1 only, never SAM 3 (its licence is asked for elsewhere).
    `ModelsUnavailable` → 503 with its sentence; unreadable frame → 404.

The outline lies on the frame as its lens correction and transform left it, as
an object mask's does. Change those afterwards and it no longer fits; that is
already so for object masks and is not solved here.

## Panel work

New file `frontend/src/BlurOutline.jsx`; `DevelopView.jsx` only mounts it and
lends it the picker, exactly as it does `WhiteBalance.jsx` (read that file's
props and its picking code; it is the pattern).

Anchors in `DevelopView.jsx`:

- where it is mounted: grep `data-testid="blur-pick"` (~5647); put it after
  that button, inside the same `blurOn` condition
- the picker to copy: grep `whitePick` (3 places: ~4093, ~4103, ~5662). Add an
  `outlinePick` ref the same way, and make the three pickers turn each other
  off (grep `setPickingWhite(on)`)
- the request body to copy: grep `focusPick.current = pickingFocus` (~5686)
- the model note to copy: grep `Getting the depth model` (~4743); the route is
  `POST /export/develop/object-model {model: "sam2.1"}`, called before the
  first outline of a visit

The component:

- **Outline what is sharp**, `data-testid="blur-outline"`; while waiting for the
  click it reads `Click it on the picture…`. With an outline it reads **Add to
  the outline**.
- On the click: post, then one change that sets `blur_outline` and
  `blur_focus` together (grep `applyGroup` for how two keys are set at once).
- With an outline: `Outlined: 1 thing` / `N things` (the parts), an `<img>` of
  `/export/develop/object-matte/{matte}` about 96 px wide on
  `bg-neutral-950 rounded border border-neutral-800`, and **Clear**
  (`data-testid="blur-outline-clear"`), which removes the key.
- A 503 or 404 shows its sentence under the button.
- Phone width: buttons wrap, targets at least 32 px high.

Do not move or restructure anything else in the file.

## The guide

`docs/guide.md`: grep `**Lens blur does what`. Add the control to the list
there and say in two or three sentences what it does and that it belongs to its
frame. Touch nothing else in the guide.

## Tests

New `tests/test_blur_outline.py`. Borrow `steps`, `detail` and `framed` from
`tests/test_depth.py` (lines 21–40) and that file's way of working without a
model (`frame._mattes["depth"] = d`). For a kept matte:
`monkeypatch.setattr(objects, "_mattes_dir", str(tmp_path))` then
`objects.save(matte)`.

Run it with `tests/test_depth.py`, `tests/test_adjust.py`,
`tests/test_render.py`, `tests/test_develop_routes.py` and
`tests/test_develop_view.py`, not the suite: `docs/setup.md` line 219.

The scene for 3–6: 400 × 300, blue stripes on the left half (far), red stripes
on the right (near); the matte is the crisp right half; the depth is that step
softened with a Gaussian of sigma 3; focus 1.0, radius 12.

1. `sides` on that scene: inside ≥ 0.99 and outside ≤ 0.01 everywhere.
2. `sides` keeps a slope: the near half's depth running 0.6 to 1.0 down the
   frame, inside follows it within 0.05 away from the top and bottom rows.
3. Without the outline (what happens today, kept as the comparison): the
   subject's first 15 columns change by more than 0.1.
4. With it: those columns change by less than 0.02; the background's detail in
   the 30 columns beside the edge is within 1.5× of its detail far from it; the
   red channel there is under 0.01.
5. An outline whose matte is not on disk, a matte of all ones, a matte of all
   zeros: `np.array_equal` with no outline.
6. A frame of 800 × 600 and one of 400 × 300 with the same matte agree, as
   `test_same_blur_at_any_preview_size` checks.
7. `blurred` is kept for the same outline and made again for another.
8. Through `develop_frame` with `blur_outline` in the edit: differs from the
   same edit without it, and only near the edge.
9. `pick`: three made-up candidates, a small one, a middling one at one
   distance, a large one spanning distances: the middling one. None holding
   together: the best score.
10. `normalize`: kept beside `lens_blur`, dropped without it, dropped with a
    bad `matte`; its parts cleaned. It is in `OWN_KEYS`. Sidecar round trip.
11. The route, with `objects.predict`, `objects.selection` and
    `depth.whole_map` monkeypatched to made-up answers: 200, a matte kept, the
    earlier parts carried, `pick` set.
12. `npm run build` in `frontend` succeeds.

If a tolerance here turns out wrong for a good reason, change it and say so.

## By eye, in Chrome

Say in the PR which of these were done, with a before and after at 1:1 of one:

- a portrait against a busy background: hair and shoulders end cleanly, no
  sharp ring of background, no smear of the person into the blur
- clicking a shirt outlines the person, not the shirt
- a second click adds a second person; Clear takes it back to today's blur
- the loupe at 1:1, the preview and an export agree
- a frame with no outline is unchanged
- the outline survives a reload, and is not carried by Paste or a saved look
- 390 px wide: it fits the panel

## Out of scope

Outlining without being asked (people, the subject), reusing an object mask's
outline, SAM 3 or naming things, a box or Alt-click to refine, stepping through
the candidates with Tab, the edge between two outlined things, the depth mask,
re-fitting an outline after the lens or transform changes, any refactor of
`DevelopView.jsx`.

## When done

Say in the PR what was not verified by a test and was checked by eye instead,
and anything in this brief that turned out wrong.
