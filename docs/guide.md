# Photool in depth

How each part of Photool works and why it works that way: the features, the
measurements behind the numbers, and the design decisions. For installing and
configuring it, see [setup.md](setup.md); for an overview, the
[README](../README.md).

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Browser UI (React)                             │
│  ┌───────────┐ ┌──────────┐ ┌────────────────┐ │
│  │  Importer  │ │  Culler  │ │     Develop    │ │
│  │   Panel    │ │  Grid +  │ │   Exposure +   │ │
│  │           │ │  Single  │ │  tone, presets │ │
│  └───────────┘ └──────────┘ └────────────────┘ │
└──────────────────┬──────────────────────────────┘
                   │ HTTP / WebSocket
┌──────────────────▼──────────────────────────────┐
│  FastAPI Backend (Python)                       │
│  ┌──────────┐ ┌──────────┐ ┌─────────────────┐ │
│  │ Importer │ │ Catalog  │ │ RAW Processor   │ │
│  │ SD→Disk  │ │ SQLite + │ │ rawpy + PIL     │ │
│  │ checksum │ │ XMP      │ │ thumbnails      │ │
│  └──────────┘ └──────────┘ └─────────────────┘ │
└─────────────────────────────────────────────────┘
```

**Stack:** Python 3.11+ · FastAPI · rawpy · SQLite · React · Tailwind CSS

## Modules

| Module | Status | Description |
|--------|--------|-------------|
| **Importer** | ✅ v1 | SD card detect → copy to date folders, checksum verify, skip dupes |
| **Catalog** | ✅ v2 | SQLite index + XMP sidecar dual-write for portable ratings |
| **Thumbnails** | ✅ v1 | Extract embedded JPEG from a RAW (CR3/CR2) or a camera JPG, resize to grid/preview, disk cache |
| **Culler UI** | ✅ v2 | Grid view, single view, keyboard rating (1-5, P/X flags, color labels) |
| **Event tags** | ✅ v2.1 | Text labels per photo (school event vs family), stored in the catalog and in XMP `dc:subject` so Lightroom sees them too |
| **Bulk select** | ✅ v2.1 | Filter a folder by rating / flag / colour / tag, then act on the whole match set — add it to the basket, or tag/untag it |
| **Measurements** | ✅ v2.2 | Brightness, colour cast, clipping, sharpness and a perceptual hash per photo, measured in the background from the cached thumbnail — the basis for culling hints |
| **RAW-first export** | ✅ v2.2 | A CR3/JPG pair is developed from the CR3, with the capture dates copied back out of it, so a set starts from the same kind of data instead of the camera's per-shot auto settings |
| **Burst grouping** | ✅ v2.3 | Frames of the same shot are recognised as one burst — badged in the grid, shown as a strip in the cull view, and one key press keeps the sharpest frame and rejects the rest |
| **Review flow** | ✅ v2.3 | Undecided-only filter, resume where you stopped, progress through a folder |
| **Quality hints** | ✅ v2.3 | Soft-frame and blown-highlight warnings from the same measurements — a mark on the frame, never an automatic reject |
| **Compare view** | ✅ v2.3 | 2-up / 3-up side-by-side of one burst, with a 1:1 crop to judge focus |
| **Develop** | ✅ v4 | A full-screen develop view (`D`) for one photo at a time, working in linear light from the RAW data — exposure (±5 EV), highlights, shadows, whites, blacks, HDR squeeze, contrast, clarity, white balance, vibrance, saturation, fade and split toning, a point tone curve (RGB and per channel) and an 8-band colour mixer (hue, saturation, luminance), noise reduction, sharpening, lens corrections and transform — with Auto, one-click rescue and look presets, saved presets, copy/paste, apply-to-burst, a before/after split, a histogram and a clipping overlay. Library thumbnails show the edit. The edit is non-destructive and per-photo: the catalog and the XMP sidecar are both written on save, so it survives the basket, a catalog rebuild, and opens in Lightroom |
| **Crop & level** | ✅ v4.1 | In the develop view (`C`): drag a crop box, pick a shape (Original, 1:1, 4:5, 5:4, 3:2, 2:3, 4:3, 3:4, 16:9, 9:16 for Stories/Reels) and level a horizon with the Level control. It is part of the edit — non-destructive, shown in library thumbnails, applied on export, and saved to the XMP sidecar (also as Lightroom's `crs:Crop*` when the frame is not turned). Copy/paste, presets and *To burst* carry the tone and leave each frame's own crop alone |
| **Virtual copies** | ✅ v4.2 | More than one edit of the same file — a colour and a black-and-white, a square and a 9:16. In the develop view, **Versions → + Copy** starts a new edit from the one you are on; each has its own tone and crop, a name, and an *export* switch. No second file is made: copies live in the catalog and in `<photo>.copies.json` beside the photo (so a catalog rebuild keeps them). Exporting a basket frame writes the main edit as `IMG_0001.jpg` and each copy beside it as `IMG_0001_vc1-name.jpg`; the basket shows `+N` for a frame with copies. The grid and thumbnails show the main edit; paste, apply-to-many and *To burst* never touch copies |
| **1:1 loupe** | ✅ v4.1 | Click the picture in the develop view (or `L`) to see that spot at full size in a floating pane — for checking a face, a highlight or a shadow. **1:1** shows the photograph's own pixels one for one, from the full-resolution RAW decode, and **2:1** shows each as four screen pixels, kept square rather than smoothed so a soft upscale is not mistaken for soft focus. The patch is sent losslessly (PNG): as a JPEG at quality 88 it lost more than a third of its fine detail, and at 95 a sixth. It follows the sliders as you drag them, is toned from the same whole-frame base as the preview so the two agree, and stays on the same spot when you step to the next frame |
| **Histogram & readout** | ✅ v1 | The develop view's histogram shows red, green and blue on their own as well as the brightness, so a channel pinned at white (a blue sky at 255 in blue, only 75% bright) shows at the right edge where the readout counts it. The pixel under the cursor is read out as its place in the whole photograph and its colour in the edited picture, in RGB, RGB %, HSL or Lab |
| **Tone curve & colour mixer** | ✅ v5.2 | In the develop view: a point curve for RGB and for R, G and B (click the line to add a point, drag to shape, double-click to remove; monotone, so it never overshoots between points), and Lightroom's eight colour bands with hue, saturation and luminance each. The mixer keeps a colour's brightness unless luminance is what you move, so a blue taken to grey stays a dark grey. Both are part of the edit like any slider — presets, copy/paste and *To burst* carry them, the mixer is written to the sidecar under Lightroom's own `crs:HueAdjustmentRed`… names, and the curves under this app's own `crs:PhotoolToneCurve…` |
| **Black & white** | ✅ v5.3 | A *Treatment* switch in the develop view turns the photo to grey properly (from its brightness, not by draining the colour), with a B&W mix of eight colour bands — how light each colour turns, like Lightroom's — and one-click film filters (yellow, orange, red, green, blue, infrared). Toning on top: sepia, selenium, platinum, cyanotype, split, or by hand with the split-tone hues. Film grain (amount and size) and a vignette, per photo, on the picture as cropped; the grain is the same grains every time and sized to the picture, so the preview, the 1:1 loupe and the export agree. Looks: *B&W punchy*, *film*, *dramatic*, *soft*, *sepia*. Search with `is:bw`. Written to the sidecar under Lightroom's names (`crs:ConvertToGrayscale`, `crs:GrayMixer…`, `crs:GrainAmount`, `crs:PostCropVignetteAmount`) |
| **Gradients & brushes** | ✅ v1 | Local adjustments in the develop view: a **linear** gradient (whole at one end, fading out to the other: a darker sky, a lighter foreground), a **radial** one (whole inside an ellipse, fading out across its feather, or outside it with *invert*: light on a face, a vignette of your own, turned by its handle on the picture or its Angle slider), a **brush**, only where it is painted (size, feather and flow; `[` `]` resize, down to 0.05% of the frame for a hair's breadth, Alt erases), or the **subject** or **sky** as a model finds them in the frame (U²-Net-p and SegFormer-B0, 4.6 and 15 MB, fetched into `models/` on first use; the edges follow the picture through a guided filter; *invert* for the background or the land), or the **people** and their **faces**: everyone (SegFormer's person class) and a soft oval on each face the People tab's detector finds, kept inside the people, so +0.5 EV and −20 clarity lights and softens the faces of a group. Either can be narrowed to **one person**: the faces are numbered on the picture from the left, a click picks one, and each part of the people selection goes to the face it is under or nearest to. Faces under 40% as wide as the largest (the crowd behind a family) are left out. Any mask can be narrowed by a **range**, as Lightroom's range masks are. **Brightness** keeps it to a band of the picture's brightness (0–100, with a smooth fade beyond), so darkening a sky with a gradient leaves the band members against it alone. **Colour** keeps it to a colour picked on the picture and the colours near it, compared as a share of their brightness, so a sash in shade is still the sash; the *Range* slider widens it. The range is judged on the picture as shown before the edit, so a mask's own exposure does not move what it reaches. Any mask can be **sectioned**: a rectangle, dragged on the picture and feathered, that it reaches only inside — a radial's light stopped where the wall begins, a linear gradient kept off the foreground. Any of them can be painted into or **erased** from, so a radial's light on a face need not reach the sky beside it, and the red *Show where it reaches* shading shows the result. Each has its own exposure (±4 stops), contrast, highlights, shadows, temperature, saturation and clarity, performed where the global sliders are, in linear light, so a gradient's +1 EV is the Exposure slider's stop where it is whole. Drawn on the whole frame (positions are fractions of it before crop and level, so a later crop does not move them), shown in the loupe and exports, carried by *To burst*, paste and presets, cleared by Reset, and kept in the sidecar as this app's own `crs:PhotoolMasks` (Lightroom does not read it) |
| **Spot removal** | ✅ v1 | Dust, a blemish or a wire, painted out in the develop view (Q): click a spot and a clean source nearby is chosen for it (drag either circle to move it). **Heal** brings the source's texture in, in the spot's own light and colour; **Clone** copies it as it is. Size (`[` `]`), feather and opacity. Done first, on the linear frame, so every other control sees the cleaned picture; shown in the loupe at 1:1 (sources read from the full decode) and in exports. A frame's own, like its crop: not carried by paste, presets or *To burst*, and kept by Reset. Kept in the sidecar as `crs:PhotoolSpots` |
| **Match a look** | ✅ v1 | Give a frame the tone and colour of any reference picture: a photo of yours as you edited it (*Use this photo*, then *Match* on another frame) or a JPEG/PNG from elsewhere (*From a file…*). The look is read as percentiles of each channel and handed back as ordinary controls — red, green and blue curves and a saturation, or black and white with a curve for a monochrome reference — with an *Amount* to take it part of the way. Each level moves at most 70 of 255. Best with a reference from the same shoot: across shoots it copies the reference's content as well as its look (see *Paste and match*). Editable, undoable and saved like any edit |
| **People** | ✅ v5 | Faces found and grouped on this machine with ArcFace — a name per group, written as an event tag; "is this Alice?" suggestions, merge and detach to correct a wrong answer, and names kept across a re-scan; a person's *timeline* — their best face for every month, played as a flip book |
| **Year in review** | ✅ v1 | A page per year on the Library front page: the year in numbers, its best frames chosen across every day out (rated or not), who was in it, the biggest shoots and the places away from home — and a collage of the best to download. See [Year in review](#year-in-review) |
| **Collage** | ✅ v1 | 2 to 9 photos on one fixed page: a 9:16 phone story, a square, a 4×6, 5×7 or 8×10 print at 300 dpi, or a 3:2 screen. Each photo is cut to fill its cell around the people in it, and the layouts offered cut away the least. See [Collages](#collages) |
| **Agent control (MCP)** | ✅ v5 | The library as MCP tools at `http://127.0.0.1:8000/mcp`: find, cull, tag, develop, preview and export, the same actions the UI takes, on the same catalog. See [Driving it from an agent](#driving-it-from-an-agent-mcp) |
| **Library search** | ✅ v5.2 | One line of text over the whole library — `rating:>=3 iso:>3200 date:2026-04 -is:edited` — and, with smart search, by what the photos show: `kids at the beach rating:>=3`. See [Searching the library](#searching-the-library) |
| **Styles** | ✅ v1 | A look learned from many frames — a Ricoh's Positive Film, Canon's Neutral, your own edited exports — kept under a name and rendered on any RAW, from any camera |
| **Other computers** | ✅ v1 | The album on a laptop or phone in the house, for devices you have paired; nobody else on the Wi-Fi sees a thumbnail |

## Selecting a set of photos

The Library toolbar filters the current folder by star rating, pick/reject flag,
colour label, event tag, and RAW/JPG pairing. The row under it acts on whatever
that filter currently shows:

- **⬡ Add N to basket** copies every matching photo into the export basket in one
  call, so a whole event goes out without clicking ⬡ photo by photo. The basket
  stays a plain editable list — remove anything that slipped in before exporting.
- **Tag N / Untag** attaches or removes an event label across the same selection.
  Tags are free text (autocomplete offers existing names) and are written into
  each photo's XMP sidecar as `dc:subject` keywords.

Tag matching is pair-aware: if either half of a CR3/JPG pair carries a tag, the
photo stays visible in the grid whichever pairing mode you are viewing.

A filter change redraws from the folder already in memory, so it answers on the
click however large the folder is. Only the derived counts — how many bursts, how
many soft frames — go back to the server, and they are asked for once, after the
last change settles: clicking through three colour tags in a row costs one folder
query, not three, and an answer that a newer change has already replaced is
discarded rather than painted over the current one.

## Measurements

Culling and consistent output both need numbers, not just pictures: how bright a
frame is, what colour the camera was treating as white, whether it is softer
than its neighbours, and whether two frames are the same shot. Every indexed
photo gets one row in the `photo_stats` table:

| Field | What it is |
|-------|------------|
| `mean_lum`, `lum_p05` / `lum_p50` / `lum_p95` | the brightness distribution |
| `clip_high`, `clip_low` | fraction of pixels blown out or blocked up |
| `wb_r`, `wb_b` | grey-world white balance, as channel gains against green |
| `sharpness` | variance of the Laplacian — comparable between frames of one folder |
| `dhash` | 64-bit difference hash: the same shot differs in a few bits, a different shot in many |

They are measured from the cached grid thumbnail, which for a CR3 is the JPEG the
camera already embedded, so a folder costs a few hundred small JPEG decodes and
never a demosaic. Each row records the source file's size and modification time
and the version of the measuring code, so a stored number is only ever used while
it still describes the file it came from — a re-shot or re-edited file is
measured again.

The thumbnail queue does this by itself: after a folder's thumbnails, and one
folder at a time whenever the queue has had no thumbnail work for two minutes, so
a library that already exists fills in gradually in the background instead of in
one long scan. It walks the folders newest event first — folder names are
`YYYY/YYYY-MM-DD`, so descending name order is descending shoot date, and the
first hour of idle time goes on the shoot you are about to cull rather than on a
2016 holiday — and it keeps its place, so a folder it cannot read does not stop
the ones after it. Set
`analysis_auto_enabled: false` in `config.yaml` to stop that; `POST
/api/analysis/run/<folder>` measures one folder on demand and `GET
/api/analysis/status/<folder>` reports how much of it is still missing.

A folder that has not been measured shows nothing for any of this, which looks
exactly like a folder that has nothing in it. So the row under the Library
toolbar says how many frames in the open folder are still unmeasured and offers
**▷ Measure this folder** to run that pass now rather than wait for the queue's
turn. It reads thumbnails that are already on disk and writes only derived
numbers — nothing in the library changes. A folder whose thumbnails are still
being generated takes longer, because those have to exist first.

The export path reads these numbers to match a set as it exports — see
[Matching a set](#matching-a-set) below. The culling path reads them to group
near-identical frames and to mark soft or clipped ones — see
[Culling a burst](#culling-a-burst) and [Quality hints](#quality-hints). The
side-by-side compare uses the same rows and is the next step.

### Exporting a paired photo

When a CR3 and its JPG are both on disk, the export develops the **CR3** and
copies the capture dates back out of it, because rawpy writes pixels and no
metadata. That is what makes a set comparable at all: the JPG is the camera's own
per-shot auto exposure, so a set of JPGs inherits a different rendering in every
frame. The basket and the filenames are unchanged — only the file the pixels come
from. Set `export_prefer_raw: false` to go back to exporting exactly the file the
basket holds.

### How a photo is written

Every photo in the basket is written as it stands, at the size asked for:

- A photo you have **developed** is written as you developed it. That includes
  its noise reduction, sharpening and crop, on top of the camera's learned look.
  Each virtual copy marked for export is written beside it.
- A photo you have **not** developed is written as the camera shot it. For a RAW
  that means the camera's own JPEG: the file beside it when the shoot was
  RAW+JPEG, else the JPEG inside it. Only a RAW that has neither falls back to a
  decode. A RAW is not a picture; the camera's JPEG is, with its white balance,
  Picture Style, lens correction and noise reduction already in it. Measured on
  an R6 II frame: the JPEG resized is identical to it. A decode with the learned
  camera look is 8.5 levels of 255 away, and a plain decode is 48 levels away and
  a stop dark.

The export does no editing of its own. The panel once had batch styles (*Clean*,
*Vibrant*, …) and a *Match set* nudge towards the set's median brightness and
colour. Both are gone. They were a second, cruder edit on top of the
photograph's own: a style's sharpening on top of the develop view's, or a
match moving a frame away from how it was developed. A set is now made to look
alike in the develop view, by copying one edit and pasting it onto the rest. A
request that still names a style or matching is answered as if it did not.

**Sharing an event.** The panel remembers the credit, the size and the quality on
each device:

- The **Credit** is written into every photo as its EXIF Artist and Copyright.
- **Size** is 2048 px (plenty for Google Photos, which keeps it as it is, and for
  any screen), full size, or another longest side.
- **Keep only the date taken and the credit** strips the location and the camera
  details. The dates are kept, so an album in Google Photos is in the order the
  photos were shot.

When an export on this computer finishes, **Open folder** shows it in Explorer,
ready to drag into an album at photos.google.com. On a phone, the share sheet
sends the photos to Google Photos directly.

### Adjusting one photo

Sometimes one frame needs a hand — a sky
that was blown, a face in shadow, a frame shot two stops under — or just a little
more polish. The develop view (`D`) is that hand: it is built to **rescue
exposure**, to **develop a style**, and to add **subtle improvements** to a frame
that was already right.

**An edit is made in linear light, from the sensor data.** The first version of
this view adjusted the finished 8-bit picture, and that is why it felt mild and
could not rescue anything: exposure multiplied the *encoded* values (a "stop" was
really about two and a bit, and anything that reached 255 was gone), and a
highlights control on a pixel that was already white has nothing to pull back.
Now a RAW file (CR2, CR3 and the rest) is decoded to 16-bit linear with headroom
above white, a JPEG is brought to the same footing, and the edit is applied to
that:

1. **Light.** Exposure is a multiplication in linear light — +1 EV is one stop of
   light — and white balance rides in the same step. The range is ±5 EV, enough
   for a frame shot three or four stops under.
2. **Tone.** The frame is split into a *base* (its large-scale brightness, found
   with an edge-preserving guided filter) and a *detail* layer (everything finer).
   Highlights, shadows, whites, blacks, contrast and Squeeze reshape the base
   only; the detail goes back on top. That is what lets a window come down three
   stops without its frame going flat, and a shadow open without the surface in
   it washing out — and it is why a hard edge such as a horizon gets no halo. The
   guided filter's coefficients are solved on a small copy but applied to the
   full-size frame, so the base is exact at an edge.
3. **Display.** Values above white are rolled off with a smooth shoulder instead
   of being cut, then encoded, dithered by half a level and rounded **once** — a
   shadow lifted by three stops would otherwise show steps.
4. **Look.** Vibrance, saturation, fade and split toning, on the finished tones.

| Group | Controls |
|-------|----------|
| **Light** | Exposure (±5 EV), Highlights, Shadows, Whites, Blacks, **Squeeze** |
| **Presence** | Contrast, **Clarity** |
| **Colour** | Warmth, Tint, **Vibrance**, **Saturation** |
| **Look** | **Fade**, **split toning** (hue and strength for the shadows and the highlights) |

**Squeeze** is the answer to "can a wide scene be put into a normal 8-bit JPEG":
it compresses the scene's overall range — a bright window and a dark room, ten
stops across — and keeps the local contrast, so the result reads as a photograph
and not as a flat grey one. Combine it with Highlights and Shadows to choose where
the room goes. The tone controls reach far on purpose (Shadows is up to 3.5 stops
of fill light at the deep end, Highlights up to 2.8 stops of pull-back); the curve
they build is forced monotonic, so a pile of extreme sliders can flatten part of
the picture but never reverse it.

One-click starting points sit above the sliders. **Rescue** sets the light and
leaves the styling — *Recover sky*, *Open shadows*, *Backlit*, *Squeeze HDR*,
*Lift 2 stops* — and **Looks** set the styling and leave the light and the white
balance alone — *Subtle polish*, *Clean pop*, *Matte*, *Warm film*, *Teal &
orange*, *Black & white* — so the two combine. Double-click a slider (or its
label) to reset just that control. `J` overlays the **clipped areas** — red where
the picture is pure white, blue where it is pure black — and the histogram reports
how much of the frame is pinned at each end.

Two things it deliberately does not do. It does not invent detail: a pixel the
sensor clipped holds nothing, so pulling highlights back over it shows a grey
patch, not a cloud; what is recovered is what the sensor kept and an 8-bit
rendering would have thrown away. And it does not decide what the photograph
should look like.

**Auto** reads the frame and sets the **light** from its own histogram: exposure
(moved most of the way to a sensible median, not all of it, and not at all inside a
deadband, so a frame meant to be dark stays dark), highlights if the bright end
runs past white, shadows if the dark end is buried, and Squeeze when the scene
spans more range than a picture can show. It replaces the light controls and
leaves colour and look alone, so pressing it twice is the same as once. It
measures the frame it is about to change — the linear decode the preview is made
from — never the stored thumbnail statistics: those were taken from the camera's
own rendering, with its tone curve already in them.

**A RAW frame starts from the camera's own look.** A Canon file carries two
renderings of one exposure: the sensor data and the JPEG the camera made from it.
The JPEG is the sensor data *plus* the camera's white balance, its Picture Style
tone curve and saturation, its highlight and shadow handling and often its Auto
Lighting Optimizer, while a flat decode of the RAW has none of that — which is why
a default CR2/CR3 render used to sit so far from what the back of the camera
showed. The develop view now **learns that rendering from the file's own embedded
JPEG** and uses it as the starting point (`adjust/look.py`):

- a **colour matrix** from the decode's linear light into the camera's rendering
  primaries, which carries the Picture Style's saturation and hue character;
- one **tone curve applied to every channel** of that, as the camera's own image
  processor does. Applied per channel (not to brightness with the colour carried
  along), a bright colour rolls off towards white and an exposure change moves a
  colour along the path the camera would have, which is what makes a small
  exposure edit look like the camera rather than like a different picture.

The camera's JPEG is corrected for lens distortion and the RAW is not, so the JPEG
is **registered onto the RAW** (scale, shift, two radial terms) first, and only flat
areas are compared, so sharpening does not pass for tone. The matrix is polished
by Gauss-Newton steps on the error in the picture itself: Highlight Tone Priority
makes the top of Canon's curve nearly flat, and fitting through the inverted curve
there pulled the matrix well off on bright frames.

**The camera's colour is pooled per camera and Picture Style; its tone is
per frame.** One frame is a poor witness to its camera's colour — a stage of black
clothes and warm wood holds too little to pin a matrix down — so every RAW that is
opened adds what it teaches to a pool for its camera and its style settings
(Picture Style, contrast, saturation, colour tone, ALO, HTP, read with `exiftool`),
kept in `camera_profiles` beside the thumbnail cache. The tone is different: Auto
Lighting Optimizer lifts a dark or high-contrast frame more than a bright, flat one
(on an R6 II bracket, the mid-tones of the frame two stops under were 0.08 of full
scale brighter), so no one curve fits every frame. A frame with a JPEG gets the
pool's colour matrix and **its own tone curve**, which is exact at the exposure it
was shot at; an **exposure edit moves the curve towards the pool's** over the first
stop, because at another exposure the camera would not have used this frame's
curve, and the pool has seen it at many. A RAW with no JPEG takes the pool's colour
and tone. A frame whose colour the pool does not describe as well as its own fit
(another style that the metadata did not tell apart) keeps its own look and stays
out of the pool; a fit far from the camera's picture — a different crop, a preview
of another exposure — is discarded and the frame is rendered flat.

**Other cameras (Ricoh GR II and the like).** Nothing in the look is Canon-specific:
it learns from whatever JPEG the camera wrote. A Ricoh GR II writes `R0000001.DNG`
and `R0000001.JPG` into `DCIM/100RICOH`; add `".dng"` to `import_extensions` in your
`config.yaml` (it is in the default and the example, but a config file that lists the
extensions overrides them) and the card imports, pairs the two files as one frame and
develops the DNG with the camera's JPEG as its reference. Which pool a file joins is
decided by the camera's colour data and the rendering settings it records: for Canon
the Picture Style, contrast, saturation, colour tone, Auto Lighting Optimizer and
Highlight Tone Priority; for other makes also the Ricoh Image Control, toning, hue and
highlight correction, Sony's Creative Style and Fujifilm's film simulation, read from
the RAW or, failing that, from the JPEG beside it. A Canon file's pool key is
unchanged. A RAW with no embedded preview takes its thumbnail from the JPEG beside it.

Before trusting a new camera, run `python -m tools.camera_profile inspect <files>` from
`backend`: it prints the settings the pool is keyed on, whether the file has a colour
matrix (without one, highlights are rebuilt whole instead of channel by channel), the
sensor's levels, the JPEGs it could learn from and whether they are the same shape as the
RAW (a crop mode makes them differ and the look is then not learned), and the lens setting.
Then `learn` and `report` as for any camera.

**The develop view says, quietly, where the look came from.** Under the Look
heading, one small grey line: "from 12 frames of this camera" once its pool is
established; "still learning (n frames)" while a camera, lens or style is new and the
look rests on its first few frames; "from this frame alone" when the camera's other
frames did not describe it (another style) or the camera cannot be told apart from
others; "none for this file" when there is no matching JPEG to learn from. The
backend sends it with the preview (`X-Camera-Look-State` and `X-Camera-Look-Frames`).
When more frames would help, the line adds one sentence and a link: "This frame may
sit away from the camera's own JPEG" (the whole picture is far from the JPEG), or "No
other frame here looks like this scene yet, so exposure edits are less certain". The
link, "Learn from this folder", opens every RAW of the folder in the background
(`POST /api/camera-look/learn`, progress from `GET`) and redraws the frame when done.

**Exposure edits take their tone from frames of the same kind of scene.** One camera
and style does not have one tone curve: Auto Lighting Optimizer lifts a dark, contrasty
scene much more than a bright, flat one. Each pooled frame keeps a signature of its
scene (how its brightness spreads with the exposure taken out, and how colourful it
is); an edit moves towards the tone of the pool's frames within `SCENE_RADIUS` of the
frame's scene, and keeps the frame's own tone when the pool has none like it. Colour is
still pooled across scenes.

No learning step is needed for any of this: frames teach their camera as they are
opened, and `tools.camera_profile learn` only does that in a batch.

**A Picture Style's hue character is a third stage.** A matrix is linear: it can
raise every colour's saturation or turn every hue by one angle, but not pull greens
towards yellow and leave reds alone, which is what Canon's Standard and Neutral
styles do. After the curve, the look applies a small correction in OkLab: chroma,
hue and lightness are each moved by an amount that varies smoothly with hue (three
Fourier harmonics, and the same again scaled by lightness), fading to nothing
towards grey. It is fitted to the pool with the matrix, and each frame then adds a
shrunken correction of its own (the camera's treatment of a bright green differs
from frame to frame with Auto Lighting Optimizer, as its tone does). On a bright
garden frame the rendering had the right brightness overall but greens 0.06
(OkLab lightness) too light and blues 0.09 too dark; with this stage the frame goes
from ΔE 4.5 to 3.2 against its JPEG, and exposure edits of the bracket's ends from
3.4–4.6 to 2.5–2.8 (a five-frame pool). With a large pool the gain is smaller.

**The lens's vignetting is corrected as the camera corrects it.** Lenses are
darker at the corners, a stop or more at the wide end of a zoom, and Canon's JPEG
engine brightens them back (with Digital Lens Optimizer on, whatever the
Peripheral Illumination setting reads); a RAW decode does not. Uncorrected, a
global tone curve has to split the difference — on R6 II frames at 24 mm the
middle came out too bright and the corners far too dark. The correction (a
brightening of a·r² + b·r⁴ stops from the centre) is learned from each frame's
JPEG together with its look, kept only if it brings the look closer to the
camera, and applied to the linear frame, so edits, crops and the loupe all see
the corrected picture. Measured: about +1.0 stop at the corners at 24 mm and +0.4
at 72 mm, and a third less tone and colour error on wide-angle frames.

The camera's correction is a fixed function of lens, focal length and aperture, so
one frame is a poorer witness to it than the frames of that setting together: the
search overshoots on the brightest frame of a bracket (+1.33 stops where the camera
applied +1.0) or finds nothing, and was the largest remaining error on those frames.
Each frame's finding joins a pool for its lens setting (camera, lens, focal length
and f-number from `exiftool`, kept in `lens_*.json` beside the colour profiles),
and once three frames agree, the pool's median correction is used for every frame
of that setting, a frame's own search standing only when there is no pool. On the
R6 II bracket this took the brightest frame from no correction found (coarse 2.83) to +0.98
(coarse 1.92); the tool's `learn` and `report` say which a frame used.

**The lens's distortion is corrected as the camera corrects it, when asked.** The
camera straightens its JPEG for the lens's distortion and the RAW is not
straightened. Learning the camera's look already registers the JPEG onto the RAW
with a lens model (a shift, a scale and two radial terms). In the develop view,
**Lens corrections → Distortion: Camera's correction** keeps that registration
(measured at 1536 px, once per file) and draws the RAW through it.

- **No lens database.** The correction is right for any lens, focal length and
  camera, because it comes from the photo's own JPEG.
- **Measured** on an R6 II with the RF24-105 at 24 mm: the corners move by 1.4% of
  the half-diagonal, about 85 px at full size. A developed frame's corners sit
  25 px from the camera's JPEG at 1600 px, and 0.3 px once corrected.
- **By hand:** the **Distortion** slider straightens a barrel (right) or a
  pincushion (left) on top, enlarging the picture just enough to keep its corners.
- **Where it runs:** after noise reduction and before the spots, so masks, spots
  and the crop are placed on the corrected picture. The loupe cuts its 1:1 patch
  from the corrected picture too.
- **Off by default.** Turning it on moves things near the edges slightly, so masks
  and spots drawn before then may need nudging.
- **Chromatic aberration:** lateral fringing needs no control. The decode's colour
  smoothing (below) removes it, and what is left on the same frame measures 0.0 px
  between the channels.
- **Sidecar:** kept as `crs:PhotoolLensProfile` and `crs:PhotoolDistortion`.
  Lightroom's `crs:LensProfileEnable` names a profile from its own database.

**Transform and Upright straighten what leans.** The **Transform** group, after
Lens corrections, has three controls:

- **Vertical** tilts the picture plane about its horizontal axis. Left straightens
  verticals that lean in towards the top (a building or a stage shot from below),
  as Lightroom's does.
- **Horizontal** does the same for a wall shot from one side.
- **Aspect** undoes the squat look a strong correction leaves.

The picture stays centred and is enlarged only enough to keep its corners. It is
drawn through the same warp as the lens correction, so the loupe, masks, spots and
exports follow it. Like Level, it belongs to the frame and is not carried by
presets or Reset.

**Auto: verticals** finds the tilt and the level that make the photo's own
vertical lines vertical. **Auto: level** levels its horizon. Both put the answer
on the sliders, where you can see it and change it.

- **Finding the lines:** OpenCV's line segment detector, letting go of lines
  through busy texture. With edges and a Hough transform, a tree beside a building
  produced 261 "lines" in its leaves against 16 on the building.
- **When Auto changes nothing:** the lines it would set upright have to agree.
  Tested on photos found with the library's smart search, buildings, an auditorium
  and a street of houses agree within 0.4–0.9° once corrected, and they are
  corrected. A portrait, a band group under palm trees and a guard huddle stay at
  1.3–1.7°: their "verticals" are arms, trunks and flagpoles. Those are left as
  they are, with the reason shown.

Kept as `crs:PhotoolTransform*`; agents use `develop.upright`.

**Pink highlights are rebuilt in the camera's own colour space.** A sensor clips
one channel at a time. White balance then multiplies the channels by different
gains, so a pixel whose green has saturated still has a red and a blue that kept
climbing: it comes out redder and bluer than the scene was, which is magenta — the
pink in a flash-blown face, or on a sheet of music under a stage light.
`adjust/clip.py` fixes it, and `export/engine.py` runs it **before** the colour
matrix (the frame is decoded in camera space, repaired, and only then taken to sRGB
with the matrix LibRaw would use). After the matrix, a clipped channel is mixed into
the other two and nothing can be repaired on its own.

1. **Find where each channel really saturates, from the data** — a spike of pixels
   piled at one value — rather than trusting the file's white level. (A Canon R6 II's
   red saturates below its green and blue.)
2. **Replace only the channels that clipped.** The others hold real data and are
   kept: on a blown sheet of music blue never clipped, and the printed notes come
   back with it. The colour is borrowed from nearby pixels that are *bright* and
   unclipped — the same surface, not the dark stand or the warm stage next to it —
   and falls back to neutral when it disagrees with the channels that kept their
   data. A pixel clipped in every channel takes its colour from its partly clipped
   edge.
3. **Smooth the colour, not the brightness**, over a few pixels, as the camera's JPEG
   engine does: it removes high-ISO colour speckle and the thin coloured lines
   lateral chromatic aberration draws along a bright edge.

It does not invent detail where every channel clipped: there the result is a
smooth patch of the right colour. A RAW whose colour matrix LibRaw does not expose
(some DNGs) is repaired after LibRaw's own conversion, a whole pixel at a time.

**Measuring the look offline.** `python -m tools.camera_profile` (run from
`backend`) does on a folder what the app does as frames are opened, and says how
close it comes:

```bash
python -m tools.camera_profile report ~/Pictures/2025-05-10 --json report.json
python -m tools.camera_profile learn ~/Pictures/2026-10-08   # add to the app's pools
python -m tools.camera_profile health --render              # every kept look, worst first
python -m tools.camera_profile list
```

`report` measures every RAW's look against its camera JPEG (colour error as OkLab
distance x100, about 1 at the edge of visible; brightness error as a fraction of
full scale) without touching the stored pools. Frames shot as an **exposure
bracket** are found from their capture settings and time, and each is moved to the
bracket's middle exposure with the exposure control and compared with the camera's
JPEG of that middle frame — whether a small exposure edit looks like what the
camera would have made, asked of real data. `learn` adds frames to the pools the
app uses, and says for each frame which pool it went to and, when it was kept out,
the colour errors that decided it; `list` shows the pools. A profile folder that
refuses writes is reported once and `learn` stops (pass `--profiles` with a folder
you can write to); exposure settings are read with rawpy, or with exiftool where
an older rawpy cannot give them.

**A broken look is seen, not only a wrong one.** Mean error against the camera's
JPEG cannot see posterization: a tone curve flat across a stop of skin tones and
steep above it draws faces as flat patches and still matches the camera's picture
on average (on one stage frame, better than the right look did). Two measures
see it. `even` is the curve's own shape, how evenly it rises through the
mid-tones (about 0.5 for a camera's curve, near 0 for a plateau). `banding` is the
picture's: tone band by tone band, how much of the camera JPEG's local contrast
ours keeps, as the spread in stops between the flattest band and the steepest
(0.2 to 0.3 for a look that draws like the camera; 1 or more posterized).
A plateau marks a look OFF, since it does harm wherever an edit moves tones into
it; `banding` says whether it shows in the photograph as it is (SHOWS). Banding
alone is not a fault: a camera JPEG with heavy noise reduction is softer than any
develop. `report` shows `banding` for each frame; `health` checks every kept look,
worst first, read-only (`--render` adds `banding`, by drawing each photograph with
its kept look). Agents ask the same of one photo with `develop.look_health`.

**RAW+JPEG pairs learn from the JPEG beside the file.** When a RAW has a JPEG with
the same name, that full-size rendering is the reference for the camera look, ahead
of the preview embedded in the RAW; one that does not match (a different crop) is
passed over for the next. A black-and-white image control is learned too — the
colour step may remove colour entirely.

**A photograph's look is kept, and the next one is got ready.** Learning a frame's
look was most of the time it took to open (6.7 s of 8.6 s on an R6 II frame, against
0.1 s for LibRaw's decode), and it was learned again after every restart. A settled
look is now kept beside the pools (`camera_profiles/looks`), tied to the file's size
and modification time, and learned again only once its camera's pool has doubled.
While one photograph is open, the ones either side of it are decoded in the
background, and this one at full resolution for the loupe. Measured: reopening after
a restart 9.8 s → 3.3 s, stepping to the next frame 0.9 s, the first loupe 6 s → 0.2 s.

**Noise reduction is a learned denoiser on the RAW's own data.** The Noise reduction
sliders — **Luminance** (the grain) and **Colour** (the speckle and blotches), 0..100
each — run RawRefinery's TreeNet denoiser ([RawForge](https://github.com/rymuelle/RawForge),
MIT) on the graphics card through ONNX Runtime (DirectML on Windows). It works on
the sensor's mosaic, as RawForge feeds it, and its result is sampled back into a
mosaic at the file's own levels and decoded exactly as the original is, so the
highlight repair, the camera's look and the lens correction are untouched. It runs
once per photograph (~10 s for 24 MP on an RTX 3070) and the mosaic is kept
(`thumbnails/denoised`, 10 GB at most); the sliders only blend the two decodes, so
moving them is as quick as any other. The model is told the photograph's ISO, but
never more than 12800: told 25600, it smooths an R6 II frame into flat patches.
The develop view draws the photo without it until it is ready; an export waits for
it. RAW files only. It replaces the export panel's old Denoise (Intel OIDN on the
finished 8-bit picture), whose "Auto" never ran and whose other settings barely
changed a high-ISO frame.

**Sharpening has Lightroom's four controls.** A RAW decode is soft: the camera
sharpens its own JPEG, and the camera look learned from that JPEG deliberately does
not copy the sharpening. The Sharpening group (after Noise reduction) has:

- **Amount**, 0–150. 0 is off; 40 is Lightroom's default for a RAW.
- **Radius**, 0.5–3 px.
- **Detail**, 0–100. Low holds strong edges back so they get no halo; high
  sharpens every edge as hard as fine texture.
- **Masking**, 0–100. It keeps the sharpening to edges, so skin and sky stay
  smooth.

It runs on brightness only, so edges get no coloured fringes. It runs after the
noise reduction, so the grain that was taken out is not put back, and after the
colour, before the grain and the crop.

The radius is in pixels of the whole photograph. A smaller picture (the preview,
a 2048 px export) has its radius scaled down with it, but never below 0.7 of its
own pixel. So a web-sized export is still visibly crisp instead of having the
sharpening averaged away. Judge the amount at 1:1 with the loupe. Kept in the
sidecar under this app's own `crs:PhotoolSharpen*` names: Lightroom writes
`crs:Sharpness 40` into most sidecars it touches, so reading that as this app's
sharpening would quietly sharpen every one of those frames.

**Library thumbnails show the edit.** The grid and the cull view normally draw the
JPEG the camera embedded in the file. A frame that carries an edit is rendered
through this same pipeline at thumbnail size and cached under its own key, beside
the plain thumbnail and never over it: change the edit and it is redrawn, clear it
and the camera's rendering is straight back. Only edited frames pay for this, a
couple at a time.

An edit worked out once is worth keeping twice over.

- **Presets** save the controls under a name and put them back later.
  Applying one replaces the whole edit set — a preset is a complete answer to "how
  should this frame be developed", not a tweak on top of whatever was there.
- **Copy and paste** carry an edit between frames: `Ctrl+C` on the frame that looks
  right, then `Ctrl+V` on the one that should look the same. A whole grid selection
  can be pasted onto at once.
- **To burst** puts the edit on the frame being developed and on the rest of its
  burst, whose frames were shot seconds apart under one light. Every frame of the
  burst is written, basket or not: the edit belongs to the photograph, so choosing
  what to export and choosing how it should look are separate decisions and only
  the first one is the basket's. The group is the measured burst, not the filtered
  grid view, so a frame hidden by a filter still gets the edit.

**Paste and match.** Copy decides *what* is carried (its ▾ ticks the sections);
Paste's ▾ decides *what happens to each frame* it lands on. The choice is kept on
the device and used by Paste, To burst, the grid's Paste edit (which has the same
choice beside it) and `develop.paste` (`paste_as`):

| Paste | Each frame | For |
|---|---|---|
| **As copied** | gets the edit exactly | frames shot in the same light at the same exposure |
| **Keep each frame's exposure & white balance** | keeps those three, takes the rest | frames already corrected one by one |
| **Match light to the copied frame** | has its exposure and white balance brought to the copied frame's, from the best evidence it has (below), as ordinary Exposure, Warmth and Tint | a set where the meter drifted, the light changed, or the photographer moved |
| **Match the look to the copied frame** | is matched to the copied frame as developed (Match a look, at 75%) | frames of the same shoot only |

Measured on one burst with the meter off by ⅓ or ⅔ stop: the capture settings took
a frame back to within 0.02 coarse ΔE of where it should be (camera look 0%; 0.11 at
100%), a solve from the picture's brightness to 0.53, Match to 0.90, and a plain
paste stayed at 1.96. The picture check is there because the capture settings are
only right while the light is: on one stage the photographer opened up ⅔ stop for a
darker scene and kept the pasted exposure, and correcting by the settings would
have undone that. Within a shoot, frames shot at the same settings differ in picture
brightness by up to 0.21 stop in nine pairs of ten (the subject moving), and where
the settings changed, half the pairs agreed within 0.26 stop and the rest disagreed
by 0.5 to 1.2.

**How Match light decides, frame by frame.** The capture settings (shutter, ISO,
aperture) are exact while the light stays the same, so they are the answer
whenever they can be checked and pass. The check is what the two frames share:
the same people (from the face groups, each person compared only with themselves,
so complexion does not matter) and points of the scene matched between them (the
backdrop, the floor, an instrument), compared in the RAW's linear light. That
works at any angle and focal length. When the shared things agree with the
settings, the settings are used; when they disagree, the light itself changed
(a cue, a cloud, a re-metered stage) and they set the exposure; at least two
shared faces or thirty matched points also set the white balance. A frame that
shares nothing with the copied one falls back to checking its settings against
its picture's brightness, and is left as pasted when those disagree. Each frame's
note says which: "+0.32 EV from capture settings, confirmed by 3 shared faces", or
"the light changed: −0.41 EV and warmer from 2 shared faces". The grid names the
frames where the light changed: a lighting cue may be a mood worth keeping, and
undo or a paste As copied puts any of them back.

Measured on a stage shoot, the same person's skin against the copied frame's
(OkLab ΔE×100, that person left out of the measurement): where the capture
settings changed between the frames, 4.5 this way against 5.9 by the settings
alone and 7.2 for a plain paste; where they did not, about 4 for all (the same
face under stage light from another side). Scene points alone hardly helped skin,
since a backdrop is lit differently from the people in front of it; faces are
weighted ten to one. A frame zoomed 1.33× and shifted half a stop and warmer was
brought back within 0.05 stop on every channel. Between an R6 II and a 5D Mark II
at one event, capture settings and anchors both halved the difference a plain
paste left (9.4 → 5.0); a full colour matrix between the bodies overfitted.

**Match a look works within a shoot, not across shoots.** It makes each channel's
brightness spread look like the reference's, so it cannot tell a picture's look
from its subject. Carried to frames of other shoots it ended further from the
reference's edit than leaving them unedited in 174 of 210 comparisons: a gym floor
matched to a sunlit parade turns pink, a concert is lifted and washed blue. To put
an edit on another scene, copy and paste it; a pasted edit is applied the same way
whatever is in the picture. The panel says so when the reference is from another
folder, and when camera look, exposure or white balance change after a match (the
curves were fitted to the frame as it was), it offers to match again.

**Camera look and exposure.** With the camera look at 100%, the exposure slider
also moves the tone curve towards what the camera would have drawn at that
exposure (adjust.look.curve_for), so an exposure change is not only an exposure
change. At 0% it is, which makes a set easier to keep consistent by hand.

**The edit belongs to the photograph, and it is non-destructive.** Nothing is
written to the image itself. It lives in two places that are kept in step:

- the **XMP sidecar**, which is the durable copy — it travels with the
  photographs, another editor can read it, and a catalog rebuilt from scratch
  recovers every edit from the files;
- the **catalog**, which is the index over it — so drawing a grid of two thousand
  frames does not mean opening two thousand sidecar files.

A save writes both. `index_folder` reconciles them: a sidecar carrying an edit
that the catalog disagrees with wins, which is how a rebuilt catalog — or a change
made in another editor — arrives. A sidecar *without* an edit is not treated as a
deletion, only clearing one is, so a sidecar that failed to write never silently
discards work the catalog is holding.

Editing happens in a **full-screen develop view**, opened with `D` from the grid
or the cull view and closed with `Esc` or `D` again. It exists because the adjustment is
subtle: a dozen controls on a 200px thumbnail are a number in a panel and nothing
on the picture. The frame is rendered at 1600px, `\` holds the original for
comparison, and a **split** puts the original and the edit side by side with the
same frame under both halves. A luminance **histogram** is drawn from the picture
rather than from the numbers — the original in grey behind the edit — because a
tonal change is far easier to read as a curve that moved than as two
similar-looking photographs. `←`/`→` walk the folder without leaving the view,
`A` runs Auto, `0` resets, and `B` copies to the burst. Grid frames that carry an
edit are marked `✎`, and the filter bar can show only those.

The preview is fast because the frame is decoded once. The first request for a
photo at a given size decodes it to linear light and keeps it in memory, with its
base/detail split (the three most recent, keyed on the file's modification time so
an edit on disk is never served stale); every slider move after that rebuilds the
curve and re-applies it, a fraction of a second at 1600px. Edits are saved on
a short debounce after a slider moves, so dragging one control is one write and not
a hundred, and reopening a frame puts the sliders back where they were left.

The same edit reaches other editors through the sidecar as `crs:*`, so Lightroom
opens the frame with the same numbers. Values that carry over exactly are
written as themselves — `crs:Exposure2012`, `crs:Highlights2012` and the rest.
Warmth cannot: Lightroom's `crs:Temperature` is an absolute colour temperature in
kelvin, while this is a relative channel shift, so it is written as
`crs:TemperatureShift`, a tag Lightroom will not read but which does not pretend to
be something it is not. Tint is worth converting, since Lightroom's scale is the
same axis at ±150, so it is written at 1.5× — and *also* written exactly under
`crs:TintShift`, because the 1.5× conversion is a whole number only sometimes:
7 becomes 10.5, which rounds to 10, which reads back as 6. A sidecar that is the
durable record of an edit cannot lose a unit every time it is opened.

Writing an edit replaces **only this app's own attributes**. A crop, a curve or a
lens profile another editor put in the same `crs:` block is left where it is, and
clearing an edit takes this app's settings off without taking someone else's with
them — a shared file that loses half its contents when a slider moves is worse
than no sidecar at all.

## Styles: one look on any camera

A camera look (above) is what one camera's engine does to its own files. A
**style** is the same thing kept under a name and put on *other* cameras' files:
the Ricoh GR's Positive Film on a Canon CR3, Canon's Neutral on a Nikon NEF, or
the rendering of your own edits — a RAW with an edited JPEG of the same name
beside it is a frame that teaches that edit (`adjust/styles.py`).

**In the develop view**, under Look: **Style** picks one ("Camera's own look" is
none), **Style fade** dials it back towards the camera's own look, and **+ New
style from this folder** learns one. Learning reads up to 40 of the folder's RAW
frames, spread over the shoot, that were taken with *this frame's* camera and
rendering settings, each against the JPEG beside it or, failing that, the one
inside it. It never touches the camera pools, so learning your edited exports
does not teach the app that your camera looks like your edits. A style is an
edit setting like any other: it is saved per photo, copied and pasted, written to
the sidecar as `crs:PhotoolStyle`, and shown in the library thumbnails. Styles
are kept in `styles/` beside the catalog.

**Why it translates — the middle space.** Every RAW, from any maker, is decoded
into one space before a look sees it: the sensor's values white-balanced with
the camera's own gains and taken through *that sensor's* colour matrix (LibRaw's,
from Adobe's calibration of the sensor against CIE XYZ) into scene-linear Rec.709
primaries, with headroom above white and the lens's vignetting corrected. That
describes the scene's light rather than the camera's numbers, so a look fitted
on a Ricoh's decode is "what Positive Film does to light" and applies to a
Canon's decode as it stands.

**What does not translate is where middle grey sits.** Makers calibrate ISO
differently, and Canon's Highlight Tone Priority records a stop darker on
purpose. A tone curve is a function of brightness, so at the wrong level it is
the wrong curve. Each frame is therefore **anchored**: one exposure factor,
found so that the style puts the frame's middle tones (the 35th-65th
percentiles) where the frame's own camera put them in its JPEG. The style brings
the contrast, roll-off and colour; the frame keeps the brightness its
photographer chose. A RAW without a JPEG to learn from is anchored to the
style's own typical brightness. Measured on a Canon 5D Mark II CR2 and Nikon
D3S and D4 NEFs: a style on its own camera changes nothing (anchor 0.00 stops);
moving a frame's raw numbers a stop (another calibration) gives the same picture
after anchoring; and a Canon style on a Nikon frame keeps the frame's median
brightness to within 0.01.

Two limits. A sensor's matrix is one 3×3 fitted under one light, so two cameras'
decodes of a very saturated colour can differ by a few ΔE before any style is
applied. And a style learned only from one kind of scene carries that scene's
choices — learned only from night shots, it will lift shadows.

## Culling a burst

A school concert produces forty frames of the same three seconds. Choosing between
them is the slowest part of culling, and it is the part the measurements were
built for.

Two frames are the same shot when **both** of these hold:

- their perceptual hashes are within 12 bits of each other — the same composition
  differs in a few bits, a different one in many; and
- they were captured within 10 seconds of each other — `burst_max_gap_seconds` in
  `config.yaml` changes that window.

The hash alone would merge a static stage shot ten minutes apart; the time alone
would merge every frame of a pan. Together they say *the same shot, held down on
the shutter*.

Capture time comes from the file's modification time, which is what the
measurement row already stores — `photos.capture_date` is only an EXIF cache and
is empty for anything whose EXIF has not been read.

Grouping is **seeded, not chained**. Each frame gathers look-alikes within ten
seconds *of itself*, so a slow sequence of frames nine seconds apart cannot
collapse into one ten-minute group; it becomes a chain of small ones.

A few rules keep the result trustworthy:

- Grouping honours the grid's pairing filter, so a CR3 never groups with its own
  JPG and a burst is always a set of frames you can actually see.
- Sharpness is the variance of the Laplacian on a same-size thumbnail, so it is
  only comparable between frames of one folder — which is exactly how it is used.
- A group whose frames are all rejected is not offered: it is already culled.
- Frames that have not been measured yet are skipped, and the toolbar says how
  many, so a folder still working through the queue shows fewer bursts rather than
  pretending to be finished.

**In the grid**, every frame of a burst carries a small `⧉N` badge. **In the bulk
row**, `⧉ Keep sharpest of N` runs the whole folder at once: it keeps the sharpest
frame of each burst and rejects the rest, after telling you exactly how many
photos that is. **In the cull view**, a strip along the bottom shows the other
frames of the current burst, marks the sharpest one, and `Shift+R` rejects the
rest. Press `C` there to lay the whole burst out side by side — see
[Comparing frames](#comparing-frames).

Keeping is implicit — the frame you are looking at is never touched by
`Shift+R`, and the keeper is whichever frame the measurements say is sharpest,
which may not be the one on screen. Nothing is deleted and nothing is hidden: the
tool only sets the reject flag, which is written to the XMP sidecar like any other
flag, so `U` on a photo undoes it and Lightroom sees the same decision.

The endpoints are `GET /api/catalog/bursts/<folder>` (same filter parameters as
the photo list, plus `?max_gap_seconds=` to try a different window),
`POST /api/catalog/bursts/reject-rest` for one burst, and
`POST /api/catalog/bursts/reject-duplicates` for a whole folder. Every response
says which window produced it.

That window is configurable because the right number depends on how the camera was
being held — and because it should be measured before anything is made the
default. Across the 40 folders of one 26,000-photo library that hold more than 100
photos each: 10 seconds finds 280 bursts covering 726 frames, 60 seconds finds 283
covering 772. Five folders change at all. The fast ones barely move, or merge two
bursts into one; the two where a wider window genuinely finds more are slow,
deliberate shoots (2016-11-19: 6 bursts become 11). Ten seconds stays the default.

### Comparing frames

The strip answers *which frames are the same moment*. Comparing answers *which one
of them do I want*, and that needs them at a size you can judge.

`C` in the cull view — or the **Compare** button on the strip — replaces the
single frame with the whole burst: two or three across, four frames as a 2×2
rather than a row of four, in capture order, each labelled with its filename, its
rating if it has one, and the suggested keeper marked. Clicking a frame, or `←`
`→`, chooses which one is bordered in white.

That white border is the whole design. Compare is a layout, not a second viewer:
the bordered frame is the photo the top bar, the star keys, `P`, `X`, `B` and `T`
all act on, so there is one idea of "the current photo" rather than two that could
disagree. `Esc` leaves compare and stays in the cull view.

`Z` switches every frame to a **1:1 centre crop** — the same coordinates of each
frame at full size, filling its cell. That is the part that makes comparing worth
doing: at fit-to-window, a 2000px preview is downscaled enough to hide the focus
difference between a frame that missed and one that nailed it.

`Shift+R` means something different here on purpose. In the single view it keeps
whichever frame the measurements prefer; in compare you have just looked at all of
them, so it keeps **the frame you are pointing at** and rejects the rest.

## Working through a folder

Culling a two-thousand-frame school event is a session, not a click, so the tool
keeps track of where the session stands.

**Undecided only** is the review pass: the frames that carry no decision yet — no
star, no flag, no colour label — and nothing else. Two rules make it usable:

- A **reject counts as decided.** The point is to find frames nobody has judged,
  not frames that were judged well.
- An **event tag is not a decision.** A tag says which event a photo belongs to,
  not whether to keep it, so labelling a folder does not empty the review pass.

Like every other filter it is pair-aware: if the CR3 of a pair was rejected, its
JPG twin is not shown again asking for a second opinion. It composes with the other
filters — `Undecided only` with `≥ 2★` means *not judged yet but already looks
good* — and the bulk actions act on exactly that set, so "basket the undecided
frames of this event" is one call.

**Progress** appears under the toolbar as a bar and a count: *312 of 1406 decided*.
Counted per image rather than per file, because that is what the grid shows. The
Library cards carry the same figure per folder, so it is obvious which events still
need work.

**Resume.** The last frame you looked at in a folder is remembered, and reopening it
offers `Resume at IMG_2214.CR3` beside the progress bar. It is a button rather than
an automatic jump, because opening a folder should not drop you somewhere
unexpected. If that frame has since been decided and the review pass filtered it
out, the button takes you to the next frame after it — which is where the review
actually stands — and `✕` forgets the position. The position lives in the catalog,
not in the sidecars: browsing position is not a cull decision, and it is the one
piece of state this tool keeps to itself.

The endpoints are `GET` and `PUT /api/catalog/folder-state/<folder>`, plus
`?unreviewed=true` on the photo-list, burst, basket and tag endpoints.

**One camera at a time.** A day shot on two cameras, often by two people, is
usually imported into one folder. When a folder holds more than one camera, a row
of chips above the filters shows each with its count: `All 755 · GR II 415 ·
Canon EOS R6m2 340`. One click narrows the grid to that camera, and the cull view,
slideshow, bursts, hints and bulk actions follow. A RAW+JPEG pair goes by
whichever file knows its camera, and photos with no camera recorded have their own
*Unknown* chip. Opening another folder starts on *All* again. The endpoints take
`?camera=` (empty for unknown).

**Less important on this device.** A library shared by a family holds everyone's
shoots, and not all of them matter to everyone. In a folder, **↓ Less important
here** marks the whole folder for this device. With a camera chip selected, **↓ GR
II photos less important here** marks only that camera's photos in that folder,
for example someone else's camera on a shared day. Each device keeps its own list,
so one person's side event can be another's main one. Nothing is hidden or
deleted:

- A demoted folder is dimmed and sorted to the end of its year in the folder
  list. A card with one camera demoted says which.
- In a search, demoted photos come after the rest, and the result line says how
  many there are.
- On this day and Year in review leave them out for this device.
- The chip shows `↓`, and *restore* takes it back.

The endpoints are `GET` and `PUT /api/library/demoted` (`{folder, camera}`, with
camera `*` for the whole folder and `""` for photos with no camera recorded).

**Searching in a folder.** With a folder open, the search box searches that folder
only, shown by an `in 2026-05-22 ✕` chip in front of it. The grid narrows to what
was found and stays a folder: cull, bursts and bulk actions work on the matches,
and Esc or *Clear* brings the folder back. Click the chip to search the whole
library instead.

## Quality hints

Two warnings sit on top of the measurements, both of them marks on a frame and
neither of them a decision. A soft frame can be exactly the photo you want, and a
blown-out sky can be the point of the picture — the tool cannot know your intent,
so it says where to look closer and leaves the rest to you. Nothing here rejects,
ranks, or hides anything, and no flag is written.

**Soft frame** is measured against the folder, not against a number. Sharpness is
read from a 400px thumbnail, so the value only means something next to other
400px thumbnails of the same shoot: a telephoto crowd shot and a wide-angle
portrait are not comparable, three frames of the same moment are. So the folder's
median sharpness is the reference, and a frame below 40% of it is marked. On real
event folders that flags around one frame in eleven overall, and anywhere from one
in fifty to one in five depending on how steady that particular shoot was.

**Blown highlights** is absolute, because a fraction of pixels pinned at white is
a fact about the file rather than about its neighbours: 5% of the frame at white
is marked. That is deliberately high — at 2% it flags a fifth of a typical
library, most of it harmless specular highlights.

Both appear as a small `soft` / `☀` mark on the thumbnail, as a count in the bulk
row (`⚠ 12 soft · 3 blown out`), and as a line in the cull view while the frame
carries one. The count only appears when there is something to say, and a folder
that is still working through the measurement queue says so instead of pretending
to be finished.

Hints are derived when they are asked for and never stored. The measurements are
the expensive part and they are already on disk; the interpretation is a couple of
numbers in `backend/analysis/hints.py`, and changing them changes every folder at
once rather than requiring anything to be re-measured. A folder with fewer than a
dozen measured frames gets no soft-frame marks, because there is no trustworthy
median to compare against.

The endpoint is `GET /api/analysis/hints/<folder>`, with the same filter
parameters as the photo list, so the marks always describe what the grid is
showing.

## People

The **People** tab finds the faces in your photographs and groups them by who
they are. Everything happens on this machine: OpenCV finds each face and turns it
into a vector of numbers (512 with ArcFace, the default; 128 with SFace), the numbers are grouped locally, and your photographs never
leave the library. A group becomes a person the moment you name it — and the name
is written as an ordinary event tag, so filtering, tagging, sending to the export
basket, and `dc:subject` in the sidecar all work on a person without any of them
knowing that people exist.

### What changed in version 5

The first version described faces with SFace at 1024px, and on a family library
it was weakest exactly where it matters: siblings landed in one group, a child in
profile became a second person, and most faces across a room were too small to
use. Five changes, all measured or tested:

- **ArcFace instead of SFace** (`face_model: arcface`, InsightFace's ResNet-50
  trained on WebFace600K, run by OpenCV's DNN module — no new package). On a set
  of labelled portraits, the gap between the least-alike pair of one person and
  the most-alike pair of two people went from 0.35 to 0.50, and two strangers
  now sit around 0 rather than 0.1. It is a one-time 290 MB download;
  `arcface_lite` (MobileFaceNet, 15 MB) is the small option, `sface` the old one.
  The InsightFace weights are released for non-commercial use, which a family
  album is. Each face is described as it is and mirrored, and the two added.
- **The 2000px preview, not the 1024px rendering** (`face_scan_size: preview`).
  It is the rendering the cull view already keeps, and it doubles every face: a
  face 25px across at 1024 is 50px here, over the floor.
- **Quality.** Every face gets a score from its size, its pose (how far the nose
  sits off the line between the eyes), its sharpness and the detector's
  confidence. Clear faces are grouped first and define people; a poor face can
  only join someone who is already clear, at a slightly stricter bar, so blurred
  profiles no longer make one-photo "people". The person's picture is their
  clearest face, not merely their largest.
- **A person has several faces.** A group keeps up to six *exemplars* that are
  unlike each other — frontal, profile, a year older — and a face can join
  through the one it resembles, as long as it is near the group's average and not
  a stranger to *any* exemplar (which keeps the chain rule below).
- **Your answers are kept.** A face taken out of a group remembers it, so the
  next regroup does not put it straight back (it used to). An unnamed group close
  to a named person is shown as **"Alice? yes / no"**: yes merges, no is
  remembered. And re-scanning with a new model keeps every name: a face found
  again over the box of a named one keeps its person.

Changing the model makes the next scan describe every frame again (the panel
says how many are left); `face_threshold` overrides the ArcFace join threshold
(0.42 against a group's average) if you want groups tighter or looser.

### Where the numbers come from

These were measured for SFace at 1024px, the first version; the reasoning
still holds for ArcFace, at its own threshold.

Two of the numbers were measured on a real library rather than taken from the
models' defaults.

- **A face is looked for in a 1024px rendering, not the grid thumbnail.** At
  400px, one person's face scored 0.61–0.91 against itself across a single burst
  — a range that overlaps what two *different* people score, which is exactly how
  a face-recognition feature ends up calling two children the same person. At
  1024px the same burst scores 0.83–0.98, and across 97 pairs of different people
  sampled from four folders over two years the highest score was 0.41.
- **The join threshold is 0.5** (`face_match_threshold`), between those two
  ranges and nearer the low end. A person who comes out as two groups is a merge
  you make in a click; a group that swallowed two people is a name on the wrong
  photographs, and you would have to find them before you could fix it.

### How a group is decided

A face joins a group when it points the same way as the group's average, by at
least the threshold. Three decisions sit around that one line:

- **The largest faces are placed first.** The order of a greedy pass defines the
  groups, and a 300px face gives a far more trustworthy vector than a 30px one.
- **A group is its average, not its first member.** Comparing against every
  member instead would let a chain of look-alikes join two people: A resembles B
  and B resembles C, so A and C end up together.
- **Faces under `face_min_px`** (40px at 1024, a head filling about 4% of the
  frame) are dropped where they are found, before anything expensive happens to
  them. They could not be grouped anyway, and the size gate costs nothing: a
  detector's box that is never described is a box that was free.

A named group is a seed: regrouping keeps its identity and its faces and can only
add to it. A closing pass merges *unnamed* groups whose averages agree closely,
which is what tidies up the usual fragmentation, but a named group is never
absorbed — a name may well have been the decision to keep two groups apart, and
quietly undoing it would be worse than the fragmentation it was tidying up.

### Scanning, and what it costs

*Find faces in this folder*, or *Whole library*, on the People tab. The pass
follows the catalog rather than the directory, skips any frame that has not
changed since it was last looked at, reports progress as it goes, and stops
between frames when you ask it to. One scan runs at a time; a second request is
refused rather than run alongside the first, because two passes would compete for
the same models and the same thumbnail cache.

**A photograph is looked at once, however many files it has.** A CR3 and the JPG
beside it are two files of one moment, so the pass follows the pairing the grid
uses and reads whichever half that is. Reading both would put every person in the
library into the list twice — and reading both was exactly what the first version
did, on 1380 files where there were 719 photographs.

**The renderings are made a few dozen frames at a time.** Handing a whole folder
to the batch at once writes every embedded JPEG to a temporary directory and
holds them all in memory — gigabytes of it, on a folder of six hundred — and the
exiftool call that does it runs longer than its own timeout, after which every
frame falls back to the slowest path there is, one process each. A chunk keeps
the memory bounded and the bar moving.

At 1024px a face is worth describing only above a size, and **that decision is
made before the description, not after**. Detection is cheap and describing a
face is not: one real frame produced 221 detections, none wider than 28 pixels,
which at one recognition pass each is three seconds spent on nothing at all.
Anything below `face_min_px` is dropped at the point it is found, so it is never
described, stored, or counted.

**The frames are looked at several at a time.** Both models run on one thread
inside OpenCV, so a serial pass over a folder held 0.99 of a core on a
twenty-core machine — a scan that was working correctly looked like it was
ignoring the CPU. Frames are independent, so a chunk of them is handed to a
small pool; the results are then stored on the scan's own thread, which keeps
the SQLite write lock uncontended and every face row landing from one place.
The ceiling is the part of each frame that has to hold the GIL, not the core
count, so eight threads is the most that helps and past that it costs more than
it buys.

Measured on a real library after all four: **about a fifth of a second a frame,
or roughly 90 minutes for 26,000 photographs**, and about 3 GB of cache at 127 KB
a frame. The first version took 1.16 seconds a frame — eight hours — which is what
a doubled frame count, a timed-out extraction and fifteen thousand faces nobody
could use cost between them. Looking at frames in parallel cut the same folder
1.6x while its renderings were being built and 2.2x once they were cached: the
first pass over a library is dominated by rendering, a re-scan by detection. That
cache is evictable like the rest: set
`thumbnail_cache_max_mb` and the least-recently-used renderings go first, at the
price of the next scan having to build them again.

The two models are 39 MB and are fetched into `models/` the first time a scan
runs, checked against a known SHA-256; on a machine with no network, put the two
files there by hand and the panel will stop complaining.
`opencv-python-headless` is the only package the feature adds.

### Fixing the answers

Grouping is arithmetic on photographs of people, so it will be wrong sometimes,
and the controls that matter most are the ones that undo it.

- Click a group's picture to see **every photograph that person appears in** —
  the only view in the app where one person's photos are together, since the grid
  shows one folder at a time.
- **Name** a group to tag its photographs.
- **Merge** folds one group into another: the usual fix for one person split
  across two groups, which happens most often between a frontal view and a
  profile. When both groups had names, the name that loses is taken off those
  files.
- **Detach a group** (✕) takes it apart and takes its name off the photographs;
  **detach one face** does the same for a single frame, which is the fix for one
  photograph of a child joining their sibling's group.

Two limits worth knowing. A camera-shaped detector is not a person-shaped one:
YuNet occasionally finds a face in a dog, and a group of dogs will sit in the list
until you detach it. And a person photographed in profile can come back as a
second group, because a single 128-number description of a face is not the same
number from the side.

### A person's timeline

**◷ Timeline**, on a person's page, lays out their best face for every month
they were photographed (or every year), oldest first, and **▶ Play** steps
through them as a flip book: a child growing up in a minute. Each frame is cut
from its photograph so the face sits in the same place at the same size, so
from one month to the next only the person changes; *Whole photos* shows the
frames uncut instead.

A month's face is its best one, not its first: large and clear first, then a
frame you picked or rated, never a reject. A photograph's month is its EXIF
date or the date in its folder's name. One taken with a camera whose clock was
wrong lands in the wrong month — fix the date and it moves.

## Year in review

The Library's front page lists the years under *On this day*. A year opens a
page that says what that year of photographs was:

- **The numbers.** Photos, days out and shoots, a bar per month, the busiest day
  and the cameras used.
- **The year's best.** About two dozen frames, chosen even when nothing has been
  rated. Picks, stars and edits count first; then named people at their clearest
  and frames sharp for their own shoot. Soft frames and blown highlights count
  against a frame, and rejects never show.
- **Spread across the year.** Each day gets a share that grows with the square
  root of how much was taken, so one weekend of two thousand frames does not
  fill the page. Frames are kept ten minutes apart, or apart by look where the
  time is unknown. Two pictures that smart search describes as the same scene
  count as one.
- **Who was in it.** The named people, most photographed first. A face opens
  that person's timeline.
- **The biggest shoots and the places away from home.** Each opens its folder.
- **⤓ Collage.** Nine of the best, at most one per day, laid out in rows that
  keep every frame whole, under the year as a title, saved as one JPEG to your
  downloads. Tap ✓ on any frame to put it in or take it out (up to 30).

**▶ Play the year** shows the year's best as a slideshow.

**Perspectives.** A family library holds school events, family days and
friends' parties, and not everyone wants the same year back. The switch next to
the year (`2026 · Family ▾`) shows the year through one perspective, and the
numbers, best frames, people, shoots and collage are all worked out from that
perspective's shoots only. The collage is titled with it too.

- **What a perspective is.** A name and the people it is about. People are
  chosen by name, or by a name prefix that covers a group: `BAND*` is every band
  member named "BAND …", including ones named later.
- **How shoots are assigned.** A perspective takes whole shoots, not single
  photos: the food and the venue of a family dinner have no faces and are still
  the dinner. A shoot belongs to a perspective when its people are in at least
  two of its photos and in at least 60% as many as the perspective most present
  there. So a band review full of the band, with the family's child in it, is
  School, while a dinner with that child is Family.
- **Starting points.** *Edit perspectives…* suggests one from each name prefix
  several people share, and a Family of the most-photographed people with no
  prefix.
- **Shared and per device.** Perspectives are shared by everyone using the
  library. **☆ open on this here** makes one the default on the current device,
  so each person's phone opens on their own view.

## Collages

There are four places to start one:

- **A selection.** Select 2 or more photos, in a folder or in library-wide search
  results (`tag:Robin` finds one person across every folder), and press
  **▦ Collage** in the selection bar.
- **The export basket.** It collects photos from any folder and day, so add them
  as you browse, then press **▦ Collage** on the Export tab. A RAW+JPEG pair
  counts once.
- **A person's timeline.** **▦ Collage** starts with nine of their months, spread
  from the first to the latest, on a square.
- **Year in review.** **▦ Print layout…** starts from the frames ticked for the
  year's collage.

A collage holds up to nine photos. Given more, the first nine go in and the rest
wait in a strip below; tap one to swap it in after leaving another out.

- **Size.** A *Story* (9:16, 1080×1920, for a phone), a *Square*, a *4×6*, *5×7* or
  *8×10* print (300 dpi), or a *3:2* screen. Prints start landscape; *Portrait*
  turns them.
- **Layout.** Rows or columns of the photos, in the order shown. Each line of
  photos first takes the height that fits them whole across the page, then the
  lines are stretched together to fill it. That stretch is all the cutting there
  is, and the four layouts offered are those that need least of it, weighed
  against cells coming out very unequal. The evenest layout (nine photos as a
  three-by-three grid) is always one of them. Each choice is drawn as a small diagram,
  and its tooltip says how much is cut away.
- **Cropping.** Each photo is cut to fill its cell, kept around the people in it
  where their faces are known (the bigger faces decide), otherwise around its
  middle.
- **Space between, border and background** (white, cream or black), and the
  order, which ◀ ▶ change and ✕ shortens.

The page shows a small render of exactly what *Save* will make. Photos are drawn
from their previews, with edits. The endpoints are `POST /api/library/collage/layouts`
and `POST /api/library/collage` with a `format`.

## Searching the library

The box at the top of the Library tab searches every folder at once (or, with a
folder open, only that folder; see [Working through a folder](#working-through-a-folder)). A query is
terms separated by spaces, and a photo is found when every term holds. Results
open in the same grid, cull and develop views as a folder does. `?` beside the
box lists the terms; clicking one runs it.

| Term | Finds |
|------|-------|
| `rating:>=3`, `stars:2..4` | Stars, compared with `=` `>` `>=` `<` `<=` or a range |
| `flag:pick`, `flag:reject,none` | Cull flag; a comma means either |
| `label:red` | Colour label (red, yellow, green, blue, purple, none) |
| `tag:"school play"` | A tag, exactly; quotes for spaces |
| `camera:r6`, `lens:35` | Any part of the camera or lens name |
| `iso:>3200`, `f:<=2.8`, `focal:24..35`, `shutter:<1/100` | Exposure data, as numbers |
| `date:2026-04`, `date:>=2026-03`, `date:2024..2025` | A year, month or day. Frames whose EXIF has no date are dated by their folder name (`2024/2024-01-18`, `20230102_party`) |
| `folder:hawaii` | Any part of the folder path |
| `type:raw`, `type:cr3` | What the photo has: a pair counts as both `raw` and `jpg` |
| `is:edited`, `is:unreviewed`, `is:basket`, `is:bw` | Also `unedited`, `reviewed`, `pick`, `reject`, `colour` |
| `shows:"kids at the beach"` | What the picture shows, in words, best match first (smart search, below) |
| `-tag:family` | A minus turns any term around |
| `garden` | A plain word: filename, folder, camera, lens or tags — or, with **✦ Describe** on, what the pictures show |

A RAW+JPEG pair is one result, carrying the pair's rating, flag, label, tags and
edit whichever file they were set on. A query that cannot be read says which term
and why (`Unknown field 'lense:'`) and leaves the grid as it was.

The same search is `GET /api/catalog/search?q=…` and the `library.search` MCP tool.

**Smart search.** With **✦ Describe** on, plain words say what is in the photo —
`kids at the beach rating:>=3`, `a red car`, `snow date:2023` — and the other terms
still hold alongside. Each photograph is described once by CLIP (ViT-B/32, run in
OpenCV from its ONNX export) from its grid thumbnail, in the background, and a
search ranks the library by how much nearer each description is to the words
than to "a photo" at all (which keeps blank skies and blurs, near to every
description, from topping every search). The cut was set by looking down the
ranked results of a 29,000-photo library until they stopped being what was
asked for; when even the best match is weak the result says so ("no clear match
— these are the closest"). Setting it up is one click and a one-time
305 MB download into `models/` (checked against a pinned revision's hashes);
describing takes roughly 15 minutes per 40,000 photos on a CPU, and searches work
on what is done so far. A RAW+JPEG pair is described once, by its JPEG.
`GET /api/library/smart-search` says how far it is.

## Driving it from an agent (MCP)

The backend serves the library over [MCP](https://modelcontextprotocol.io) at
`http://127.0.0.1:8000/mcp` (streamable HTTP), in the same process as the UI. An
agent works on the same catalog and sidecars you do: a rating it sets is in the
grid at the next refresh, and an edit it makes opens in the develop view.

**Claude Code** picks it up from `.mcp.json` in the project folder while the backend is
running. Elsewhere:

```bash
claude mcp add --transport http photool http://127.0.0.1:8000/mcp
```

Clients that only speak stdio (some desktop apps) can reach it through a bridge
such as `npx mcp-remote http://127.0.0.1:8000/mcp`.

Every tool is one entry of the command registry in `backend/commands/`. The same
commands answer plain HTTP, which is handy from a script:

```bash
curl localhost:8000/api/commands                       # every command and its parameters
curl -X POST localhost:8000/api/commands/library.photos      -H 'Content-Type: application/json' -d '{"folder": "2025/2025-03-06", "min_rating": 3}'
```

| Tools | What they do |
|-------|--------------|
| `library.search`, `library.folders`, `library.photos`, `library.photo`, `library.stats`, `library.bursts`, `tags.list` | Find photos: a search across the library, folders, the grid's filters, one photo in full, burst groups |
| `photo.rate`, `photo.set_tags`, `tags.apply`, `bursts.reject_duplicates` | Cull: stars, pick/reject, colour labels, tags, keep-the-sharpest |
| `develop.controls`, `develop.get`, `develop.set`, `develop.reset`, `develop.auto`, `develop.paste`, `develop.presets` | Develop: every control with its range and meaning (tone curves and the colour mixer included), then read, change (merge or replace), Auto, paste onto many |
| `develop.look_health` | Whether a RAW's camera look draws like the camera, or is posterized somewhere in its tones |
| `render.preview`, `render.loupe` | Look: the developed photo, or a 1:1 patch, returned as an image. `render.preview` takes trial `settings` without saving them |
| `cull.measure`, `cull.signals`, `cull.best` | Know before looking: each frame's scene, burst and keeper, sharpness rank in the shoot, soft and blown warnings, faces, people and closed eyes; the library's own best picks |
| `render.thumbnail`, `render.contact_sheet` | Look cheaply: the camera's JPEG from the thumbnail cache, or up to 36 frames numbered on one sheet to compare a burst or a scene |
| `cull.propose`, `cull.proposals`, `cull.accept`, `cull.dismiss`, `cull.undo`, `cull.evaluate` | Suggest marks with a reason, for you to accept in the grid; measure an agent against a folder you have culled |
| `basket.list`, `basket.add`, `basket.remove`, `basket.clear`, `export.run`, `export.progress` | Export: fill the basket and write it out, with the export panel's options |

Writing tools are marked as such, and are refused on a read-only catalog rather
than reporting a save that did not happen. The endpoint answers only to
`localhost` / `127.0.0.1` host names, so a web page cannot reach it through DNS
rebinding. Another device on the network is refused like any API call unless it
is paired (see [Other computers](#other-computers)), and with `lan_access` off it is
not answered at all.

### Culling with a vision model

An agent culls by **proposing**, not by rating. `cull.propose` keeps its stars,
picks and rejects, each with a short reason, apart from your own marks: the grid
shows them as violet ✦ chips with a bar to *Accept all*, *Dismiss all* or *Select
them*, and the photo view shows the reason with **A** to accept and **N** to
dismiss. Accepting all never replaces a mark you have set; accepting one photo by
hand does. *Undo accept* puts back what was there, except where you have changed
it since. Nothing reaches the sidecar until it is accepted.

The flow the server describes to the agent: `cull.measure` once, then
`cull.signals` (compact, ~50 tokens a frame) to drop technical failures without
looking, a `render.contact_sheet` per scene or burst to choose among frames that
look alike, `render.loupe` on a finalist's face when focus is in doubt, and
`cull.propose`.

Before trusting one on new folders, try it on a folder you have finished:
`cull.signals` with `blind: true` and contact sheets with `marks: false` hide
your marks from it, then `cull.evaluate` reports keep/drop agreement, precision
(of what it kept, how much you kept), recall (of what you kept, how much it
found), and each disagreement with its reason. Proposals left pending or
dismissed are what is compared; `cull.dismiss` the run afterwards to clear the
grid. [calibration/culling.example.md](../calibration/culling.example.md) has the
prompt to paste in and how to read the results. Your own copy, with the folders
you calibrate against, what earlier runs found and a log of results, lives at
`<library>/.photomanager/culling.md`.

Adding a tool is adding a command: a function with `@command("area.verb",
params=SomeModel)` in `backend/commands/`. Its docstring is what the agent reads.

## Other computers

The album can be opened from a laptop, a tablet or a phone in the house — but
only from devices you have **paired**, never from "anything on the Wi-Fi".

1. On the computer that holds the library, put `lan_access: true` in
   `backend/config.yaml` and double-click **`serve.bat`** (on Windows). It
   installs any missing packages, builds the page the first time, and prints
   the address to open on the other device, e.g. `http://192.168.1.12:8000`.
   Leave its window open; closing it stops the server. After updating the code
   the page is rebuilt by itself when the server starts (`serve.bat rebuild`
   forces it). (Elsewhere: `cd frontend && npm run build`, then
   `cd backend && python serve.py`.) The first time, Windows Firewall asks
   about Python: allow it on **Private** networks only.

   **Without a window, from Windows start-up:** `serve.bat background` starts it
   with no window (`pythonw serve.py --background`); put a shortcut to
   `…\photool-env\venv\Scripts\pythonw.exe` with arguments
   `serve.py --background`, starting in `backend`, in the Startup folder
   (`Win+R`, `shell:startup`). Its output goes to `serve.log` beside the catalog,
   a second start while it runs does nothing, and **`stop.bat`** stops it.

   Running this way, it shows a **Photool icon in the system tray**. The menu has:

   - **Open Photool** (also on a double-click)
   - the addresses other devices use
   - **Open log**
   - **Restart**, which stops it, starts it again on the code now in the folder,
     and rebuilds the page if that changed
   - **Quit**

   Windows may first put the icon under the tray's ^ arrow; drag it out to keep it
   in sight. Without `pystray` (in `requirements.txt`) it runs as before, with no
   icon, and says so in the log.
2. On the other device, open that address, name the device and press **Ask to
   join**. It shows six digits.
3. On the computer, a bar appears: *"Living-room iPad (192.168.1.31) wants to see
   the album. Its screen should show 251 132"*. Approve only if the digits
   match. The device is in, for a year, until you revoke it under **Devices**.

Until it is paired a device gets the pairing screen and a 401 for every API call
— no thumbnail, no file name, no face. A paired device can browse, cull and edit,
but only the computer itself can approve or revoke devices. With `lan_access`
off (the default) anything arriving from the network is refused outright.

**Why pairing and not the alternatives.** A MAC address is not an identity: any
device can claim any MAC, phones randomise theirs on purpose, and it never
crosses a router. A code sent by e-mail or text needs an internet mail or SMS
service to send it and proves only that someone can read an inbox. Approving on
the computer that holds the photos proves that someone in the house said yes.
The device keeps a 256-bit random key in an HttpOnly, SameSite=Strict cookie;
only its SHA-256 is stored (`devices.db` beside the catalog).

**Encryption, and outside the house: Tailscale.** On plain HTTP the traffic is
readable by anyone who can see your Wi-Fi traffic, which on a shared-password
network is everyone who has the password. Set `tls_cert`/`tls_key` (from
`mkcert`, say) for HTTPS, or better, use [Tailscale](https://tailscale.com):
leave `lan_access` off, run the app as usual, and run `tailscale serve --bg 8000`.
Every device then reaches it over an encrypted WireGuard link, from home or
anywhere, with no port opened on your router — the zero-trust part. Requests
through Tailscale are never mistaken for the computer itself: they pair like
any device, or, for logins you list in `lan_trusted_logins` (e.g. your own Google
account), are let in on Tailscale's word. Run the backend with
`--no-proxy-headers` (`dev.sh`, `dev.bat` and `serve.py` do) so the app can tell
the two apart.

## Keyboard Shortcuts (Cull View)

| Key | Action |
|-----|--------|
| `→` / `←` or `j` / `k` | Next / previous image |
| `1-5` | Set star rating |
| `0` | Clear star rating |
| `P` | Toggle pick flag |
| `X` | Toggle reject flag |
| `U` | Clear flag |
| `]` | Pick + advance to next |
| `[` | Reject + advance to next |
| `6-9` | Color label (Red / Yellow / Green / Blue) |
| `T` | Jump to the event-tag box (Enter adds, `✕` removes, `Esc` leaves it) |
| `B` | Add / remove the current photo from the export basket |
| `Shift+R` | Reject the other frames of this burst. In the single view this keeps the sharpest frame; in compare it keeps the frame you are pointing at (only shown when the current photo is part of a burst) |
| `C` | Compare this burst side by side, or go back to one frame |
| `D` | Develop this photo — opens the full-screen develop view (press `D` again to come back; the cull view moves to the frame you ended on) |
| `Z` | In compare: switch between a 1:1 centre crop and fitting the frame |
| `I` | Toggle EXIF info overlay |
| `Esc` | Leave compare, or back to the grid |

### In the Develop View

| Key | Action |
|-----|--------|
| `→` / `←` | Next / previous photo, without leaving the view |
| `\` | Hold to see the original |
| `0` | Reset the edit |
| `A` | Auto — set exposure, highlights, shadows and squeeze from this frame's own histogram |
| `J` | Show / hide the clipping overlay (red = pure white, blue = pure black) |
| `B` | Put this edit on the rest of the burst |
| `Ctrl+C` / `Ctrl+V` | Copy this edit / paste it onto the frame on screen |
| `D` / `Esc` | Close and go back |
