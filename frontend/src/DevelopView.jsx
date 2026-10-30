import { useState, useEffect, useCallback, useRef, useMemo, useId, createContext, useContext } from "react";
import { copiedEdit, copyEdit, useCopiedEdit, copyChoice, keepCopyChoice, PASTE_AS, pasteAs, keepPasteAs,
  usePasteAs, pasteSummary } from "./editClipboard.js";

// Relative, so the page works from another computer on the network as well as
// through the dev server's proxy (vite.config.js) on this one.
const API = "/api";

// The develop preview is rendered for the screen showing it: 1600px leaves a
// 4K or ultrawide monitor upscaling the picture it is supposed to judge. Picked
// once, from a few fixed sizes, so the decode held for the preview is reused by
// Auto rather than read again at a size one pixel different.
const PREVIEW_SIZE = (() => {
  try {
    const dpr = window.devicePixelRatio || 1;
    const w = (window.innerWidth - 320) * dpr;     // less the slider panel
    const h = (window.innerHeight - 144) * dpr;    // less the top bar
    const shown = Math.min(w, h * 1.5);            // a 3:2 frame fitted to that
    return shown > 2300 ? 2560 : shown > 1750 ? 2048 : 1600;
  } catch {
    return 1600;
  }
})();

// Where the loupe was looking. Same reason: moving to the next frame of a burst
// to compare a face should land the loupe on the same face.
let lastLoupe = null;
// How far the loupe magnifies: 1 shows the photograph's pixels one for one, 2
// each as four screen pixels. Kept from frame to frame, as the point is.
let lastLoupeZoom = 1;
const LOUPE_SIDE = 480;
// The picture whose look "Match a look" gives other frames: a library photo
// ({ filepath, name }) or a file ({ image, name }). Kept from frame to frame.
let lookReference = null;
// The brush as it was last set (size a share of the frame's longer side): it
// stays from one frame to the next, as a brush in the hand does.
let lastBrush = { tool: null, size: 0.03, feather: 50, flow: 100 };

const signed = (v) => `${v > 0 ? "+" : ""}${v}`;

// The controls, in the groups the panel shows them (in the order a photo is
// worked through: see the panel's layout). The ranges and the meaning
// are the backend's (adjust.develop); these are the same numbers it clamps to,
// so a slider cannot ask for something the export will not do.
const GROUPS = [
  {
    id: "profile",
    title: "Profile",
    hint: "What the picture starts from: colour or black and white, the camera's own rendering, or a style learned from other frames.",
    sliders: [
      // Stored as how far to leave the camera's look behind, shown as how much of
      // it there is, so that an untouched frame reads 100: the camera's own
      // rendering, learned from the JPEG embedded in the RAW file.
      { key: "flat", label: "Camera look", min: 0, max: 100, step: 1, invert: true,
        needsLook: true, format: (v) => `${v}`,
        title: "How much of the camera's own rendering to keep — its tone curve and colour, learned from the JPEG embedded in this file. 100 starts from what the camera made; 0 is the plain, flat RAW decode. Exposure, highlights and shadows still work on the RAW data underneath." },
      // Only shown once a style is chosen (see StylePicker): 0 is the whole style,
      // 100 is back to the camera's own look.
      { key: "style_fade", label: "Style fade", min: 0, max: 100, step: 1, needsStyle: true,
        format: (v) => `${v}`,
        title: "Fade the style back towards this camera's own look. 0 is the whole style." },
    ],
  },
  {
    id: "wb",
    title: "White balance",
    hint: "A correction, so it comes first: a colour cast changes how bright the picture looks.",
    sliders: [
      { key: "temperature", label: "Warmth", min: -100, max: 100, step: 1, format: signed,
        title: "Warmer or cooler. A relative channel shift, not a colour temperature." },
      { key: "tint", label: "Tint", min: -100, max: 100, step: 1, format: signed,
        title: "Green or magenta, for a cast the white balance cannot reach." },
    ],
  },
  {
    id: "light",
    title: "Light",
    hint: "Exposure is in stops of light. Highlights, shadows, whites and blacks reshape one part of the range each, and Squeeze compresses the whole scene.",
    sliders: [
      { key: "exposure", label: "Exposure", min: -5, max: 5, step: 0.05,
        format: (v) => `${v > 0 ? "+" : ""}${Number(v).toFixed(2)}`,
        title: "Brighten or darken, in stops of light. Decoded from the RAW file, so a frame shot under can be brought up." },
      { key: "highlights", label: "Highlights", min: -100, max: 100, step: 1, format: signed,
        title: "Pull the bright end back (left) to recover sky and windows, or push it further." },
      { key: "shadows", label: "Shadows", min: -100, max: 100, step: 1, format: signed,
        title: "Open up the dark end (right) like a fill light, or deepen it." },
      { key: "whites", label: "Whites", min: -100, max: 100, step: 1, format: signed,
        title: "Move the white point — the very top of the range only." },
      { key: "blacks", label: "Blacks", min: -100, max: 100, step: 1, format: signed,
        title: "Move the black point — the very bottom of the range only." },
      { key: "compress", label: "Squeeze", min: 0, max: 100, step: 1, format: (v) => `${v}`,
        title: "Squeeze a scene that is wider than a picture can show — a bright window and a dark room — into a normal 8-bit JPEG, keeping the local detail." },
    ],
  },
  {
    id: "noise",
    title: "Noise reduction",
    hint: "A learned denoiser on the RAW's own data, run on the graphics card. The first use on a photo takes about ten seconds; after that the sliders are instant. RAW files only.",
    sliders: [
      { key: "nr_luma", label: "Luminance", min: 0, max: 100, step: 1, format: (v) => `${v}`,
        title: "How much of the grain is taken out. 100 is all of it; 50–80 leaves a fine, natural grain. Check it at 1:1 with the loupe (L)." },
      { key: "nr_color", label: "Colour", min: 0, max: 100, step: 1, format: (v) => `${v}`,
        title: "How much of the coloured speckle and blotching is taken out. Usually 100 on a high-ISO photo." },
    ],
  },
  {
    // Lightroom's Detail panel. Stored so 0 is its default for each control
    // (backend adjust/sharpen.py): the radius as tenths of a pixel added to 1.0,
    // the detail as Lightroom's 0..100 less 25.
    id: "sharpen",
    title: "Sharpening",
    hint: "After noise reduction, on brightness only. A downsized preview hides most of it: judge it at 1:1 with the loupe (L).",
    sliders: [
      { key: "sharpen", label: "Amount", min: 0, max: 150, step: 1, format: (v) => `${v}`,
        title: "How much. 0 is off; 40 is Lightroom's default for a RAW, 60–90 for a crisp web picture." },
      { key: "sharpen_radius", label: "Radius", min: -5, max: 20, step: 1, needsSharpen: true,
        format: (v) => `${(1 + v / 10).toFixed(1)} px`,
        title: "The size of the detail sharpened. 0.5–1.0 px for fine detail like hair and leaves; 1.5–3 px for a soft subject." },
      { key: "sharpen_detail", label: "Detail", min: -25, max: 75, step: 1, needsSharpen: true,
        format: (v) => `${25 + v}`,
        title: "Low holds strong edges back so they do not get halos; high sharpens every edge and texture fully. 25 is Lightroom's default." },
      { key: "sharpen_masking", label: "Masking", min: 0, max: 100, step: 1, needsSharpen: true,
        format: (v) => `${v}`,
        title: "Keep the sharpening to edges. 0 sharpens everywhere; 30–60 leaves skin and sky smooth while eyes and hair stay crisp." },
    ],
  },
  {
    // Lens corrections (backend adjust/lens.py). `lens_profile` is the switch
    // drawn above the slider (0 off, 1 the camera's own correction).
    id: "lens",
    title: "Lens corrections",
    switches: ["lens_profile"],
    hint: "The camera corrects its JPEG for the lens's distortion and the RAW is not corrected. The camera's correction is learned from this photo's own JPEG, so it is right for any lens and focal length. Vignetting is already corrected as the camera does, and colour fringes are removed in the decode.",
    sliders: [
      { key: "distortion", label: "Distortion", min: -100, max: 100, step: 1, format: signed,
        title: "By hand, on top of the camera's correction: right straightens lines that bow outwards (barrel), left lines that bow in. The picture is enlarged just enough to keep its corners." },
    ],
  },
  {
    // Transform (backend adjust/lens.py): the picture plane tilted, as Lightroom's
    // Transform panel. A frame's own, like its level: not carried by presets.
    id: "transform",
    title: "Transform",
    hint: "Straighten converging verticals (a building or a stage shot from below) and walls shot from one side. Auto finds it from the photo's own lines. The picture is enlarged just enough to keep its corners.",
    sliders: [
      { key: "tf_vertical", label: "Vertical", min: -100, max: 100, step: 1, format: signed,
        title: "Left straightens verticals that lean in towards the top (shot from below); right the other way." },
      { key: "tf_horizontal", label: "Horizontal", min: -100, max: 100, step: 1, format: signed,
        title: "For a wall or a stage shot from one side." },
      { key: "tf_aspect", label: "Aspect", min: -100, max: 100, step: 1, format: signed,
        title: "Taller (right) or wider, for the squat look a strong vertical correction leaves." },
    ],
  },
  {
    id: "presence",
    title: "Presence",
    sliders: [
      { key: "contrast", label: "Contrast", min: -100, max: 100, step: 1, format: signed,
        title: "More or less separation about the mid-tones." },
      { key: "clarity", label: "Clarity", min: -100, max: 100, step: 1, format: signed,
        title: "Mid-tone local contrast: punch and texture without darkening the shadows or crunching the sky." },
      { key: "vibrance", label: "Vibrance", min: -100, max: 100, step: 1, format: signed,
        title: "Saturation that favours flat colours and spares skin tones." },
      { key: "saturation", label: "Saturation", min: -100, max: 100, step: 1, format: signed,
        title: "All colours alike. Fully left is black and white." },
    ],
  },
  {
    id: "toning",
    title: "Fade & toning",
    hint: "The matte lift and the split tone: a colour in the shadows and another in the highlights.",
    sliders: [
      { key: "shadow_hue", label: "Shadow hue", min: 0, max: 360, step: 1, format: (v) => `${v}°`,
        hue: true, title: "The colour tint in the dark end." },
      { key: "shadow_sat", label: "Shadow tint", min: 0, max: 100, step: 1, format: (v) => `${v}`,
        title: "How strongly the shadows take that colour." },
      { key: "highlight_hue", label: "Light hue", min: 0, max: 360, step: 1, format: (v) => `${v}°`,
        hue: true, title: "The colour tint in the bright end." },
      { key: "highlight_sat", label: "Light tint", min: 0, max: 100, step: 1, format: (v) => `${v}`,
        title: "How strongly the highlights take that colour." },
      { key: "fade", label: "Fade", min: 0, max: 100, step: 1, format: (v) => `${v}`,
        title: "Lift the blacks and ease the whites: the matte, filmic look." },
    ],
  },
  {
    id: "effects",
    title: "Effects",
    hint: "On the picture as cropped. Grain is the same grains every time, sized to the picture, so the preview, the loupe and the export agree.",
    sliders: [
      { key: "vignette", label: "Vignette", min: -100, max: 100, step: 1, format: signed,
        title: "Darken (left) or lighten (right) the corners of the picture as cropped." },
      { key: "grain", label: "Grain", min: 0, max: 100, step: 1, format: (v) => `${v}`,
        title: "Film grain, strongest in the mid-tones as on film. Check it at 1:1 with the loupe (L)." },
      { key: "grain_size", label: "Grain size", min: -100, max: 100, step: 1, format: signed,
        title: "0 is about a 35 mm film's grain; left finer, right coarser. Needs some grain." },
    ],
  },
];

// What the panel says under Noise reduction, by the state the server reports.
const NOISE_NOTES = {
  working: "Reducing noise… the photo is shown without it until it is ready.",
  failed: "Noise reduction could not be made for this photo.",
  "raw-only": "Noise reduction works on RAW files only.",
  unavailable: "Noise reduction is not set up on the server (onnxruntime is missing).",
};

// Black and white. `mono` is the treatment (0 colour, 1 black and white); the B&W
// mix is how light each band of colour turns in grey, the colour filters of film.
const BW_MIX_KEYS = ["red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta"].map((b) => `bw_${b}`);
const BW_KEYS = ["mono", ...BW_MIX_KEYS];
// Controls with nothing to work on in black and white, hidden while it is on.
const COLOUR_ONLY_KEYS = ["vibrance", "saturation"];

// A film photographer's colour filters, as B&W mixes. Each one lightens its own
// colour and darkens the opposite one: a red filter lightens skin and brick and
// turns a blue sky dark.
const BW_FILTERS = [
  { name: "Neutral", note: "Each colour as bright as it is", mix: {} },
  { name: "Yellow", note: "A touch more sky, the classic landscape filter",
    mix: { bw_orange: 15, bw_yellow: 30, bw_green: 5, bw_aqua: -15, bw_blue: -30, bw_purple: -10 } },
  { name: "Orange", note: "Darker sky, smoother skin",
    mix: { bw_red: 25, bw_orange: 35, bw_yellow: 20, bw_green: -10, bw_aqua: -30, bw_blue: -45, bw_purple: -20 } },
  { name: "Red", note: "Dramatic, near-black sky; bright brick and skin",
    mix: { bw_red: 45, bw_orange: 35, bw_yellow: 10, bw_green: -25, bw_aqua: -45, bw_blue: -70, bw_purple: -35, bw_magenta: 15 } },
  { name: "Green", note: "Bright foliage, deeper skin tones",
    mix: { bw_red: -30, bw_orange: -15, bw_yellow: 15, bw_green: 45, bw_aqua: 20, bw_blue: -10 } },
  { name: "Blue", note: "Pale sky and haze, dark reds",
    mix: { bw_red: -40, bw_orange: -30, bw_yellow: -20, bw_green: -10, bw_aqua: 20, bw_blue: 45, bw_purple: 25 } },
  { name: "Infrared", note: "Glowing white foliage under a black sky",
    mix: { bw_red: 20, bw_yellow: 60, bw_green: 90, bw_aqua: 10, bw_blue: -80, bw_purple: -40 } },
];

// Toning on the grey, made with split toning: a hue and a strength at each end.
const SPLIT_KEYS = ["shadow_hue", "shadow_sat", "highlight_hue", "highlight_sat"];
const BW_TONES = [
  { name: "None", note: "Neutral grey", tone: {} },
  { name: "Sepia", note: "Warm brown throughout",
    tone: { shadow_hue: 30, shadow_sat: 30, highlight_hue: 40, highlight_sat: 25 } },
  { name: "Selenium", note: "Cool purple-brown shadows, neutral lights",
    tone: { shadow_hue: 300, shadow_sat: 18 } },
  { name: "Platinum", note: "Soft warm-neutral, like a platinum print",
    tone: { shadow_hue: 35, shadow_sat: 12, highlight_hue: 45, highlight_sat: 10 } },
  { name: "Cyanotype", note: "Prussian blue",
    tone: { shadow_hue: 215, shadow_sat: 45, highlight_hue: 200, highlight_sat: 30 } },
  { name: "Split", note: "Cool shadows, warm lights",
    tone: { shadow_hue: 220, shadow_sat: 25, highlight_hue: 42, highlight_sat: 25 } },
];

// What to say about where the camera look came from. Plain and low: it is
// information about how much to trust the starting point, not a warning.
const LOOK_NOTES = {
  ready: (n) => ({
    text: `from ${n} frames of this camera`,
    title: "The camera's look was learned from many frames of this camera and style, so it should sit close to the camera's own JPEG.",
  }),
  learning: (n) => ({
    text: n > 1 ? `still learning (${n} frames)` : "still learning (1 frame)",
    title: "Learned from only the first frames of this camera, lens or style. It is a fair start and gets closer to the camera's JPEG as more of its frames are opened.",
  }),
  own: () => ({
    text: "from this frame alone",
    title: "This frame's colour differs from the camera's other frames (another style, perhaps), so it uses a look learned from its own JPEG only.",
  }),
  alone: () => ({
    text: "from this frame alone",
    title: "The app could not tell this camera from others (no readable model or colour data), so its frames do not share what each learns.",
  }),
  none: () => ({
    text: "none for this file",
    title: "There is no JPEG in this file that matches the picture, so there is nothing to learn the camera's look from. The plain decode is shown.",
  }),
};

// When a look may be off, and more frames would help. A suggestion, in the same low
// voice as the note: nothing is blocked and nothing is done until it is clicked.
const LOOK_HINTS = {
  rough: "This frame may sit away from the camera's own JPEG.",
  scene: "No other frame here looks like this scene yet, so exposure edits are less certain.",
};

// The colour mixer: hue, saturation and luminance for eight bands of hue, the
// bands and centres Lightroom uses (adjust.develop.HSL_BANDS). Shown a kind at a
// time, as Lightroom's HSL panel does, so eight sliders rather than twenty-four.
const HSL_BANDS = [
  { id: "red", label: "Red", hue: 0 },
  { id: "orange", label: "Orange", hue: 30 },
  { id: "yellow", label: "Yellow", hue: 60 },
  { id: "green", label: "Green", hue: 120 },
  { id: "aqua", label: "Aqua", hue: 180 },
  { id: "blue", label: "Blue", hue: 240 },
  { id: "purple", label: "Purple", hue: 270 },
  { id: "magenta", label: "Magenta", hue: 300 },
];
const HSL_KINDS = [
  { id: "hue", label: "Hue", title: "Shift each colour toward its neighbour: red toward orange, blue toward aqua or purple." },
  { id: "sat", label: "Saturation", title: "More or less of each colour. Fully left turns that colour grey and leaves the rest." },
  { id: "lum", label: "Luminance", title: "Lighter or darker, colour by colour: deepen a blue sky, brighten skin, darken foliage." },
];
const HSL_KEYS = HSL_KINDS.flatMap((k) => HSL_BANDS.map((b) => `${k.id}_${b.id}`));

// Tone curves: [[x, y], …] on 0..255, for all channels and for each. A list, not
// a number, so they sit beside the sliders rather than among them; null is none.
const CURVE_CHANNELS = [
  { key: "curve", label: "RGB", name: "all three channels together", colour: "#d4d4d4" },
  { key: "curve_r", label: "R", name: "the red channel alone", colour: "#f87171" },
  { key: "curve_g", label: "G", name: "the green channel alone", colour: "#4ade80" },
  { key: "curve_b", label: "B", name: "the blue channel alone", colour: "#60a5fa" },
];
const CURVE_KEYS = CURVE_CHANNELS.map((c) => c.key);
const MAX_CURVE_POINTS = 16;
const curveEdited = (points) =>
  Array.isArray(points) && points.length >= 2 && points.some(([x, y]) => x !== y);

const ALL_SLIDERS = GROUPS.flatMap((g) => g.sliders);
// Every control that is a number: the panel's sliders and the mixer's.
const KEYS = [...ALL_SLIDERS.map((s) => s.key), ...HSL_KEYS, ...BW_KEYS, "lens_profile"];

// Framing is part of the edit but not of its tone: presets carry the tone and
// leave each frame's own crop and level alone, because a crop is a composition
// and fits only the frame it was drawn on. Copy carries them only when ticked
// (COPY_SECTIONS), for frames made from a camera that did not move.
const GEOM_KEYS = ["angle", "crop_x", "crop_y", "crop_w", "crop_h"];
// What belongs to one frame and is not carried to another by presets, nor
// undone by Reset: its framing and the spots removed from it.
const TRANSFORM_KEYS = ["tf_vertical", "tf_horizontal", "tf_aspect"];
const OWN_KEYS = [...GEOM_KEYS, "spots", ...TRANSFORM_KEYS];
const ZERO = {
  ...Object.fromEntries(KEYS.map((k) => [k, 0])),
  ...Object.fromEntries(CURVE_KEYS.map((k) => [k, null])),
  // A learned style, by id ("" = the camera's own look). The one control that is
  // a name rather than a number.
  style: "",
  // Gradients (local adjustments, backend adjust/masks.py): a list, each with
  // its own position and controls.
  masks: [],
  // Spots removed (backend adjust/spots.py): a frame's own, like its crop.
  spots: [],
  angle: 0, crop_x: null, crop_y: null, crop_w: null, crop_h: null,
};

// Ratios are width:height of the finished picture. Both orientations are listed
// because the choice between them is the composition.
const ASPECTS = [
  { id: "free", label: "Free", ratio: null, note: "Any shape" },
  { id: "orig", label: "Original", ratio: "orig", note: "Keep the camera's shape" },
  { id: "1:1", label: "1:1", ratio: 1, note: "Square: Instagram grid, profile pictures" },
  { id: "4:5", label: "4:5", ratio: 4 / 5, note: "Instagram, Facebook and Threads portrait" },
  { id: "5:4", label: "5:4", ratio: 5 / 4, note: "8×10 print" },
  { id: "3:2", label: "3:2", ratio: 3 / 2, note: "35 mm frame, 4×6 print" },
  { id: "2:3", label: "2:3", ratio: 2 / 3, note: "35 mm frame upright, Pinterest" },
  { id: "4:3", label: "4:3", ratio: 4 / 3, note: "Phone and compact cameras" },
  { id: "3:4", label: "3:4", ratio: 3 / 4, note: "Phone portrait" },
  { id: "16:9", label: "16:9", ratio: 16 / 9, note: "Widescreen, YouTube, slides" },
  { id: "9:16", label: "9:16", ratio: 9 / 16, note: "Stories, Reels, TikTok, Shorts" },
];

const MIN_BOX = 0.03;

// Mirrors adjust.geometry: a turned picture has empty corners, and a crop may not
// reach into them.
function boxFits(box, angle, W, H) {
  if (!angle) return true;
  const cos = Math.cos((angle * Math.PI) / 180);
  const sin = Math.sin((angle * Math.PI) / 180);
  const { x, y, w, h } = box;
  for (const [px, py] of [[x, y], [x + w, y], [x, y + h], [x + w, y + h]]) {
    const dx = px * W - W / 2;
    const dy = py * H - H / 2;
    const sx = cos * dx + sin * dy;
    const sy = -sin * dx + cos * dy;
    if (Math.abs(sx) > W / 2 + 0.5 || Math.abs(sy) > H / 2 + 0.5) return false;
  }
  return true;
}

function fitBox(box, angle, W, H) {
  if (boxFits(box, angle, W, H)) return box;
  const at = (cx, cy, k) => ({ x: cx - (box.w * k) / 2, y: cy - (box.h * k) / 2, w: box.w * k, h: box.h * k });
  for (const [cx, cy] of [[box.x + box.w / 2, box.y + box.h / 2], [0.5, 0.5]]) {
    if (!boxFits(at(cx, cy, 0.01), angle, W, H)) continue;
    let lo = 0.01;
    let hi = 1;
    for (let i = 0; i < 24; i += 1) {
      const mid = (lo + hi) / 2;
      if (boxFits(at(cx, cy, mid), angle, W, H)) lo = mid; else hi = mid;
    }
    return at(cx, cy, lo);
  }
  return box;
}

const FULL_BOX = { x: 0, y: 0, w: 1, h: 1 };

function boxOf(settings) {
  const { crop_x: x, crop_y: y, crop_w: w, crop_h: h } = settings;
  return [x, y, w, h].every((v) => typeof v === "number") ? { x, y, w, h } : null;
}

// The largest box of a pixel ratio that fits inside `within` (fractions of the
// frame of W x H pixels), centred on it.
function boxOfRatio(ratio, within, W, H) {
  const target = ratio === "orig" ? W / H : ratio;
  const fracRatio = (target * H) / W; // width over height, in fractions
  let w = within.w;
  let h = w / fracRatio;
  if (h > within.h) { h = within.h; w = h * fracRatio; }
  return { x: within.x + (within.w - w) / 2, y: within.y + (within.h - h) / 2, w, h };
}

// What "Auto" and the rescue presets own: how much light, and where it sits.
const LIGHT_KEYS = ["exposure", "highlights", "shadows", "whites", "blacks", "compress"];
// What the looks own: the styling on top of a finished exposure. White balance
// is left to the person, since it is a correction and not a taste.
// A look also chooses colour or black and white, its filter and its grain, so
// picking a colour look after a black-and-white one gives a colour picture.
const LOOK_KEYS = [
  "contrast", "clarity", "vibrance", "saturation", "fade",
  ...SPLIT_KEYS, ...BW_KEYS, "grain", "grain_size", "vignette",
];

// What Copy can carry, in the panel's own sections. The tone is ticked to begin
// with and the framing is not: a crop and a level fit the frame they were made
// on, unless the camera did not move (a tripod), which is what ticking Level or
// Crop is for. Spots likewise: sensor dust is in the same place frame after
// frame. The ticks are kept on this device (editClipboard.js).
const COPY_SECTIONS = [
  { id: "profile", label: "Profile", keys: ["mono", "flat", "style", "style_fade"] },
  { id: "wb", label: "White balance", keys: ["temperature", "tint"] },
  { id: "light", label: "Light", keys: LIGHT_KEYS },
  { id: "noise", label: "Noise reduction", keys: ["nr_luma", "nr_color"] },
  { id: "sharpen", label: "Sharpening", keys: ["sharpen", "sharpen_radius", "sharpen_detail", "sharpen_masking"] },
  { id: "lens", label: "Lens corrections", keys: ["lens_profile", "distortion"] },
  { id: "presence", label: "Presence", keys: ["contrast", "clarity", "vibrance", "saturation"] },
  { id: "curve", label: "Tone curve", keys: CURVE_KEYS },
  { id: "mixer", label: "Colour mixer / B&W mix", keys: [...HSL_KEYS, ...BW_MIX_KEYS] },
  { id: "toning", label: "Fade & toning", keys: ["fade", ...SPLIT_KEYS] },
  { id: "effects", label: "Effects", keys: ["vignette", "grain", "grain_size"] },
  { id: "masks", label: "Gradients & brushes", keys: ["masks"] },
  { id: "transform", label: "Transform", keys: TRANSFORM_KEYS, framing: true },
  { id: "level", label: "Level", keys: ["angle"], framing: true },
  { id: "crop", label: "Crop", keys: ["crop_x", "crop_y", "crop_w", "crop_h"], framing: true },
  { id: "spots", label: "Spots", keys: ["spots"], framing: true },
];
const TONE_SECTIONS = COPY_SECTIONS.filter((s) => !s.framing).map((s) => s.id);
// Tied to where things are in the frame: not put on a frame turned the other way.
const PLACED_KEYS = ["crop_x", "crop_y", "crop_w", "crop_h", "spots"];
const chosenSections = () => {
  const ids = copyChoice(TONE_SECTIONS);
  return COPY_SECTIONS.filter((s) => ids.includes(s.id));
};
// What the sections come to, said briefly: "Tone + Level" rather than eleven names.
function sectionNames(sections) {
  const tone = sections.filter((s) => !s.framing);
  const names = tone.length === TONE_SECTIONS.length ? ["Tone"] : tone.map((s) => s.label);
  return [...names, ...sections.filter((s) => s.framing).map((s) => s.label)];
}

// One-click starting points. Rescue sets the light and leaves the styling;
// looks set the styling and leave the light, so the two can be combined — pull a
// window back, then choose how the picture should feel.
//
// Tuned on RAW frames seen through the camera's look, which is where a RAW
// starts. The look's own curve is a steep S, so a lift that was right for the
// plain decode lands in its steepest part: the first set (shadows +75, squeeze
// 70) took a backlit frame's contrast to a third of the camera's and turned it
// milky blue, and pulling the whites as well as the highlights left a recovered
// sky with nothing in the picture brighter than grey. These keep the frame's
// depth (each holds its black point, and Recover sky gives the white point
// back) and were checked on blown skies, backlit stages, dim halls and night
// concerts from this library.
const RESCUES = [
  { name: "Recover sky", note: "Bring back blown skies and windows, keep the whites white",
    settings: { highlights: -70, whites: 20 } },
  { name: "Open shadows", note: "A fill-light for dark faces and foregrounds, without the haze",
    settings: { shadows: 40, blacks: -20 } },
  { name: "Backlit", note: "Bright background, dark subject",
    settings: { shadows: 45, blacks: -20, highlights: -55, whites: 15 } },
  { name: "Even out", note: "A scene wider than a picture, brought in without going flat",
    settings: { compress: 30, highlights: -45, shadows: 30, blacks: -20 } },
  { name: "Lift 1 stop", note: "A frame a little under",
    settings: { exposure: 1, highlights: -30, blacks: -10 } },
  { name: "Lift 2 stops", note: "A frame shot well under",
    settings: { exposure: 2, highlights: -45, whites: -10, blacks: -15 } },
  { name: "Very dark", note: "Three stops or more under: a dim hall, a night stage",
    settings: { exposure: 3, shadows: 20, highlights: -60, blacks: -20 } },
];

const LOOKS = [
  { name: "Subtle polish", note: "A light touch on a good exposure",
    settings: { contrast: 8, clarity: 10, vibrance: 12 } },
  { name: "Clean pop", note: "Crisp and colourful",
    settings: { contrast: 15, clarity: 20, vibrance: 28 } },
  { name: "Matte", note: "Lifted blacks, softened colour",
    settings: { fade: 35, contrast: -8, saturation: -10, vibrance: 15 } },
  { name: "Warm film", note: "Faded, golden highlights, cool shadows",
    settings: { fade: 20, contrast: 10, saturation: -5,
      shadow_hue: 215, shadow_sat: 20, highlight_hue: 40, highlight_sat: 30 } },
  { name: "Teal & orange", note: "Cinematic split",
    settings: { contrast: 20, vibrance: 20,
      shadow_hue: 195, shadow_sat: 35, highlight_hue: 35, highlight_sat: 30 } },
  { name: "B&W punchy", note: "Punchy monochrome",
    settings: { mono: 1, contrast: 25, clarity: 20 } },
  { name: "B&W film", note: "Yellow filter, fine grain, a soft edge",
    settings: { mono: 1, contrast: 15, ...BW_FILTERS[1].mix, grain: 30, vignette: -15 } },
  { name: "B&W dramatic", note: "Red filter: dark sky, hard light",
    settings: { mono: 1, contrast: 35, clarity: 25, ...BW_FILTERS[3].mix, vignette: -25 } },
  { name: "B&W soft", note: "Lifted blacks, gentle, grainy",
    settings: { mono: 1, fade: 25, contrast: -10, grain: 25, grain_size: 30 } },
  { name: "B&W sepia", note: "Warm-toned print",
    settings: { mono: 1, contrast: 10, ...BW_TONES[1].tone, vignette: -10 } },
];

function asSettings(stored) {
  const next = { ...ZERO };
  for (const key of KEYS) next[key] = Number(stored?.[key]) || 0;
  for (const key of CURVE_KEYS) {
    const points = stored?.[key];
    next[key] = Array.isArray(points) ? points.map(([x, y]) => [Number(x), Number(y)]) : null;
  }
  next.style = typeof stored?.style === "string" ? stored.style : "";
  next.masks = Array.isArray(stored?.masks) ? stored.masks.map((m) => ({ ...m })) : [];
  next.spots = Array.isArray(stored?.spots) ? stored.spots.map((sp) => ({ ...sp })) : [];
  next.angle = Number(stored?.angle) || 0;
  const box = boxOf(stored || {});
  if (box) Object.assign(next, { crop_x: box.x, crop_y: box.y, crop_w: box.w, crop_h: box.h });
  return next;
}

// An edit's tone with this frame's own framing put back: what presets and Reset
// do, so a crop is never carried to a frame it was not drawn for. (Paste carries
// what the Copy checklist says instead.)
function withFramingOf(tone, own) {
  const next = { ...tone };
  for (const key of OWN_KEYS) next[key] = own[key];
  return next;
}

function hasFraming(settings) {
  return Number(settings.angle) !== 0 || boxOf(settings) !== null;
}

function isEdited(settings) {
  return isToneEdited(settings) || hasFraming(settings) || !!settings.spots?.length;
}

function isToneEdited(settings) {
  return KEYS.some((k) => Number(settings[k]) !== 0) || CURVE_KEYS.some((k) => curveEdited(settings[k]))
    || !!settings.style || (settings.masks || []).some(maskIsActive);
}

// Only the controls that are doing something, so the server stores an edit and
// not a page of zeros.
function activeOnly(settings, withFraming = true) {
  const out = Object.fromEntries(KEYS.filter((k) => Number(settings[k]) !== 0).map((k) => [k, settings[k]]));
  for (const key of CURVE_KEYS) if (curveEdited(settings[key])) out[key] = settings[key];
  if (settings.style) out.style = settings.style;
  else delete out.style_fade;
  if (settings.masks?.length) out.masks = settings.masks;
  // Always, framed or not: a spot is part of the picture, not of its framing.
  if (settings.spots?.length) out.spots = settings.spots;
  if (withFraming) {
    if (Number(settings.angle) !== 0) out.angle = settings.angle;
    const box = boxOf(settings);
    if (box) Object.assign(out, { crop_x: box.x, crop_y: box.y, crop_w: box.w, crop_h: box.h });
  }
  return out;
}

// The framing alone, for the "original" that is shown beside a cropped edit.
const framingOnly = (settings) => Object.fromEntries(
  Object.entries(activeOnly(settings)).filter(([k]) => GEOM_KEYS.includes(k)),
);

// How dark or light the frame is, so the histogram is drawn from the picture
// rather than from the numbers. The original is kept alongside the edit so the
// two can be shown against each other: a tonal change is much easier to read as
// a moved curve than as two similar-looking photographs.
//
// Also counts how much of the frame is pinned at white or at black. A rescue is
// judged by those two numbers more than by the curve.
//
// Red, green and blue are counted each on their own as well as the brightness. A
// blue sky whose blue channel is pinned at 255 is only about 75% bright, so on a
// brightness curve alone it sat mid-high while the readout below said "white
// 5.6%": a channel's clipping is only seen on that channel's own curve.
function histogramOf(bitmap, bins = 128) {
  const width = 256;
  const height = Math.max(1, Math.round((bitmap.height / bitmap.width) * width));
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0, width, height);
  const { data } = ctx.getImageData(0, 0, width, height);

  const counts = new Uint32Array(bins);
  const channels = [new Uint32Array(bins), new Uint32Array(bins), new Uint32Array(bins)];
  // Each bin the same width of the 0..255 scale, the two ends included.
  const binOf = (v) => Math.min(bins - 1, Math.floor((v * bins) / 256));
  let hi = 0;
  let lo = 0;
  for (let i = 0; i < data.length; i += 4) {
    const value = 0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2];
    counts[binOf(value)] += 1;
    channels[0][binOf(data[i])] += 1;
    channels[1][binOf(data[i + 1])] += 1;
    channels[2][binOf(data[i + 2])] += 1;
    if (data[i] >= 253 || data[i + 1] >= 253 || data[i + 2] >= 253) hi += 1;
    if (data[i] <= 2 && data[i + 1] <= 2 && data[i + 2] <= 2) lo += 1;
  }
  // Normalised against the busiest bin, so a flat frame and a contrasty one are
  // both readable instead of one being a spike and the other a line.
  const peak = Math.max(1, ...counts);
  // The channels share one scale, so their heights compare; a pinned channel's
  // spike at either end is not allowed to flatten everything else, as Lightroom's
  // does not: it is drawn to the top and cut there.
  const inner = channels.flatMap((c) => Array.from(c).slice(1, bins - 1));
  const channelPeak = Math.max(1, ...inner);
  const total = Math.max(1, width * height);
  return {
    bins: Array.from(counts, (c) => c / peak),
    rgb: channels.map((c) => Array.from(c, (v) => Math.min(1, v / channelPeak))),
    clipHigh: hi / total,
    clipLow: lo / total,
  };
}

// A translucent map of where the picture is clipped: red where any channel is
// at white, blue where all of them are at black. Drawn at a modest size, since
// it is a warning laid over the picture and not part of it.
function clipOverlayOf(bitmap) {
  const scale = Math.min(1, 900 / Math.max(bitmap.width, bitmap.height));
  const w = Math.max(1, Math.round(bitmap.width * scale));
  const h = Math.max(1, Math.round(bitmap.height * scale));
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0, w, h);
  const image = ctx.getImageData(0, 0, w, h);
  const { data } = image;
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i], g = data[i + 1], b = data[i + 2];
    if (r >= 253 || g >= 253 || b >= 253) {
      data[i] = 255; data[i + 1] = 30; data[i + 2] = 30; data[i + 3] = 170;
    } else if (r <= 2 && g <= 2 && b <= 2) {
      data[i] = 30; data[i + 1] = 110; data[i + 2] = 255; data[i + 3] = 170;
    } else {
      data[i + 3] = 0;
    }
  }
  ctx.putImageData(image, 0, 0);
  return canvas.toDataURL("image/png");
}

// The value under the cursor, in the colour space chosen for the readout. ``rgb``
// is the edited preview's 8-bit sRGB.
const READOUTS = [
  { id: "rgb", label: "RGB", show: ([r, g, b]) => `R ${r}  G ${g}  B ${b}` },
  { id: "pct", label: "RGB %", show: (c) => {
    const [r, g, b] = c.map((v) => Math.round((v / 255) * 100));
    return `R ${r}%  G ${g}%  B ${b}%`;
  } },
  { id: "hsl", label: "HSL", show: ([r, g, b]) => {
    const [h, sat, l] = hslOf(r, g, b);
    return `H ${h}°  S ${sat}%  L ${l}%`;
  } },
  { id: "lab", label: "Lab", show: ([r, g, b]) => {
    const [L, a, bb] = labOf(r, g, b);
    return `L ${L}  a ${a}  b ${bb}`;
  } },
];

function hslOf(r, g, b) {
  const [R, G, B] = [r / 255, g / 255, b / 255];
  const max = Math.max(R, G, B), min = Math.min(R, G, B);
  const l = (max + min) / 2;
  const d = max - min;
  let h = 0;
  if (d > 1e-6) {
    h = max === R ? ((G - B) / d) % 6 : max === G ? (B - R) / d + 2 : (R - G) / d + 4;
    h = (h * 60 + 360) % 360;
  }
  const sat = d < 1e-6 ? 0 : d / (1 - Math.abs(2 * l - 1));
  return [Math.round(h), Math.round(sat * 100), Math.round(l * 100)];
}

// CIE L*a*b* (D65) of an sRGB colour.
function labOf(r, g, b) {
  const lin = (v) => { const c = v / 255; return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };
  const [R, G, B] = [lin(r), lin(g), lin(b)];
  const x = (0.4124 * R + 0.3576 * G + 0.1805 * B) / 0.95047;
  const y = 0.2126 * R + 0.7152 * G + 0.0722 * B;
  const z = (0.0193 * R + 0.1192 * G + 0.9505 * B) / 1.08883;
  const f = (t) => (t > 216 / 24389 ? Math.cbrt(t) : (24389 / 27 * t + 16) / 116);
  const [fx, fy, fz] = [f(x), f(y), f(z)];
  return [Math.round(116 * fy - 16), Math.round(500 * (fx - fy)), Math.round(200 * (fy - fz))];
}

let lastReadout = "rgb";

// Exported so it can be rendered on its own: the shape it takes is the whole
// contract with histogramOf, and a mismatch between the two is a crash inside
// render, which takes the page down with it rather than showing a wrong curve.
export function Histogram({ original, edited, channels }) {
  const width = 256;
  const height = 64;
  const COLOURS = ["#ef4444", "#22c55e", "#3b82f6"];
  const path = (counts) =>
    counts
      .map((v, i) => `${(i / Math.max(1, counts.length - 1)) * width},${height - v * height}`)
      .join(" ");

  return (
    // The drawing is 256 x 64 and is stretched to the box: without the viewBox it
    // was drawn at 256 px in a wider box, so 255 sat three quarters of the way
    // across and the brightest peak read as about 200.
    <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" data-testid="histogram"
      className="w-full h-16 bg-neutral-950 rounded border border-neutral-800">
      <defs>
        <linearGradient id="devHist" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="#3b82f6" />
          <stop offset="50%" stopColor="#a3a3a3" />
          <stop offset="100%" stopColor="#ef4444" />
        </linearGradient>
      </defs>
      {original && (
        <polygon
          points={`0,${height} ${path(original)} ${width},${height}`}
          fill="none"
          stroke="#525252"
          strokeWidth="1"
          vectorEffect="non-scaling-stroke"
        />
      )}
      {channels && channels.map((counts, k) => (
        <polygon key={k}
          points={`0,${height} ${path(counts)} ${width},${height}`}
          fill={COLOURS[k]} fillOpacity="0.28" stroke={COLOURS[k]} strokeOpacity="0.8" strokeWidth="1" vectorEffect="non-scaling-stroke"
          style={{ mixBlendMode: "screen" }} />
      ))}
      {edited && (
        <polyline
          points={path(edited)}
          fill="none"
          stroke="#e5e5e5"
          strokeOpacity="0.85"
          strokeWidth="1"
          vectorEffect="non-scaling-stroke"
        />
      )}
    </svg>
  );
}

const percent = (fraction) => {
  const p = fraction * 100;
  if (p === 0) return "0%";
  return p < 0.1 ? "<0.1%" : `${p.toFixed(p < 10 ? 1 : 0)}%`;
};

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

// Resize a crop box by dragging a handle. (hx, hy) is the handle: 0 or 1 for a
// corner or side, 0.5 for the middle of an edge. `ratio` is width over height in
// fractions of the frame, or null for a free shape.
function resizeBox(start, hx, hy, p, ratio) {
  if (hx === 0.5 || hy === 0.5) {
    if (hx === 0.5) {
      if (hy === 0) {
        const top = clamp(p.y, 0, start.y + start.h - MIN_BOX);
        return { ...start, y: top, h: start.y + start.h - top };
      }
      const bottom = clamp(p.y, start.y + MIN_BOX, 1);
      return { ...start, h: bottom - start.y };
    }
    if (hx === 0) {
      const left = clamp(p.x, 0, start.x + start.w - MIN_BOX);
      return { ...start, x: left, w: start.x + start.w - left };
    }
    const right = clamp(p.x, start.x + MIN_BOX, 1);
    return { ...start, w: right - start.x };
  }
  const ax = hx === 1 ? start.x : start.x + start.w;
  const ay = hy === 1 ? start.y : start.y + start.h;
  const dirX = hx === 1 ? 1 : -1;
  const dirY = hy === 1 ? 1 : -1;
  const maxW = dirX > 0 ? 1 - ax : ax;
  const maxH = dirY > 0 ? 1 - ay : ay;
  let w = clamp((p.x - ax) * dirX, MIN_BOX, maxW);
  let h = clamp((p.y - ay) * dirY, MIN_BOX, maxH);
  if (ratio) {
    if (w / h > ratio) w = h * ratio; else h = w / ratio;
    if (w > maxW) { w = maxW; h = w / ratio; }
    if (h > maxH) { h = maxH; w = h * ratio; }
  }
  return { x: dirX > 0 ? ax : ax - w, y: dirY > 0 ? ay : ay - h, w, h };
}

const HANDLES = [
  [0, 0, "nwse-resize"], [1, 0, "nesw-resize"], [0, 1, "nesw-resize"], [1, 1, "nwse-resize"],
  [0.5, 0, "ns-resize"], [0.5, 1, "ns-resize"], [0, 0.5, "ew-resize"], [1, 0.5, "ew-resize"],
];

// The crop box, drawn over the picture and dragged with the mouse. The box is in
// fractions of the whole frame; while it is dragged only `onDraft` is called, and
// the edit is saved once, on release.
function CropOverlay({ box, ratio, angle, dims, onDraft, onCommit }) {
  const ref = useRef(null);
  const boxRef = useRef(box);
  boxRef.current = box;

  const begin = (mode, hx, hy) => (e) => {
    e.preventDefault();
    e.stopPropagation();
    const rect = ref.current.getBoundingClientRect();
    const start = { ...boxRef.current };
    const origin = { x: e.clientX, y: e.clientY };
    let last = start;
    const move = (ev) => {
      let next;
      if (mode === "move") {
        const dx = (ev.clientX - origin.x) / rect.width;
        const dy = (ev.clientY - origin.y) / rect.height;
        next = { ...start, x: clamp(start.x + dx, 0, 1 - start.w), y: clamp(start.y + dy, 0, 1 - start.h) };
      } else {
        next = resizeBox(start, hx, hy, {
          x: (ev.clientX - rect.left) / rect.width,
          y: (ev.clientY - rect.top) / rect.height,
        }, ratio);
      }
      // The empty corners of a turned frame are not part of any crop.
      if (!boxFits(next, angle, dims.w, dims.h)) return;
      last = next;
      onDraft(next);
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      onCommit(last);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  };

  const pct = (v) => `${v * 100}%`;
  return (
    <div ref={ref} className="absolute inset-0" style={{ touchAction: "none" }}>
      <div
        onPointerDown={begin("move")}
        className="absolute border border-white/90 cursor-move"
        style={{
          left: pct(box.x), top: pct(box.y), width: pct(box.w), height: pct(box.h),
          boxShadow: "0 0 0 9999px rgba(0,0,0,0.62)",
        }}
      >
        {/* Thirds: the usual guide for placing a horizon or a face. */}
        {[1 / 3, 2 / 3].map((f) => (
          <span key={`v${f}`} className="absolute top-0 bottom-0 border-l border-white/30 pointer-events-none" style={{ left: pct(f) }} />
        ))}
        {[1 / 3, 2 / 3].map((f) => (
          <span key={`h${f}`} className="absolute left-0 right-0 border-t border-white/30 pointer-events-none" style={{ top: pct(f) }} />
        ))}
        {HANDLES.filter(([hx, hy]) => !ratio || (hx !== 0.5 && hy !== 0.5)).map(([hx, hy, cursor]) => (
          <span
            key={`${hx}-${hy}`}
            onPointerDown={begin("resize", hx, hy)}
            className="absolute w-3 h-3 bg-white border border-neutral-900 rounded-sm"
            style={{ left: `calc(${pct(hx)} - 6px)`, top: `calc(${pct(hy)} - 6px)`, cursor }}
          />
        ))}
      </div>
    </div>
  );
}

// ── Gradients and brushes ──────────────────────────────────────────
//
// A local adjustment (backend adjust/masks.py): a linear gradient, whole at one
// point and gone at another, a radial one, whole inside an ellipse and fading
// out across its feather, a brush, only where it is painted, or the subject or
// the sky, as a model finds them in the frame (backend adjust/segment.py). Any of them
// takes brush strokes, painting more in or erasing. Positions are fractions of
// the whole frame before any crop or turn, which is how the picture is shown
// while they are drawn. Any of them can be cropped by a section (a rectangle it
// reaches only inside), and a radial is turned by its rotate handle.

const MASK_SLIDERS = [
  { key: "exposure", label: "Exposure", min: -4, max: 4, step: 0.05,
    format: (v) => `${v > 0 ? "+" : ""}${Number(v).toFixed(2)}`,
    title: "Brighten or darken where the gradient reaches, in stops: the same stop as the Exposure slider, where it is whole." },
  { key: "contrast", label: "Contrast", min: -100, max: 100, step: 1, format: signed,
    title: "More or less contrast where the gradient reaches." },
  { key: "highlights", label: "Highlights", min: -100, max: 100, step: 1, format: signed,
    title: "Pull back (left) or brighten the bright end where the gradient reaches: a sky, a window." },
  { key: "shadows", label: "Shadows", min: -100, max: 100, step: 1, format: signed,
    title: "Open up (right) or deepen the dark end where the gradient reaches." },
  { key: "temperature", label: "Temp", min: -100, max: 100, step: 1, format: signed,
    title: "Cooler (left) or warmer where the gradient reaches." },
  { key: "saturation", label: "Saturation", min: -200, max: 200, step: 1, format: signed,
    title: "More or less colour where the gradient reaches, added to the Saturation slider's: with the picture at -100, +200 here is +100 where the gradient is whole." },
  { key: "clarity", label: "Clarity", min: -100, max: 100, step: 1, format: signed,
    title: "Mid-tone local contrast where the gradient reaches." },
];
const MASK_KEYS = MASK_SLIDERS.map((d) => d.key);
const FEATHER_DEF = { key: "feather", label: "Feather", min: 0, max: 100, step: 1, format: (v) => `${v}`,
  title: "How much of the ellipse the fade takes: 0 is a hard edge, 100 fades from the centre." };
const ANGLE_DEF = { key: "angle", label: "Angle", min: -90, max: 90, step: 1, format: (v) => `${v}°`,
  title: "Turn the ellipse. The handle above it on the picture does the same." };
// A mask's section (crop) fades in from each of its edges, as a share of its
// shorter side: 0 is a hard edge, 100 fades from the middle.
const SECTION_FEATHER_DEF = { key: "feather", label: "Feather", min: 0, max: 100, step: 1, format: (v) => `${v}`,
  title: "How far inside the section's edges the mask fades in, as a share of its shorter side: 0 is a hard edge and 100 fades from the middle." };

// A mask's range (backend adjust/masks.py): only where the picture, before the
// edit, is within a band of brightness, or near a colour sampled from it.
const RANGE_LUM_DEFS = [
  { key: "lo", label: "From", min: 0, max: 100, step: 1, format: (v) => `${v}`,
    title: "The darkest the picture may be and still be reached: 0 is black." },
  { key: "hi", label: "To", min: 0, max: 100, step: 1, format: (v) => `${v}`,
    title: "The brightest it may be: 100 is white." },
  { key: "smooth", label: "Smooth", min: 0, max: 100, step: 1, format: (v) => `${v}`,
    title: "How gently it fades out beyond the band." },
];
const RANGE_COLOUR_DEF = { key: "width", label: "Range", min: 0, max: 100, step: 1, format: (v) => `${v}`,
  title: "How wide a range of colours round the one picked is reached." };
// The colour a range was sampled at, back to RGB for its swatch (BT.709, as the backend).
function rangeSwatch(r) {
  const y = (r.lum ?? 50) / 100;
  const red = y + 1.5748 * r.cr, blue = y + 1.8556 * r.cb;
  const green = (y - 0.2126 * red - 0.0722 * blue) / 0.7152;
  const c = (v) => Math.round(255 * Math.min(1, Math.max(0, v)));
  return `rgb(${c(red)}, ${c(green)}, ${c(blue)})`;
}

// The edit's geometry before the crop (lens and transform), which the mask view's
// requests carry so what they find is on the picture as it is shown.
function frameGeometry(settings) {
  return Object.fromEntries(["lens_profile", "distortion", "tf_vertical", "tf_horizontal", "tf_aspect"]
    .map((k) => [k, Number(settings[k]) || 0]));
}

function maskIsActive(m) {
  return MASK_KEYS.some((k) => Number(m?.[k] || 0) !== 0);
}

function maskName(m, i) {
  const set = MASK_SLIDERS.filter((d) => Number(m[d.key] || 0) !== 0)
    .map((d) => `${d.label} ${d.format(m[d.key])}`);
  const names = MASK_NAMES[m.kind];
  const kind = names ? names[m.invert ? 1 : 0] : m.kind;
  const who = (m.person ? ` (person ${m.person})` : "")
    + (m.range ? ` · ${m.range.kind === "color" ? "colour" : "brightness"} range` : "")
    + (m.section ? " · section" : "");
  return `${kind}${who} ${i + 1}${set.length ? ` · ${set.slice(0, 2).join(", ")}${set.length > 2 ? "…" : ""}` : ""}`;
}

// A new gradient where it is most often wanted: a linear one from the top (a
// sky), a radial one round in the middle of the frame.
// The masks a model selects, by their name and their name inverted.
const CHOSEN_NAMES = {
  subject: ["Subject", "Background"], sky: ["Sky", "All but sky"],
  people: ["People", "All but people"], faces: ["Faces", "All but faces"],
};
// Every kind by its name and its name inverted.
const MASK_NAMES = {
  linear: ["Linear", "Linear, turned round"], radial: ["Radial", "Outside radial"],
  brush: ["Brush", "All but brush"], ...CHOSEN_NAMES,
};
// What the invert box says for each kind.
function invertLabel(m) {
  if (m.kind === "linear") return "The other way round";
  if (m.kind === "radial") return "Outside the ellipse instead";
  if (m.kind === "brush") return "Everything but where it is painted";
  return `Everything but the ${m.person ? "person" : m.kind} instead`;
}
// The chosen kinds that can be narrowed to one person (1 = leftmost face).
const OF_PEOPLE = ["people", "faces"];
// Which selection a chosen mask shades with: one person's is its own picture.
const matteKey = (m) => (m.person ? `${m.kind}:${m.person}` : m.kind);

function newMask(kind, dims) {
  if (kind === "brush") return { kind, strokes: [] };
  if (CHOSEN_NAMES[kind]) return { kind, invert: false };
  if (kind === "linear") return { kind, x1: 0.5, y1: 0.0, x2: 0.5, y2: 0.4 };
  const rx = 0.2;
  return { kind, cx: 0.5, cy: 0.5, rx, ry: Math.min(0.45, (rx * dims.w) / dims.h),
    angle: 0, feather: 50, invert: false };
}

// Where a mask reaches, drawn the way the backend weighs it: its gradient as
// the base, then each stroke in order, painting laying white over what is there
// and erasing black, with its flow as opacity. A soft edge is a blur.
function MaskShade({ mask, extra, matte, range, W, H, unit }) {
  const id = useId().replace(/:/g, "");
  const strokes = [...(mask.strokes || []), ...(extra ? [extra] : [])];
  // An inverted brush is the whole frame less its paint: white to begin with,
  // painting laying black and erasing white, which is 1 - the painted weight.
  const flip = mask.kind === "brush" && !!mask.invert;
  const base = mask.kind === "linear" ? `url(#${id}l)` : mask.kind === "radial" ? `url(#${id}r)`
    : flip ? "white" : "black";
  const chosen = !!CHOSEN_NAMES[mask.kind];
  const inner = 1 - (mask.feather ?? 50) / 100;
  const [inside, outside] = mask.invert ? ["black", "white"] : ["white", "black"];
  // A section crops the final reach: the shape and the strokes inside its
  // rectangle and nothing outside it. Its edge fades in over the same share of
  // the shorter side the backend fades in over (a blurred rect, clipped to the
  // edge); 100 fades from the middle.
  const sec = mask.section ? {
    x: mask.section.x1 * W, y: mask.section.y1 * H,
    w: (mask.section.x2 - mask.section.x1) * W, h: (mask.section.y2 - mask.section.y1) * H,
    fade: ((mask.section.feather ?? 0) / 100)
      * Math.min((mask.section.x2 - mask.section.x1) * W, (mask.section.y2 - mask.section.y1) * H) / 2,
  } : null;
  return (
    <g pointerEvents="none">
      <defs>
        {mask.kind === "linear" && (
          <linearGradient id={`${id}l`} gradientUnits="userSpaceOnUse"
            x1={mask.x1 * W} y1={mask.y1 * H} x2={mask.x2 * W} y2={mask.y2 * H}>
            <stop offset="0" stopColor={mask.invert ? "black" : "white"} />
            <stop offset="1" stopColor={mask.invert ? "white" : "black"} />
          </linearGradient>
        )}
        {mask.kind === "radial" && (
          <radialGradient id={`${id}r`} gradientUnits="userSpaceOnUse" cx="0" cy="0" r="1"
            gradientTransform={`translate(${mask.cx * W} ${mask.cy * H}) rotate(${mask.angle || 0}) scale(${mask.rx * W} ${mask.ry * H})`}>
            <stop offset={Math.min(0.999, inner)} stopColor={inside} /><stop offset="1" stopColor={outside} />
          </radialGradient>
        )}
        {chosen && (
          <filter id={`${id}i`}>
            <feComponentTransfer>
              <feFuncR type="table" tableValues="1 0" /><feFuncG type="table" tableValues="1 0" />
              <feFuncB type="table" tableValues="1 0" />
            </feComponentTransfer>
          </filter>
        )}
        {strokes.map((st, k) => (
          <filter key={k} id={`${id}f${k}`} x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation={(st.size * unit * (st.feather ?? 50)) / 100 / 3} />
          </filter>
        ))}
        {sec && (
          <>
            <clipPath id={`${id}c`}>
              <rect x={sec.x} y={sec.y} width={sec.w} height={sec.h} />
            </clipPath>
            {sec.fade > 0 && (
              <filter id={`${id}b`} filterUnits="userSpaceOnUse" x="0" y="0" width={W} height={H}>
                <feGaussianBlur stdDeviation={sec.fade / 4} />
              </filter>
            )}
            <mask id={`${id}q`} maskUnits="userSpaceOnUse" x="0" y="0" width={W} height={H}>
              <g clipPath={`url(#${id}c)`}>
                <rect x={sec.x + sec.fade / 2} y={sec.y + sec.fade / 2}
                  width={Math.max(1, sec.w - sec.fade)} height={Math.max(1, sec.h - sec.fade)}
                  fill="white" filter={sec.fade > 0 ? `url(#${id}b)` : undefined} />
              </g>
            </mask>
          </>
        )}
        <mask id={`${id}m`} maskUnits="userSpaceOnUse" x="0" y="0" width={W} height={H}>
          <rect x="0" y="0" width={W} height={H} fill={base} />
          {chosen && matte && (
            <image href={matte} x="0" y="0" width={W} height={H} preserveAspectRatio="none"
              filter={mask.invert ? `url(#${id}i)` : undefined} />
          )}
          {strokes.map((st, k) => {
            const r = st.size * unit * (1 - (st.feather ?? 50) / 200);
            const pts = st.points.map(([x, y]) => `${x * W},${y * H}`).join(" ");
            const paint = !st.erase !== flip ? "white" : "black";
            const look = { opacity: (st.flow ?? 100) / 100, filter: st.feather ? `url(#${id}f${k})` : undefined };
            return st.points.length > 1
              ? <polyline key={k} points={pts} fill="none" stroke={paint} strokeWidth={2 * r}
                  strokeLinecap="round" strokeLinejoin="round" style={look} />
              : <circle key={k} cx={st.points[0][0] * W} cy={st.points[0][1] * H} r={r} fill={paint} style={look} />;
          })}
        </mask>
      </defs>
      {range && (
        <defs>
          <mask id={`${id}g`} maskUnits="userSpaceOnUse" x="0" y="0" width={W} height={H}>
            <image href={range} x="0" y="0" width={W} height={H} preserveAspectRatio="none" />
          </mask>
        </defs>
      )}
      <g mask={range ? `url(#${id}g)` : undefined}>
        <g mask={sec ? `url(#${id}q)` : undefined}>
          <rect x="0" y="0" width={W} height={H} fill="rgba(255,40,40,0.45)" mask={`url(#${id}m)`} />
        </g>
      </g>
    </g>
  );
}

function MaskOverlay({ masks, selected, dims, brush, showShade, mattes, people, range, picking, onPick,
  onSelect, onChange, onStroke }) {
  const ref = useRef(null);
  const W = dims.w, H = dims.h;
  const unit = Math.max(W, H);
  const handle = unit * 0.011;
  const [cursor, setCursor] = useState(null);       // where the brush is, in frame units
  const [wet, setWet] = useState(null);             // the stroke being painted
  const painting = !!brush.tool && selected >= 0 && !!masks[selected];

  const fraction = (ev) => {
    const rect = ref.current.getBoundingClientRect();
    return { x: (ev.clientX - rect.left) / rect.width, y: (ev.clientY - rect.top) / rect.height, rect };
  };

  // A stroke is the hand's path from press to release; it reaches the edit
  // (and the preview) once, when the hand lifts. Alt held erases.
  const paint = (e) => {
    e.preventDefault();
    e.stopPropagation();
    const first = fraction(e);
    const stroke = { points: [[first.x, first.y]], size: brush.size, feather: brush.feather,
      flow: brush.flow, ...(brush.tool === "erase" || e.altKey ? { erase: true } : {}) };
    let last = first;
    setWet({ ...stroke });
    const move = (ev) => {
      const p = fraction(ev);
      // A point every couple of screen pixels is finer than any brush needs.
      if (Math.hypot((p.x - last.x) * p.rect.width, (p.y - last.y) * p.rect.height) < 2) return;
      last = p;
      stroke.points.push([p.x, p.y]);
      setWet({ ...stroke, points: [...stroke.points] });
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      setWet(null);
      onStroke(selected, stroke);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  };

  // Drag one part of one gradient; the edit is changed as it moves (the preview
  // keeps up on its own, and the save waits for the hand to stop).
  const begin = (index, part) => (e) => {
    e.preventDefault();
    e.stopPropagation();
    onSelect(index);
    const rect = ref.current.getBoundingClientRect();
    const at = (ev) => ({ x: (ev.clientX - rect.left) / rect.width, y: (ev.clientY - rect.top) / rect.height });
    const start = { ...masks[index] };
    const origin = at(e);
    const move = (ev) => {
      const p = at(ev);
      const dx = p.x - origin.x, dy = p.y - origin.y;
      let next = { ...start };
      if (part === "p1") next = { ...next, x1: p.x, y1: p.y };
      else if (part === "p2") next = { ...next, x2: p.x, y2: p.y };
      else if (part === "line") next = { ...next, x1: start.x1 + dx, y1: start.y1 + dy, x2: start.x2 + dx, y2: start.y2 + dy };
      else if (part === "centre") next = { ...next, cx: start.cx + dx, cy: start.cy + dy };
      else if (part === "angle") {
        // The rotate handle points up when the angle is 0. The ellipse is the
        // same turned by half a turn, so the angle stays in [-90, 90], as its
        // slider does.
        const ux = (p.x - start.cx) * W, uy = (p.y - start.cy) * H;
        const a = (Math.atan2(ux, -uy) * 180) / Math.PI;
        next.angle = Math.round((((a + 90) % 180 + 180) % 180 - 90) * 10) / 10;
      } else {
        // A radius handle: how far the pointer is from the centre along that axis.
        const t = ((start.angle || 0) * Math.PI) / 180;
        const ux = (p.x - start.cx) * W, uy = (p.y - start.cy) * H;
        if (part === "rx") next.rx = Math.max(0.01, Math.abs(ux * Math.cos(t) + uy * Math.sin(t)) / W);
        else next.ry = Math.max(0.01, Math.abs(-ux * Math.sin(t) + uy * Math.cos(t)) / H);
      }
      onChange(index, next);
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  };

  // Drag the selected mask's section: a corner or an edge handle resizes it, the
  // inside moves it. Its edges are whole-frame fractions, as the mask's are.
  const beginSection = (index, part) => (e) => {
    e.preventDefault();
    e.stopPropagation();
    onSelect(index);
    const rect = ref.current.getBoundingClientRect();
    const at = (ev) => ({ x: (ev.clientX - rect.left) / rect.width, y: (ev.clientY - rect.top) / rect.height });
    const mask = { ...masks[index] };
    const start = { ...mask.section };
    const origin = at(e);
    const min = 0.02;
    const span = (low, high) => {
      low = Math.min(1 - min, Math.max(0, low));
      return [low, Math.min(1, Math.max(low + min, high))];
    };
    const move = (ev) => {
      const p = at(ev);
      let { x1, y1, x2, y2 } = start;
      if (part === "sec-move") {
        const w = x2 - x1, h = y2 - y1;
        x1 = Math.min(1 - w, Math.max(0, x1 + p.x - origin.x));
        y1 = Math.min(1 - h, Math.max(0, y1 + p.y - origin.y));
        x2 = x1 + w;
        y2 = y1 + h;
      } else {
        const dir = part.slice(4);
        if (dir.includes("w")) x1 = Math.min(p.x, x2 - min);
        if (dir.includes("e")) x2 = Math.max(p.x, x1 + min);
        if (dir.includes("n")) y1 = Math.min(p.y, y2 - min);
        if (dir.includes("s")) y2 = Math.max(p.y, y1 + min);
        [x1, x2] = span(x1, x2);
        [y1, y2] = span(y1, y2);
      }
      onChange(index, { ...mask, section: { ...start, x1, y1, x2, y2 } });
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  };

  // The selected mask's section, as two layers: the inside, dragged to move the
  // whole rectangle (below the mask's own handles, so those still win), and its
  // frame, a dashed rectangle with a handle on each corner and each edge.
  const sectionHit = (i) => {
    const s = masks[i]?.section;
    if (!s) return null;
    return (
      <rect key={`sech${i}`} x={s.x1 * W} y={s.y1 * H} width={(s.x2 - s.x1) * W} height={(s.y2 - s.y1) * H}
        fill="transparent" style={{ cursor: "move" }} onPointerDown={beginSection(i, "sec-move")} />
    );
  };

  const sectionFrame = (i) => {
    const s = masks[i]?.section;
    if (!s) return null;
    const x1 = s.x1 * W, y1 = s.y1 * H, x2 = s.x2 * W, y2 = s.y2 * H;
    const handles = [
      ["sec-nw", x1, y1, "nwse-resize"], ["sec-n", (x1 + x2) / 2, y1, "ns-resize"],
      ["sec-ne", x2, y1, "nesw-resize"], ["sec-e", x2, (y1 + y2) / 2, "ew-resize"],
      ["sec-se", x2, y2, "nwse-resize"], ["sec-s", (x1 + x2) / 2, y2, "ns-resize"],
      ["sec-sw", x1, y2, "nesw-resize"], ["sec-w", x1, (y1 + y2) / 2, "ew-resize"],
    ];
    const sw = unit * 0.0018;
    return (
      <g key={`secf${i}`} data-testid="mask-section-shape">
        <rect x={x1} y={y1} width={x2 - x1} height={y2 - y1} fill="none" stroke="rgba(255,255,255,0.9)"
          strokeWidth={sw} strokeDasharray={`${unit * 0.012} ${unit * 0.008}`} pointerEvents="none" />
        {handles.map(([part, hx, hy, cursor]) => (
          <circle key={part} cx={hx} cy={hy} r={handle * 0.85} fill="white" stroke="black"
            strokeWidth={unit * 0.002} style={{ cursor }} onPointerDown={beginSection(i, part)} />
        ))}
      </g>
    );
  };

  const dot = (x, y, part, index, fill = "white") => (
    <circle cx={x} cy={y} r={handle} fill={fill} stroke="black" strokeWidth={unit * 0.002}
      style={{ cursor: "grab" }} onPointerDown={begin(index, part)} />
  );

  return (
    <svg ref={ref} viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none"
      className="absolute inset-0 w-full h-full" data-testid="mask-overlay"
      onPointerDown={picking ? (e) => { e.preventDefault(); e.stopPropagation(); const p = fraction(e); onPick(p.x, p.y); } : undefined}
      style={{ touchAction: "none", cursor: picking ? "crosshair" : undefined }}
      onPointerMove={painting ? (e) => { const p = fraction(e); setCursor({ x: p.x * W, y: p.y * H }); } : undefined}
      onPointerLeave={() => setCursor(null)}>
      {(showShade || painting || picking) && masks[selected] && (
        <MaskShade mask={masks[selected]} extra={wet} matte={mattes[matteKey(masks[selected])]}
          range={masks[selected].range ? range : null} W={W} H={H} unit={unit} />
      )}
      {/* The people a people or faces mask can be narrowed to, numbered from the
          left as the backend counts them: a click chooses one, again everyone. */}
      {!painting && masks[selected] && OF_PEOPLE.includes(masks[selected].kind) && (people || []).map((f, n) => {
        const m = masks[selected];
        const on = m.person === n + 1;
        const r = Math.min(unit * 0.022, Math.max(unit * 0.012, f.w * W * 0.22));
        const cx = (f.x + f.w / 2) * W, cy = Math.max(r * 1.2, f.y * H - r * 1.1);
        return (
          <g key={`p${n}`} style={{ cursor: "pointer" }} data-testid={`person-${n + 1}`}
            onPointerDown={(e) => { e.preventDefault(); e.stopPropagation(); onChange(selected, { ...m, person: on ? 0 : n + 1 }); }}>
            <title>{on ? "Back to everyone" : `Only person ${n + 1}`}</title>
            <circle cx={cx} cy={cy} r={r} fill={on ? "rgba(251,191,36,0.95)" : "rgba(0,0,0,0.6)"}
              stroke="white" strokeWidth={unit * 0.002} />
            <text x={cx} y={cy} textAnchor="middle" dominantBaseline="central" fontSize={r * 1.2}
              fill={on ? "black" : "white"} style={{ userSelect: "none" }}>{n + 1}</text>
          </g>
        );
      })}
      {/* Picking a colour for a range takes every click, so the section waits. */}
      {!painting && !picking && selected >= 0 && masks[selected]?.section && sectionHit(selected)}
      {masks.map((m, i) => {
        if (m.kind === "brush" || CHOSEN_NAMES[m.kind]) return null;
        const on = i === selected;
        const stroke = on ? "rgba(255,255,255,0.95)" : "rgba(255,255,255,0.45)";
        const sw = unit * (on ? 0.0025 : 0.0018);
        if (m.kind === "linear") {
          const x1 = m.x1 * W, y1 = m.y1 * H, x2 = m.x2 * W, y2 = m.y2 * H;
          const len = Math.hypot(x2 - x1, y2 - y1) || 1;
          // The band's two edges run across the direction of the fade.
          const px = (-(y2 - y1) / len) * unit * 2, py = ((x2 - x1) / len) * unit * 2;
          return (
            <g key={i}>
              <line x1={x1 - px} y1={y1 - py} x2={x1 + px} y2={y1 + py} stroke={stroke} strokeWidth={sw} />
              <line x1={x2 - px} y1={y2 - py} x2={x2 + px} y2={y2 + py} stroke={stroke} strokeWidth={sw}
                strokeDasharray={`${unit * 0.012} ${unit * 0.008}`} />
              <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="transparent" strokeWidth={unit * 0.02}
                style={{ cursor: "move" }} onPointerDown={begin(i, "line")} />
              <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={stroke} strokeWidth={sw} pointerEvents="none" />
              {on ? <>{dot(x1, y1, "p1", i)}{dot(x2, y2, "p2", i, "rgba(255,255,255,0.4)")}</>
                : dot(x1, y1, "line", i, "rgba(255,255,255,0.5)")}
            </g>
          );
        }
        const cx = m.cx * W, cy = m.cy * H, rx = m.rx * W, ry = m.ry * H;
        const a = m.angle || 0, t = (a * Math.PI) / 180;
        const inner = 1 - (m.feather ?? 50) / 100;
        return (
          <g key={i}>
            <ellipse cx={cx} cy={cy} rx={rx} ry={ry} transform={`rotate(${a} ${cx} ${cy})`}
              fill="none" stroke={stroke} strokeWidth={sw} />
            {on && inner > 0.02 && (
              <ellipse cx={cx} cy={cy} rx={rx * inner} ry={ry * inner} transform={`rotate(${a} ${cx} ${cy})`}
                fill="none" stroke={stroke} strokeWidth={sw} strokeDasharray={`${unit * 0.01} ${unit * 0.008}`}
                pointerEvents="none" />
            )}
            {dot(cx, cy, "centre", i, on ? "white" : "rgba(255,255,255,0.5)")}
            {on && dot(cx + rx * Math.cos(t), cy + rx * Math.sin(t), "rx", i)}
            {on && dot(cx - ry * Math.sin(t), cy + ry * Math.cos(t), "ry", i)}
            {on && (() => {
              // The rotate handle sits beyond the ellipse's top edge, on its
              // short axis: dragging it round the centre turns the ellipse.
              const rop = ry + unit * 0.04;
              const hx = cx + rop * Math.sin(t), hy = cy - rop * Math.cos(t);
              return (
                <g>
                  <line x1={cx + ry * Math.sin(t)} y1={cy - ry * Math.cos(t)} x2={hx} y2={hy}
                    stroke={stroke} strokeWidth={sw} pointerEvents="none" />
                  <circle cx={hx} cy={hy} r={handle} fill="white" stroke="black" strokeWidth={unit * 0.002}
                    style={{ cursor: "grab" }} onPointerDown={begin(i, "angle")} data-testid="mask-rotate">
                    <title>Turn the ellipse: drag round its centre (the Angle slider follows)</title>
                  </circle>
                </g>
              );
            })()}
          </g>
        );
      })}
      {!painting && !picking && selected >= 0 && masks[selected]?.section && sectionFrame(selected)}
      {painting && (
        <rect x="0" y="0" width={W} height={H} fill="transparent" style={{ cursor: "none" }}
          onPointerDown={paint} data-testid="brush-surface" />
      )}
      {painting && cursor && (
        <g pointerEvents="none">
          <circle cx={cursor.x} cy={cursor.y} r={brush.size * unit} fill="none"
            stroke="white" strokeWidth={unit * 0.0018} style={{ filter: "drop-shadow(0 0 1px black)" }} />
          {brush.feather > 0 && brush.feather < 100 && (
            <circle cx={cursor.x} cy={cursor.y} r={brush.size * unit * (1 - brush.feather / 100)} fill="none"
              stroke="white" strokeWidth={unit * 0.0012} strokeDasharray={`${unit * 0.006} ${unit * 0.005}`} />
          )}
          {brush.tool === "erase" && (
            <line x1={cursor.x - unit * 0.006} y1={cursor.y} x2={cursor.x + unit * 0.006} y2={cursor.y}
              stroke="white" strokeWidth={unit * 0.002} />
          )}
        </g>
      )}
    </svg>
  );
}

const BRUSH_DEFS = [
  { key: "size", label: "Size", min: 0.05, max: 25, step: 0.05, scale: 100, tool: true,
    format: (v) => `${Number(v.toFixed(2))}%`,
    title: "The brush's radius, as a share of the picture's longer side, from 0.05% (a few pixels, for one edge by hand) to 25%. [ and ] change it." },
  { key: "feather", label: "Feather", min: 0, max: 100, step: 1, scale: 1, tool: true, format: (v) => `${v}`,
    title: "How soft the brush's edge is: 0 is hard, 100 fades from the middle." },
  { key: "flow", label: "Flow", min: 1, max: 100, step: 1, scale: 1, tool: true, format: (v) => `${v}`,
    title: "How much one stroke lays down: below 100, going over a place again builds it up." },
];

function BrushControls({ mask, brush, onBrush, onUndo, onClear }) {
  const strokes = mask.strokes?.length || 0;
  const tool = (id, label, title) => (
    <button onClick={() => onBrush({ tool: brush.tool === id ? null : id })} title={title}
      className={`text-[11px] px-2 py-0.5 rounded border ${brush.tool === id
        ? "border-amber-400 text-amber-300" : "border-neutral-700 text-neutral-300 hover:text-white hover:border-neutral-500"}`}>
      {label}
    </button>
  );
  return (
    <div className="space-y-1.5" data-testid="brush-controls">
      <div className="flex items-center gap-1">
        {tool("paint", "✎ Paint", mask.kind === "brush"
          ? "Paint where this brush reaches. Hold Alt to erase as you go."
          : "Paint more of the picture into this gradient. Hold Alt to erase as you go.")}
        {tool("erase", "⌫ Erase", mask.kind === "brush"
          ? "Take paint away."
          : "Take part of the picture out of this gradient: the sky beside a face, a tree in a sky.")}
        <span className="flex-1" />
        {strokes > 0 && (
          <>
            <button onClick={onUndo} title="Take back the last stroke"
              className="text-[11px] text-neutral-500 hover:text-white">↶ stroke</button>
            <button onClick={onClear} title="Take back every stroke on this one"
              className="text-[11px] text-neutral-500 hover:text-red-400">clear</button>
          </>
        )}
      </div>
      {brush.tool && (
        <Controls>
          {BRUSH_DEFS.map((def) => (
            <Slider key={def.key} def={def}
              value={Number((Math.round((brush[def.key] * def.scale) / def.step) * def.step).toFixed(4))}
              onChange={(key, v) => onBrush({ [key]: v / def.scale })} onReset={() => {}} />
          ))}
        </Controls>
      )}
    </div>
  );
}

// ── Match a look ───────────────────────────────────────────────────
//
// The backend (adjust/match.py) answers with channel curves and a saturation,
// or black and white with a curve, at full strength; the amount takes them part
// of the way from no change. What the match sets, it owns: the other curves and
// the treatment are put back to none so two matches never stack.

const MATCH_KEYS = ["curve", "curve_r", "curve_g", "curve_b", "saturation", "mono"];

function matchAt(found, amount) {
  const out = Object.fromEntries(MATCH_KEYS.map((k) => [k, ZERO[k]]));
  for (const [key, value] of Object.entries(found)) {
    if (key.startsWith("curve")) out[key] = value.map(([x, y]) => [x, Math.round(x + amount * (y - x))]);
    else if (key === "saturation") out[key] = Math.round(value * amount);
    else out[key] = value;
  }
  return out;
}

// What a match was fitted on: the controls under its curves. A match is measured
// once, on the frame as it was then, so when one of these changes the curves no
// longer fit it and the panel offers to match again.
const MATCH_BASE_KEYS = ["flat", "style", "style_fade", "exposure", "temperature", "tint"];
const matchBase = (settings) => JSON.stringify(MATCH_BASE_KEYS.map((k) => settings?.[k] ?? 0));

function MatchPanel({ filepath, reference, match, busy, error, stale, onPickThis, onFile, onForget, onMatch, onAmount }) {
  const fileRef = useRef(null);
  const self = reference?.filepath === filepath;
  return (
    <div className="space-y-1.5" data-testid="match-panel">
      <div className="text-[10px] uppercase tracking-wide text-neutral-600"
        title="Give this frame the tone and colour of another picture: best a frame of the same shoot, whose light or exposure differed. The result is ordinary curves and saturation, so it can be changed like any edit.">
        Match a look
      </div>
      {reference ? (
        <div className="flex items-center gap-1 text-[11px]">
          <span className="text-neutral-500">From</span>
          <span className="flex-1 truncate text-neutral-300" title={reference.name}>{reference.name}</span>
          <button onClick={onForget} className="text-neutral-600 hover:text-red-400" title="Choose another reference">✕</button>
        </div>
      ) : (
        <p className="text-[11px] text-neutral-500">Pick a reference: this photo (then go to another), or a picture from a file.</p>
      )}
      <div className="flex flex-wrap gap-1">
        {!self && (
          <button onClick={onPickThis}
            title="Remember this photo, as it is edited now, as the look to match on other frames"
            className="text-[11px] px-2 py-0.5 rounded border border-neutral-700 text-neutral-300 hover:text-white hover:border-neutral-500">
            Use this photo
          </button>
        )}
        <button onClick={() => fileRef.current?.click()} title="A picture from elsewhere: a JPEG or PNG"
          className="text-[11px] px-2 py-0.5 rounded border border-neutral-700 text-neutral-300 hover:text-white hover:border-neutral-500">
          From a file…
        </button>
        <input ref={fileRef} type="file" accept="image/jpeg,image/png,image/webp" className="hidden"
          onChange={(e) => { const f = e.target.files?.[0]; e.target.value = ""; if (f) onFile(f); }} />
        {reference && !self && (
          <button onClick={onMatch} disabled={busy} data-testid="match-run"
            className="text-[11px] px-2 py-0.5 rounded border border-amber-500/60 text-amber-300 hover:text-amber-200 disabled:opacity-50">
            {busy ? "Matching…" : "Match"}
          </button>
        )}
      </div>
      {error && <p className="text-[11px] text-red-400">{error}</p>}
      {match && (
        <>
          <Controls>
            <Slider def={MATCH_AMOUNT_DEF} value={match.amount} onChange={(_, v) => onAmount(v)}
              onReset={() => onAmount(100)} />
          </Controls>
          {match.note && <p className="text-[11px] text-neutral-500">Matched: {match.note}.</p>}
          {match.elsewhere && (
            <p className="text-[11px] text-amber-300/80">
              The reference is from another shoot. Match copies a picture's content as well as its look,
              so a different scene is pushed towards that scene. To carry an edit to other scenes, copy and paste it.
            </p>
          )}
          {stale && (
            <p className="text-[11px] text-amber-300/80 flex items-center gap-2">
              <span>Camera look, exposure or white balance changed since this match.</span>
              <button onClick={onMatch} disabled={busy} data-testid="match-again"
                className="px-1.5 py-0.5 rounded border border-amber-500/60 text-amber-300 hover:text-amber-200 disabled:opacity-50">
                Match again
              </button>
            </p>
          )}
        </>
      )}
    </div>
  );
}

// ── Spot removal ───────────────────────────────────────────────────
//
// Each spot a circle on the whole frame, painted over from a source circle
// nearby (the backend picks one when the spot is placed; drag it elsewhere).

const SPOT_SIZE = { min: 0.002, max: 0.1 };
const MATCH_AMOUNT_DEF = { key: "amount", label: "Amount", min: 0, max: 100, step: 1, tool: true,
  format: (v) => `${v}`, title: "How far toward the reference's look" };
const SPOT_SIZE_DEF = { key: "r", label: "Size", min: SPOT_SIZE.min * 1000, max: SPOT_SIZE.max * 1000, step: 1,
  tool: true, format: (v) => `${Math.round(v) / 10}%`,
  title: "The size of the next spot, or of the one selected: a share of the picture's longer side. [ and ] change it." };
const SPOT_DEFS = [
  { key: "feather", label: "Feather", min: 0, max: 100, step: 1, tool: true, dflt: 50, format: (v) => `${v}`,
    title: "How soft the spot's edge is" },
  { key: "opacity", label: "Opacity", min: 1, max: 100, step: 1, tool: true, dflt: 100, format: (v) => `${v}`,
    title: "How much of the spot is painted over" },
];

function SpotOverlay({ spots, selected, dims, size, onSelect, onAdd, onChange }) {
  const ref = useRef(null);
  const W = dims.w, H = dims.h;
  const unit = Math.max(W, H);
  const [cursor, setCursor] = useState(null);
  const at = (ev) => {
    const rect = ref.current.getBoundingClientRect();
    return { x: (ev.clientX - rect.left) / rect.width, y: (ev.clientY - rect.top) / rect.height };
  };

  // Drag a spot (its source stays where it is, as in Lightroom) or its source.
  const begin = (index, part) => (e) => {
    e.preventDefault();
    e.stopPropagation();
    onSelect(index);
    const start = { ...spots[index] };
    const origin = at(e);
    let moved = false;
    const move = (ev) => {
      const p = at(ev);
      const dx = p.x - origin.x, dy = p.y - origin.y;
      if (!moved && Math.hypot(dx * W, dy * H) < 2) return;
      moved = true;
      const clamp = (v) => Math.min(1, Math.max(0, v));
      onChange(index, part === "source"
        ? { ...start, sx: clamp(start.sx + dx), sy: clamp(start.sy + dy) }
        : { ...start, x: clamp(start.x + dx), y: clamp(start.y + dy) });
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  };

  return (
    <svg ref={ref} viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none"
      className="absolute inset-0 w-full h-full" style={{ touchAction: "none" }} data-testid="spot-overlay"
      onPointerMove={(e) => { const p = at(e); setCursor({ x: p.x * W, y: p.y * H }); }}
      onPointerLeave={() => setCursor(null)}>
      <rect x="0" y="0" width={W} height={H} fill="transparent" style={{ cursor: "none" }}
        onPointerDown={(e) => { e.preventDefault(); const p = at(e); onAdd(p.x, p.y); }} />
      {spots.map((sp, i) => {
        const on = i === selected;
        const r = sp.r * unit;
        const x = sp.x * W, y = sp.y * H, sx = sp.sx * W, sy = sp.sy * H;
        const sw = unit * (on ? 0.0022 : 0.0016);
        const stroke = on ? "white" : "rgba(255,255,255,0.6)";
        return (
          <g key={i} style={{ filter: "drop-shadow(0 0 1px black)" }}>
            {on && (
              <>
                <line x1={x} y1={y} x2={sx} y2={sy} stroke="rgba(255,255,255,0.7)" strokeWidth={unit * 0.0012}
                  pointerEvents="none" />
                <circle cx={sx} cy={sy} r={r} fill="rgba(255,255,255,0.08)" stroke="white" strokeWidth={sw}
                  strokeDasharray={`${unit * 0.006} ${unit * 0.004}`} style={{ cursor: "move" }}
                  onPointerDown={begin(i, "source")} />
              </>
            )}
            <circle cx={x} cy={y} r={r} fill="rgba(255,255,255,0.08)" stroke={stroke} strokeWidth={sw}
              style={{ cursor: "move" }} onPointerDown={begin(i, "spot")} />
          </g>
        );
      })}
      {cursor && (
        <circle cx={cursor.x} cy={cursor.y} r={size * unit} fill="none" stroke="white"
          strokeWidth={unit * 0.0015} pointerEvents="none" style={{ filter: "drop-shadow(0 0 1px black)" }} />
      )}
    </svg>
  );
}

function SpotPanel({ spots, selected, spotMode, size, mode, busy, onSize, onMode, onToggle,
  onSelect, onRemove, onClear, onChange }) {
  const sp = selected >= 0 ? spots[selected] : null;
  const set = (key, value) => onChange(selected, { ...sp, [key]: value });
  return (
    <div className="space-y-2" data-testid="spot-panel">
      <div className="flex items-center justify-between text-[10px] uppercase tracking-wide text-neutral-600">
        <span title="Paint out dust, a blemish or a wire from a clean place nearby. Done before every other control.">
          Spot removal{spots.length ? ` · ${spots.length}` : ""}
        </span>
        {spotMode && (
          <button onClick={onToggle} className="normal-case tracking-normal text-amber-400 hover:text-amber-300"
            title="Back to the picture as framed (Q or Esc)">done</button>
        )}
      </div>
      <div className="flex items-center gap-1">
        <button onClick={onToggle} data-testid="spot-toggle"
          title="Click on the picture where a spot is; drag the circle or its source to move them (Q)"
          className={`text-[11px] px-2 py-0.5 rounded border ${spotMode
            ? "border-amber-400 text-amber-300" : "border-neutral-700 text-neutral-300 hover:text-white hover:border-neutral-500"}`}>
          ◎ {spotMode ? "Removing spots" : "Remove spots"}
        </button>
        {spotMode && ["heal", "clone"].map((m) => (
          <button key={m} onClick={() => (sp ? set("mode", m) : onMode(m))}
            title={m === "heal" ? "The source's texture in the spot's own light and colour"
              : "The source exactly as it is"}
            className={`text-[11px] px-2 py-0.5 rounded border ${(sp ? sp.mode : mode) === m
              ? "border-amber-400 text-amber-300" : "border-neutral-700 text-neutral-400 hover:text-white"}`}>
            {m === "heal" ? "Heal" : "Clone"}
          </button>
        ))}
        <span className="flex-1" />
        {busy && <span className="text-[10px] text-neutral-500">finding a source…</span>}
        {spots.length > 0 && !busy && (
          <button onClick={onClear} className="text-[11px] text-neutral-600 hover:text-red-400"
            title="Take back every spot on this frame">clear</button>
        )}
      </div>
      {spotMode && (
        <div className="space-y-1.5">
          <Controls>
            <Slider def={SPOT_SIZE_DEF} value={Math.round((sp ? sp.r : size) * 1000)} onReset={() => {}}
              onChange={(_, v) => (sp ? set("r", v / 1000) : onSize(v / 1000))} />
            {sp && SPOT_DEFS.map((def) => (
              <Slider key={def.key} def={def} value={sp[def.key] ?? def.dflt}
                onChange={set} onReset={() => set(def.key, def.dflt)} />
            ))}
          </Controls>
          {sp && (
            <>
              <div className="flex items-center justify-between text-[11px]">
                <span className="text-neutral-500">Spot {selected + 1} of {spots.length}</span>
                <button onClick={() => onRemove(selected)} className="text-neutral-500 hover:text-red-400"
                  title="Take this spot back (Delete)">✕ remove</button>
              </div>
            </>
          )}
          {!spots.length && (
            <p className="text-[11px] text-neutral-500">Click on a speck of dust or a blemish. The loupe (after "done") shows it at 1:1.</p>
          )}
        </div>
      )}
    </div>
  );
}

function HistoryPanel({ history, onJump }) {
  const [open, setOpen] = useState(false);
  const { entries, index } = history;
  if (entries.length < 2) return null;
  return (
    <div data-testid="history-panel">
      <button onClick={() => setOpen((o) => !o)}
        className="text-[10px] uppercase tracking-wide text-neutral-600 hover:text-neutral-400"
        title="Every step since this frame was opened. Click one to go back to it; a change from there drops the steps after it.">
        {open ? "▾" : "▸"} History · {entries.length - 1} step{entries.length === 2 ? "" : "s"}
      </button>
      {open && (
        <div className="mt-1 max-h-48 overflow-y-auto space-y-px">
          {entries.map((e, i) => i).reverse().map((i) => (
            <button key={i} onClick={() => onJump(i)}
              className={`block w-full text-left text-[11px] truncate rounded px-1 ${i === index
                ? "bg-neutral-800 text-amber-300" : i > index ? "text-neutral-600 italic" : "text-neutral-400 hover:text-white"}`}>
              {entries[i].label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function MaskPanel({ masks, selected, maskMode, brush, showShade, onShowShade, onBrush,
  choosing, people, picking, onPicking, onSelect, onAdd, onRemove, onChange, onDone }) {
  const m = selected >= 0 ? masks[selected] : null;
  const set = (key, value) => onChange(selected, { ...m, [key]: value });
  return (
    <div className="space-y-2" data-testid="mask-panel">
      <div className="flex items-center justify-between text-[10px] uppercase tracking-wide text-neutral-600">
        <span title="Change part of the picture: a linear gradient (a sky, a foreground), a radial one (a face, a vignette of your own) or a brush, where you paint. Each has its own exposure, tone and colour.">Gradients &amp; brushes</span>
        {maskMode && (
          <button onClick={onDone} className="normal-case tracking-normal text-amber-400 hover:text-amber-300"
            title="Back to the picture as framed (Esc)">done</button>
        )}
      </div>
      <div className="flex flex-wrap gap-1">
        {[["linear", "+ Linear", "Whole at one end, fading out to the other: drag its two ends on the picture."],
          ["radial", "+ Radial", "Whole inside an ellipse, fading out at its edge: drag its centre and its edges."],
          ["brush", "+ Brush", "Only where you paint it: a face, a doorway, a tree."],
          ["subject", "+ Subject", "The picture's subject, as a model finds it: the runners in front, not the crowd. Invert it for the background."],
          ["sky", "+ Sky", "The sky, as a model finds it. Invert it for everything below."],
          ["people", "+ People", "Everyone in the picture, or one of them: pick a number on a face. Invert it for the background."],
          ["faces", "+ Faces", "The faces in the picture, or one person's: lift them with exposure, soften skin with negative clarity."]].map(([kind, label, title]) => (
          <button key={kind} onClick={() => onAdd(kind)} title={title} disabled={!!choosing.busy}
            className="text-[11px] px-2 py-0.5 rounded border border-neutral-700 text-neutral-300 hover:text-white hover:border-neutral-500 disabled:opacity-50">
            {choosing.busy === kind ? "finding…" : label}
          </button>
        ))}
      </div>
      {choosing.error && <p className="text-[11px] text-red-400">{choosing.error}</p>}
      {masks.length > 0 && (
        <div className="space-y-0.5">
          {masks.map((mask, i) => (
            <div key={i} className={`flex items-center gap-1 text-[11px] rounded px-1 ${i === selected && maskMode ? "bg-neutral-800" : ""}`}>
              <button onClick={() => onSelect(i)}
                className={`flex-1 text-left truncate ${maskIsActive(mask) ? "text-amber-300" : "text-neutral-400"} hover:text-white`}>
                {maskName(mask, i)}
              </button>
              <button onClick={() => onRemove(i)} title="Remove this one"
                className="text-neutral-600 hover:text-red-400">✕</button>
            </div>
          ))}
        </div>
      )}
      {m && maskMode && (
        <div className="space-y-1.5 pt-1">
          <BrushControls mask={m} brush={brush} onBrush={onBrush}
            onUndo={() => onChange(selected, { ...m, strokes: (m.strokes || []).slice(0, -1) })}
            onClear={() => onChange(selected, { ...m, strokes: [] })} />
          <label className="flex items-center gap-2 text-xs text-neutral-400"
            title="Shade in red where this one reaches. It is always shown while painting.">
            <input type="checkbox" checked={showShade} onChange={(e) => onShowShade(e.target.checked)}
              className="accent-amber-400" />
            Show where it reaches
          </label>
          <Controls>
            {MASK_SLIDERS.map((def) => (
              <Slider key={def.key} def={def} value={Number(m[def.key]) || 0}
                onChange={(key, value) => set(key, value)} onReset={() => set(def.key, 0)} />
            ))}
          </Controls>
          {OF_PEOPLE.includes(m.kind) && (
            <label className="flex items-center gap-2 text-xs text-neutral-400" data-testid="mask-person"
              title="Everyone, or one person, counted from the left by their face. The numbers on the picture choose too.">
              <span className="w-20 shrink-0">Who</span>
              <select value={m.person || 0} onChange={(e) => set("person", Number(e.target.value))}
                className="flex-1 bg-neutral-900 border border-neutral-700 rounded px-1 py-0.5 text-xs text-neutral-200">
                <option value={0}>{people ? `Everyone (${people.length} found)` : "Everyone"}</option>
                {(people || []).map((_, n) => (
                  <option key={n} value={n + 1}>Person {n + 1}{n === 0 ? " (leftmost)" : ""}</option>
                ))}
              </select>
            </label>
          )}
          <label className="flex items-center gap-2 text-xs text-neutral-400" data-testid="mask-invert"
            title={m.kind === "brush" ? "Change the whole picture but where you paint: paint what to leave alone, erase to give it back." : undefined}>
            <input type="checkbox" checked={!!m.invert} onChange={(e) => set("invert", e.target.checked)}
              className="accent-amber-400" />
            {invertLabel(m)}
          </label>
          <div className="space-y-1.5" data-testid="mask-range">
            <label className="flex items-center gap-2 text-xs text-neutral-400"
              title="Narrow it to part of the picture: a band of brightness (a sky, the shadows) or a colour (a shirt, the leaves). Judged on the picture before the edit.">
              <span className="w-20 shrink-0">Range</span>
              <select value={m.range?.kind || ""} data-testid="mask-range-kind"
                onChange={(e) => {
                  const kind = e.target.value;
                  onPicking(kind === "color" && !(m.range?.kind === "color"));
                  if (kind) onShowShade(true);
                  // The bright half to begin with: a band that narrows something, so
                  // the shading shows at once what the sliders move.
                  set("range", kind === "luminance" ? { kind, lo: 50, hi: 100, smooth: 30 }
                    : kind === "color" ? (m.range?.kind === "color" ? m.range : null) : null);
                }}
                className="flex-1 bg-neutral-900 border border-neutral-700 rounded px-1 py-0.5 text-xs text-neutral-200">
                <option value="">All of it</option>
                <option value="luminance">Brightness</option>
                <option value="color">Colour</option>
              </select>
            </label>
            {m.range?.kind === "luminance" && (
              <Controls>
                {RANGE_LUM_DEFS.map((def) => (
                  <Slider key={def.key} def={def} value={Number(m.range[def.key]) || 0}
                    onChange={(key, value) => set("range", { ...m.range, [key]: value })}
                    onReset={() => set("range", { ...m.range, [def.key]: { lo: 0, hi: 100, smooth: 30 }[def.key] })} />
                ))}
              </Controls>
            )}
            {(m.range?.kind === "color" || picking) && (
              <div className="flex items-center gap-2 text-xs text-neutral-400">
                {m.range?.kind === "color" && (
                  <span className="w-4 h-4 rounded border border-neutral-600 shrink-0" data-testid="range-swatch"
                    style={{ background: rangeSwatch(m.range) }} />
                )}
                <button onClick={() => onPicking(!picking)} data-testid="range-pick"
                  className={`text-[11px] px-2 py-0.5 rounded border ${picking ? "border-amber-400 text-amber-300"
                    : "border-neutral-700 text-neutral-300 hover:text-white"}`}>
                  {picking ? "Click the colour on the picture…" : "Pick another colour"}
                </button>
              </div>
            )}
            {m.range?.kind === "color" && (
              <Controls>
                <Slider def={RANGE_COLOUR_DEF} value={Number(m.range.width) || 0}
                  onChange={(key, value) => set("range", { ...m.range, width: value })}
                  onReset={() => set("range", { ...m.range, width: 40 })} />
              </Controls>
            )}
          </div>
          <div className="space-y-1.5" data-testid="mask-section">
            <label className="flex items-center gap-2 text-xs text-neutral-400"
              title="Reach only inside a rectangle of the picture: drag its corners and edges, or drag inside it to move it. Strokes and invert stay inside it too.">
              <input type="checkbox" checked={!!m.section} data-testid="mask-section-on"
                onChange={(e) => {
                  // The red shading shows at once what the rectangle limits.
                  if (e.target.checked) onShowShade(true);
                  set("section", e.target.checked ? { x1: 0, y1: 0, x2: 1, y2: 1, feather: 0 } : null);
                }}
                className="accent-amber-400" />
              <span className="w-20 shrink-0">Section</span>
              {m.section && <span className="text-neutral-500">crop on the picture</span>}
            </label>
            {m.section && (
              <Controls>
                <Slider def={SECTION_FEATHER_DEF} value={Number(m.section.feather) || 0}
                  onChange={(key, value) => set("section", { ...m.section, feather: value })}
                  onReset={() => set("section", { ...m.section, feather: 0 })} />
              </Controls>
            )}
          </div>
          {m.kind === "radial" && (
            <>
              <Controls>
                <Slider def={FEATHER_DEF} value={m.feather ?? 50}
                  onChange={(key, value) => set(key, value)} onReset={() => set("feather", 50)} />
                <Slider def={ANGLE_DEF} value={m.angle || 0}
                  onChange={(key, value) => set(key, value)} onReset={() => set("angle", 0)} />
              </Controls>
            </>
          )}
        </div>
      )}
    </div>
  );
}

// ── History ────────────────────────────────────────────────────────
//
// Every state the edit has been in since the frame was opened, newest last,
// with what was done to reach it. Kept for the session per frame and copy, so
// stepping to the next frame and back keeps it. A drag is one step, not a
// hundred: a change of the same control within a moment of the last one
// replaces it.

const HISTORIES = new Map();
const MAX_HISTORY = 100;
const COALESCE_MS = 1500;

// An edit as the server keeps it, for asking whether two are the same: what
// is set, keys in order, numbers to the server's rounding (it keeps a stroke's
// points to four places).
function canonical(settings) {
  const round = (v) => (typeof v === "number" ? Math.round(v * 1e4) / 1e4
    : Array.isArray(v) ? v.map(round)
      : v && typeof v === "object" ? Object.fromEntries(Object.keys(v).sort().map((k) => [k, round(v[k])])) : v);
  return JSON.stringify(round(activeOnly(settings)));
}

function signedValue(v) {
  const n = Number(v);
  return `${n > 0 ? "+" : ""}${Number.isInteger(n) ? n : n.toFixed(2)}`;
}

// What one change of one control did, and which changes it may merge with.
function stepOf(key, value, prev) {
  const def = ALL_SLIDERS.find((d) => d.key === key);
  if (def) {
    const shown = def.invert ? def.max - value : value;
    return { label: `${def.label} ${def.format ? def.format(shown) : shown}`, group: key };
  }
  const hsl = key.match(/^(hue|sat|lum)_(.+)$/);
  if (hsl) {
    const kind = HSL_KINDS.find((k) => k.id === hsl[1]);
    const band = HSL_BANDS.find((b) => b.id === hsl[2]);
    return { label: `${band?.label || hsl[2]} ${kind?.label.toLowerCase() || hsl[1]} ${signedValue(value)}`, group: key };
  }
  if (key === "mono") return { label: value ? "Black & white" : "Colour", group: null };
  if (key === "lens_profile") return { label: value ? "Lens: camera's correction" : "Lens: as shot", group: null };
  if (key.startsWith("bw_")) return { label: `B&W ${key.slice(3)} ${signedValue(value)}`, group: key };
  const curve = CURVE_CHANNELS.find((c) => c.key === key);
  if (curve) return { label: `${curve.label === "RGB" ? "Tone" : curve.label} curve`, group: key };
  if (key === "style") return { label: value ? "Style" : "No style", group: null };
  if (key === "masks") return maskStep(prev.masks || [], value || []);
  if (key === "spots") return spotStep(prev.spots || [], value || []);
  if (key === "angle") return { label: `Level ${signedValue(value)}°`, group: key };
  return { label: key, group: key };
}

function maskStep(before, after) {
  if (after.length > before.length) return { label: `Added ${maskName(after[after.length - 1], after.length - 1).split(" · ")[0]}`, group: null };
  if (after.length < before.length) return { label: "Removed a mask", group: null };
  const i = after.findIndex((m, k) => JSON.stringify(m) !== JSON.stringify(before[k]));
  if (i < 0) return { label: "Masks", group: "masks" };
  const a = after[i], b = before[i];
  const name = maskName(a, i).split(" · ")[0];
  const strokes = (a.strokes || []).length, had = (b.strokes || []).length;
  if (strokes > had) return { label: `${name}: ${a.strokes[strokes - 1].erase ? "erase" : "brush"} stroke`, group: null };
  if (strokes < had) return { label: `${name}: ${strokes ? "stroke taken back" : "strokes cleared"}`, group: null };
  const moved = MASK_SLIDERS.find((d) => Number(a[d.key] || 0) !== Number(b[d.key] || 0));
  if (moved) return { label: `${name}: ${moved.label} ${moved.format(a[moved.key] || 0)}`, group: `masks:${i}:${moved.key}` };
  return { label: `${name}: moved`, group: `masks:${i}:shape` };
}

function spotStep(before, after) {
  if (!after.length && before.length > 1) return { label: "Spots cleared", group: null };
  if (after.length > before.length) return { label: `Spot ${after.length} removed`, group: null };
  if (after.length < before.length) return { label: "Spot taken back", group: null };
  const i = after.findIndex((sp, k) => JSON.stringify(sp) !== JSON.stringify(before[k]));
  const a = after[i] || {}, b = before[i] || {};
  const what = a.x !== b.x || a.y !== b.y ? "moved" : a.sx !== b.sx || a.sy !== b.sy ? "source moved"
    : a.mode !== b.mode ? (a.mode === "clone" ? "clone" : "heal") : "changed";
  return { label: `Spot ${i + 1} ${what}`, group: `spots:${i}:${what}` };
}

// A change that came all at once (a crop, a pasted edit) with no name of its own.
function diffLabel(prev, next) {
  const changed = Object.keys({ ...prev, ...next })
    .filter((k) => JSON.stringify(prev[k]) !== JSON.stringify(next[k]));
  if (changed.length === 1) return stepOf(changed[0], next[changed[0]], prev).label;
  if (changed.some((k) => k.startsWith("crop_"))) return "Crop";
  return `${changed.length} changes`;
}

// ── Control styles ───────────────────────────────────────────────────
//
// Bars, two a row with the name and value written inside, take half the room
// of a slider a row; the long slider stays one click away while the bars are
// tried. Both keep double-click to reset; a bar also takes the arrow keys when
// focused (Shift for ten steps). Picked at the top of the panel, kept on this
// device.
const CONTROL_STYLES = [
  { id: "bar", label: "Bars", note: "Thin bars with the name inside, two a row. Click or drag to set; Shift-drag for fine." },
  { id: "slider", label: "Slider", note: "One long slider a row" },
];
const CONTROLS_KEY = "photool.develop.control-style";
const ControlStyle = createContext("bar");

// How a group of controls is laid out in each style.
const CONTROL_LAYOUT = {
  slider: "space-y-2",
  bar: "grid grid-cols-2 gap-1.5",
};

function Controls({ children }) {
  return <div className={CONTROL_LAYOUT[useContext(ControlStyle)]}>{children}</div>;
}

function ControlStylePicker({ value, onChange }) {
  return (
    <div className="flex items-center gap-2 text-[10px]" data-testid="control-style">
      <span className="uppercase tracking-wide text-neutral-600">Controls</span>
      <div className="flex rounded border border-neutral-800 overflow-hidden">
        {CONTROL_STYLES.map((s) => (
          <button key={s.id} onClick={() => onChange(s.id)} title={s.note}
            className={`px-2 py-0.5 ${value === s.id ? "bg-neutral-800 text-white" : "text-neutral-500 hover:text-neutral-300"}`}>
            {s.label}
          </button>
        ))}
      </div>
    </div>
  );
}

// A value on the control's step and inside its range, without the float dust
// (0.1 + 0.2) a drag adds up.
function onStep(def, v) {
  const places = (String(def.step).split(".")[1] || "").length;
  const stepped = def.min + Math.round((v - def.min) / def.step) * def.step;
  return Number(clamp(stepped, def.min, def.max).toFixed(places));
}

// Dragging a control relative to where it was: the full range over `span`
// pixels, ten times finer with Shift. Arrow keys nudge it while it has focus.
function useNudge(def, shown, set, span) {
  const drag = useRef(null);
  const rate = (e) => (def.max - def.min) / span / (e.shiftKey ? 10 : 1);
  return {
    tabIndex: 0,
    onPointerDown(e) {
      if (e.button !== 0) return;
      e.currentTarget.setPointerCapture(e.pointerId);
      drag.current = { x: e.clientX, v: shown };
    },
    onPointerMove(e) {
      const d = drag.current;
      if (!d) return;
      const px = e.clientX - d.x;
      d.x = e.clientX;
      if (!px) return;
      d.v = clamp(d.v + px * rate(e), def.min, def.max);
      const next = onStep(def, d.v);
      if (next !== shown) set(next);
    },
    onPointerUp() { drag.current = null; },
    onKeyDown(e) {
      const dir = { ArrowRight: 1, ArrowUp: 1, ArrowLeft: -1, ArrowDown: -1 }[e.key];
      if (!dir) return;
      // The view's own arrows change frame; these are the control's.
      e.preventDefault();
      e.stopPropagation();
      set(onStep(def, shown + dir * def.step * (e.shiftKey ? 10 : 1)));
    },
  };
}

const HUE_GRADIENT = "linear-gradient(to right, hsl(0 80% 50%), hsl(60 80% 50%), hsl(120 80% 45%), hsl(180 80% 45%), hsl(240 80% 55%), hsl(300 80% 50%), hsl(360 80% 50%))";

function Slider(props) {
  const style = useContext(ControlStyle);
  const Drawn = style === "slider" ? LineSlider : BarControl;
  return <Drawn {...props} />;
}

// What every style shares: the value as shown (an inverted control from the
// other end), whether it is moved, and a setter that stores it the right way up.
// A tool's setting (a brush size) has no untouched state, so is never marked.
function controlState(def, value, onChange) {
  const stored = Number(value) || 0;
  const shown = def.invert ? def.max - stored : stored;
  const set = (v) => onChange(def.key, def.invert ? def.max - v : v);
  return { shown, set, changed: !def.tool && stored !== 0 };
}

// Where along the range a value sits, 0 to 1, and where the zero is (a signed
// control fills from its middle).
const fraction = (def, v) => (v - def.min) / (def.max - def.min);
const origin = (def) => (def.min < 0 && def.max > 0 ? fraction(def, 0) : 0);

// A bar half the panel wide, its name and value written on it. A click or a
// drag puts the value where the pointer is; Shift-drag nudges it finely.
function BarControl({ def, value, onChange, onReset }) {
  const { shown, set, changed } = controlState(def, value, onChange);
  const { tabIndex, ...nudge } = useNudge(def, shown, set, 600);
  const ref = useRef(null);
  const toPointer = (e) => {
    const r = ref.current.getBoundingClientRect();
    set(onStep(def, def.min + clamp((e.clientX - r.left) / r.width, 0, 1) * (def.max - def.min)));
  };
  const f = fraction(def, shown);
  const o = origin(def);
  return (
    <div ref={ref} tabIndex={tabIndex} onKeyDown={nudge.onKeyDown} data-testid={`bar-${def.key}`}
      title={`${def.title}\n\nDouble-click to reset.`}
      onPointerDown={(e) => {
        if (e.button !== 0) return;
        nudge.onPointerDown(e);
        if (!e.shiftKey) toPointer(e);
      }}
      onPointerMove={(e) => {
        if (!e.currentTarget.hasPointerCapture(e.pointerId)) return;
        if (e.shiftKey) nudge.onPointerMove(e); else toPointer(e);
      }}
      onPointerUp={nudge.onPointerUp}
      onDoubleClick={onReset}
      className="relative h-6 rounded bg-neutral-800/80 overflow-hidden cursor-ew-resize select-none touch-none focus:outline-none focus-visible:ring-1 focus-visible:ring-amber-400">
      {def.hue && <div className="absolute inset-0 opacity-35" style={{ background: HUE_GRADIENT }} />}
      {!def.hue && (
        <div className={`absolute inset-y-0 ${changed ? "bg-amber-500/25" : "bg-neutral-600/30"}`}
          style={{ left: `${Math.min(f, o) * 100}%`, width: `${Math.abs(f - o) * 100}%` }} />
      )}
      <div className={`absolute inset-y-0 w-0.5 ${changed ? "bg-amber-300" : "bg-neutral-400"}`}
        style={{ left: `calc(${f * 100}% - 1px)` }} />
      <div className="relative h-full flex items-center justify-between gap-1 px-1.5 text-[11px] pointer-events-none">
        <span className={`flex items-center gap-1 truncate ${changed ? "text-amber-200" : "text-neutral-300"}`}>
          {def.swatch && <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ background: def.swatch }} />}
          {def.label}
        </span>
        <span className={`font-mono ${changed ? "text-amber-200" : "text-neutral-500"}`}>{def.format(shown)}</span>
      </div>
    </div>
  );
}

function LineSlider({ def, value, onChange, onReset }) {
  // An inverted control is stored as the distance from its default, so the
  // untouched state is absent from the edit, but shown from the other end.
  const shown = def.invert ? def.max - value : value;
  const changed = !def.tool && Number(value) !== 0;
  return (
    <label className="flex items-center gap-2 text-xs">
      <span
        className={`w-20 shrink-0 cursor-default select-none flex items-center gap-1.5 ${changed ? "text-amber-300" : "text-neutral-400"}`}
        title={`${def.title}\n\nDouble-click to reset this control.`}
        onDoubleClick={onReset}
      >
        {def.swatch && <span className="w-2 h-2 rounded-full shrink-0" style={{ background: def.swatch }} />}
        {def.label}
      </span>
      <input
        type="range" min={def.min} max={def.max} step={def.step} value={shown}
        onChange={(e) => onChange(def.key, def.invert ? def.max - +e.target.value : +e.target.value)}
        onDoubleClick={onReset}
        title={def.title}
        style={def.hue ? { background: HUE_GRADIENT } : undefined}
        className={`flex-1 accent-amber-400 ${def.hue ? "h-1.5 rounded appearance-none" : ""}`}
      />
      <span
        className={`w-12 text-right font-mono cursor-default ${changed ? "text-amber-300" : "text-neutral-500"}`}
        onDoubleClick={onReset}
      >
        {def.format(shown)}
      </span>
    </label>
  );
}

// ── Tone curve ───────────────────────────────────────────────────────

// The same monotone cubic the backend renders with (adjust.render.curve_table),
// so the line drawn is the curve applied: smooth through every point, never
// overshooting between them, flat beyond the first and last.
function curveSampler(points) {
  const xs = points.map((p) => p[0] / 255);
  const ys = points.map((p) => p[1] / 255);
  const n = xs.length;
  const h = [];
  const d = [];
  for (let i = 0; i < n - 1; i++) {
    h.push(xs[i + 1] - xs[i]);
    d.push((ys[i + 1] - ys[i]) / h[i]);
  }
  const m = new Array(n);
  m[0] = d[0];
  m[n - 1] = d[n - 2];
  for (let i = 1; i < n - 1; i++) m[i] = (d[i - 1] + d[i]) / 2;
  for (let i = 0; n > 2 && i < n - 1; i++) {
    if (d[i] === 0) { m[i] = 0; m[i + 1] = 0; continue; }
    let a = m[i] / d[i];
    let b = m[i + 1] / d[i];
    if (a < 0) { m[i] = 0; a = 0; }
    if (b < 0) { m[i + 1] = 0; b = 0; }
    const r = a * a + b * b;
    if (r > 9) {
      const t = 3 / Math.sqrt(r);
      m[i] = t * a * d[i];
      m[i + 1] = t * b * d[i];
    }
  }
  return (v) => {
    const x = clamp(v, xs[0], xs[n - 1]);
    if (n === 2) return clamp(ys[0] + (ys[1] - ys[0]) * ((x - xs[0]) / (h[0] || 1)), 0, 1);
    let i = 0;
    while (i < n - 2 && x > xs[i + 1]) i++;
    const t = (x - xs[i]) / h[i];
    const t2 = t * t;
    const t3 = t2 * t;
    const y = (2 * t3 - 3 * t2 + 1) * ys[i] + (t3 - 2 * t2 + t) * h[i] * m[i]
      + (-2 * t3 + 3 * t2) * ys[i + 1] + (t3 - t2) * h[i] * m[i + 1];
    return clamp(y, 0, 1);
  };
}

const CURVE_SIZE = 224;
const IDENTITY_CURVE = [[0, 0], [255, 255]];

// A point curve, Lightroom-style. Click the line to add a point and drag it;
// drag a point to move it between its neighbours; double-click a point (not an
// end) to remove it. The ends move too: that is how the black and white points
// are set. Every change goes to onChange as the whole list for that channel.
function CurveEditor({ settings, onChange }) {
  const [channel, setChannel] = useState("curve");
  const svgRef = useRef(null);
  const dragRef = useRef(null);
  const def = CURVE_CHANNELS.find((c) => c.key === channel);
  const points = settings[channel] || IDENTITY_CURVE;
  const sample = useMemo(() => curveSampler(points), [points]);
  const path = useMemo(() => {
    const steps = 96;
    let d = "";
    for (let i = 0; i <= steps; i++) {
      const x = i / steps;
      d += `${i ? "L" : "M"}${(x * CURVE_SIZE).toFixed(1)},${((1 - sample(x)) * CURVE_SIZE).toFixed(1)}`;
    }
    return d;
  }, [sample]);

  const valueAt = (e) => {
    const rect = svgRef.current.getBoundingClientRect();
    return [
      clamp(Math.round(((e.clientX - rect.left) / rect.width) * 255), 0, 255),
      clamp(Math.round((1 - (e.clientY - rect.top) / rect.height) * 255), 0, 255),
    ];
  };
  // A curve back on the diagonal is no curve, so it is stored as none.
  const commit = (next) => onChange(channel, curveEdited(next) ? next : null);

  const onPointerDown = (e) => {
    if (e.button !== 0) return;
    e.preventDefault();
    const [x, y] = valueAt(e);
    let base = points.map((p) => [...p]);
    // The nearest point within reach is taken; otherwise a new one is placed on
    // the line where it was clicked, so adding a point never moves the curve.
    const reach = 10;
    let index = base.findIndex(([px, py]) => Math.abs(px - x) <= reach && Math.abs(py - y) <= reach * 1.5);
    if (index < 0) {
      if (base.length >= MAX_CURVE_POINTS || base.some(([px]) => px === x)) return;
      base = [...base, [x, Math.round(sample(x / 255) * 255)]].sort((a, b) => a[0] - b[0]);
      index = base.findIndex(([px]) => px === x);
      commit(base);
    }
    dragRef.current = { index, base };
    svgRef.current.setPointerCapture(e.pointerId);
  };
  const onPointerMove = (e) => {
    const drag = dragRef.current;
    if (!drag) return;
    const [x, y] = valueAt(e);
    const next = drag.base.map((p) => [...p]);
    const lo = drag.index > 0 ? next[drag.index - 1][0] + 1 : 0;
    const hi = drag.index < next.length - 1 ? next[drag.index + 1][0] - 1 : 255;
    next[drag.index] = [clamp(x, lo, hi), y];
    drag.base = next;
    commit(next);
  };
  const onPointerUp = () => { dragRef.current = null; };
  const removePoint = (index) => {
    if (index === 0 || index === points.length - 1) return;
    commit(points.filter((_, i) => i !== index));
  };

  return (
    <div className="space-y-1.5">
      <div className="flex items-center gap-1">
        {CURVE_CHANNELS.map((c) => {
          const touched = curveEdited(settings[c.key]);
          return (
            <button key={c.key} onClick={() => setChannel(c.key)} title={`Curve for ${c.name}`}
              className={`text-[10px] px-1.5 py-0.5 rounded border ${channel === c.key
                ? "border-neutral-500 text-white" : "border-neutral-800 text-neutral-500 hover:text-neutral-300"}`}
              style={touched ? { color: c.colour } : undefined}>
              {c.label}{touched ? " •" : ""}
            </button>
          );
        })}
        {curveEdited(settings[channel]) && (
          <button onClick={() => onChange(channel, null)}
            className="ml-auto text-[10px] text-neutral-600 hover:text-neutral-300">reset {def.label}</button>
        )}
      </div>
      <svg ref={svgRef} viewBox={`0 0 ${CURVE_SIZE} ${CURVE_SIZE}`} data-testid="curve-editor"
        className="w-full aspect-square bg-neutral-900 rounded border border-neutral-800 touch-none cursor-crosshair select-none"
        onPointerDown={onPointerDown} onPointerMove={onPointerMove}
        onPointerUp={onPointerUp} onPointerCancel={onPointerUp}>
        {[0.25, 0.5, 0.75].map((f) => (
          <g key={f} stroke="#262626" strokeWidth="1">
            <line x1={f * CURVE_SIZE} y1="0" x2={f * CURVE_SIZE} y2={CURVE_SIZE} />
            <line x1="0" y1={f * CURVE_SIZE} x2={CURVE_SIZE} y2={f * CURVE_SIZE} />
          </g>
        ))}
        <line x1="0" y1={CURVE_SIZE} x2={CURVE_SIZE} y2="0" stroke="#404040" strokeDasharray="3 3" />
        <path d={path} fill="none" stroke={def.colour} strokeWidth="1.5" />
        {points.map(([x, y], i) => (
          <circle key={i} cx={(x / 255) * CURVE_SIZE} cy={(1 - y / 255) * CURVE_SIZE} r="4"
            fill="#0a0a0a" stroke={def.colour} strokeWidth="1.5"
            onDoubleClick={(e) => { e.stopPropagation(); removePoint(i); }}>
            <title>{`${x} → ${y}${i > 0 && i < points.length - 1 ? " · double-click to remove" : ""}`}</title>
          </circle>
        ))}
      </svg>
      <p className="text-[10px] text-neutral-600">
        Click the line to add a point, drag to shape. Double-click a point to remove it.
      </p>
    </div>
  );
}

// ── Black and white ──────────────────────────────────────────────────

// Which preset, if any, the current values are: the filter or tone whose every
// value matches, so its button can show as chosen.
function matching(presets, field, keys, settings) {
  return presets.find((p) => keys.every((k) => Number(settings[k]) === Number(p[field][k] || 0)))?.name;
}

function BwMix({ settings, onChange, onReset, onPreset }) {
  const filter = matching(BW_FILTERS, "mix", BW_MIX_KEYS, settings);
  const tone = matching(BW_TONES, "tone", SPLIT_KEYS, settings);
  const chip = (active) => `text-[11px] px-2 py-0.5 rounded border ${active
    ? "border-amber-400/70 text-amber-200" : "border-neutral-700 text-neutral-300 hover:text-white hover:border-neutral-500"}`;
  return (
    <div className="space-y-2">
      <div className="space-y-1">
        <div className="text-[10px] text-neutral-600">Filter</div>
        <div className="flex flex-wrap gap-1">
          {BW_FILTERS.map((f) => (
            <button key={f.name} title={f.note} className={chip(filter === f.name)}
              onClick={() => onPreset(BW_MIX_KEYS, f.mix, `${f.name} filter`)}>{f.name}</button>
          ))}
        </div>
      </div>
      <Controls>
      {HSL_BANDS.map((band) => {
        const key = `bw_${band.id}`;
        return (
          <Slider key={key} value={settings[key]} onChange={onChange} onReset={() => onReset(key)}
            def={{
              key, label: band.label, min: -100, max: 100, step: 1, format: signed,
              swatch: `hsl(${band.hue} 75% 55%)`,
              title: `How light the ${band.label.toLowerCase()}s turn in grey. A grey or nearly grey surface is not moved.`,
            }} />
        );
      })}
      </Controls>
      <div className="space-y-1">
        <div className="text-[10px] text-neutral-600">Toning</div>
        <div className="flex flex-wrap gap-1">
          {BW_TONES.map((t) => (
            <button key={t.name} title={t.note} className={chip(tone === t.name)}
              onClick={() => onPreset(SPLIT_KEYS, t.tone, t.name === "None" ? "Toning off" : `${t.name} toning`)}>{t.name}</button>
          ))}
        </div>
        <p className="text-[10px] text-neutral-600">Fine-tune the tone with Shadow and Light hue under Look.</p>
      </div>
    </div>
  );
}

// ── Styles ──────────────────────────────────────────────────────────
//
// The line under the style's name: which frames it will learn from, and why the
// rest are left out (backend export/engine.py style_preview).
function stylePreviewText(p) {
  if (!p) return "Counting this folder's frames…";
  if (p.failed) return "Could not count this folder's frames.";
  const frames = (n) => `${n} frame${n === 1 ? "" : "s"}`;
  if (!p.filtered) {
    return `Learns from ${Math.min(p.total, p.limit)} of ${frames(p.total)}${p.total > p.limit ? ", spread over all of them" : ""}: this frame's settings cannot be read, so none are left out.`;
  }
  const used = Math.min(p.matching, p.limit);
  const spread = p.matching > p.limit ? `, spread over the ${p.matching} that match` : "";
  const left = p.others.length
    ? ` Left out: ${p.others.map((o) => `${o.frames} with ${o.differs}`).join("; ")}.`
    : "";
  return `Learns from ${used} of ${frames(p.total)}${spread} — ${p.describe || "this frame's camera and settings"}.${left}`;
}

// A style is a look learned from many frames and kept under a name — a Ricoh's
// Positive Film, Canon's Neutral, the look of your own edited exports — that can
// be rendered on any RAW, from any camera (backend adjust/styles.py). Learning
// one reads the RAW frames of this folder shot with this frame's camera and
// settings, each against the JPEG beside it or inside it.
// Every learned style on this frame at once, as it is edited, to choose by eye:
// one small render per style through the same preview the develop view draws,
// two at a time. Taken when opened, and again when the frame changes; a slider
// moved meanwhile shows in the big preview, not here, until it is reopened.
const GALLERY_SIZE = 260;

function StyleGallery({ styles, value, onChange, filepath, edit }) {
  const [tiles, setTiles] = useState({});
  const editKey = JSON.stringify(edit);
  useEffect(() => {
    const base = { ...edit };
    delete base.style;
    delete base.style_fade;
    const ids = ["", ...styles.map((s) => s.id)];
    const ctl = new AbortController();
    const made = [];
    setTiles({});
    let next = 0;
    const worker = async () => {
      while (next < ids.length && !ctl.signal.aborted) {
        const id = ids[next++];
        try {
          const res = await fetch(`${API}/export/preview`, {
            method: "POST", headers: { "Content-Type": "application/json" }, signal: ctl.signal,
            body: JSON.stringify({ filepath, linear: true, size: GALLERY_SIZE,
              adjustment: id ? { ...base, style: id } : base }),
          });
          const url = res.ok ? URL.createObjectURL(await res.blob()) : "";
          if (url) made.push(url);
          setTiles((t) => ({ ...t, [id]: url || "failed" }));
        } catch {
          if (!ctl.signal.aborted) setTiles((t) => ({ ...t, [id]: "failed" }));
        }
      }
    };
    worker();
    worker();
    return () => { ctl.abort(); made.forEach((u) => URL.revokeObjectURL(u)); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filepath, styles, editKey]);

  const entries = [{ id: "", name: "Camera's own look" }, ...styles];
  return (
    <div className="grid grid-cols-2 gap-1.5 pt-1" data-testid="style-gallery">
      {entries.map((s) => {
        const tile = tiles[s.id];
        const on = s.id === value;
        return (
          <button key={s.id || "camera"} onClick={() => onChange(s.id)} title={s.source || s.name}
            className={`text-left rounded overflow-hidden border ${on ? "border-sky-400" : "border-neutral-800 hover:border-neutral-500"}`}>
            <div className="aspect-[3/2] bg-neutral-900 flex items-center justify-center">
              {tile && tile !== "failed"
                ? <img src={tile} alt="" className="w-full h-full object-contain" />
                : <span className="text-[10px] text-neutral-600">{tile === "failed" ? "could not render" : "…"}</span>}
            </div>
            <div className={`px-1 py-0.5 text-[10px] truncate ${on ? "text-sky-300" : "text-neutral-400"}`}>{s.name}</div>
          </button>
        );
      })}
    </div>
  );
}

// What an improvement from other folders came to, in a line.
function improveText(report) {
  if (!report) return "";
  if (report.added > 0) {
    const pct = report.before > 0 ? Math.round((1 - report.after / report.before) * 100) : 0;
    return `Improved: added ${report.added} frame${report.added === 1 ? "" : "s"} from ${report.folders} other folder${report.folders === 1 ? "" : "s"}`
      + (pct > 0 ? `; ${pct}% closer to the camera on its own frames held out.` : "; no worse on its own frames.");
  }
  const looked = report.looked_in ? ` (looked in ${report.looked_in} folder${report.looked_in === 1 ? "" : "s"} with matching frames)` : "";
  return `Kept as it was: ${report.reason || "nothing to add"}${looked}.`;
}

function StylePicker({ value, onChange, folder, filepath, edit, onRefitted }) {
  const [list, setList] = useState([]);
  const [gallery, setGallery] = useState(false);
  const [job, setJob] = useState(null);
  const [naming, setNaming] = useState(false);
  const [name, setName] = useState("");
  const [error, setError] = useState("");
  // What the last improvement from other folders came to.
  const [note, setNote] = useState("");
  // What the style would learn from, asked for while it is being named: null
  // while counting (the first look at a folder reads every RAW's settings).
  const [preview, setPreview] = useState(null);

  useEffect(() => {
    if (!naming || !folder) return undefined;
    const ctl = new AbortController();
    setPreview(null);
    const q = new URLSearchParams({ folder, like: filepath || "" });
    fetch(`${API}/styles/learn/preview?${q}`, { signal: ctl.signal })
      .then((r) => (r.ok ? r.json() : null))
      .then((p) => setPreview(p || { failed: true }))
      .catch(() => { if (!ctl.signal.aborted) setPreview({ failed: true }); });
    return () => ctl.abort();
  }, [naming, folder, filepath]);

  const load = useCallback(async () => {
    try {
      const data = await fetch(`${API}/styles`).then((r) => r.json());
      setList(data.styles || []);
      if (data.job?.status === "running") setJob(data.job);
    } catch { /* the picker just stays empty */ }
  }, []);
  useEffect(() => { load(); }, [load]);

  useEffect(() => {
    if (job?.status !== "running") return undefined;
    const timer = setInterval(async () => {
      try {
        const state = await fetch(`${API}/styles/learn`).then((r) => r.json());
        setJob(state);
        if (state.status === "done" && state.kind === "improve") {
          await load();
          setNote(improveText(state.report));
          if (state.style) onRefitted?.();
        } else if (state.status === "done" && state.style) {
          await load();
          onChange(state.style.id);
        } else if (state.status === "failed") {
          setError(state.error || "Could not learn a style here");
        }
      } catch { /* try again on the next tick */ }
    }, 1500);
    return () => clearInterval(timer);
  }, [job, load, onChange, onRefitted]);

  const learn = async () => {
    setError("");
    try {
      const res = await fetch(`${API}/styles/learn`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: name.trim(), folder, like: filepath }),
      });
      const data = await res.json();
      if (!res.ok) { setError(data.detail || "Could not start"); return; }
      setJob(data);
      setNaming(false);
      setName("");
    } catch { setError("Could not start"); }
  };

  // Deleting a style takes it away from every photo and every folder, so it is
  // never what the ✕ beside the menu does: that only takes it off this photo.
  const remove = async () => {
    const style = list.find((s) => s.id === value);
    if (!style) return;
    if (!window.confirm(`Delete the style “${style.name}” from this computer?\n\n`
      + "It disappears from every folder, and photos using it go back to their camera's own look.")) return;
    try { await fetch(`${API}/styles/${encodeURIComponent(value)}`, { method: "DELETE" }); } catch {}
    onChange("");
    load();
  };

  // Frames shot with the style's camera and settings in other folders, added
  // only if the style does no worse on its own frames held out (backend
  // adjust/styles.improve). The version before is kept for one undo.
  const improve = async () => {
    setError("");
    setNote("");
    try {
      const res = await fetch(`${API}/styles/${encodeURIComponent(value)}/improve`, { method: "POST" });
      const data = await res.json();
      if (!res.ok) { setError(data.detail || "Could not start"); return; }
      setJob(data);
    } catch { setError("Could not start"); }
  };
  const undo = async () => {
    setError("");
    try {
      const res = await fetch(`${API}/styles/${encodeURIComponent(value)}/restore`, { method: "POST" });
      if (!res.ok) { setError("No earlier version to go back to"); return; }
      setNote("Back to the version before the last improvement.");
      await load();
      onRefitted?.();
    } catch { setError("Could not undo"); }
  };

  const chosen = list.find((s) => s.id === value);
  const known = !!chosen;
  const running = job?.status === "running";
  const phase = { looking: "looking in folders", reading: "reading frames", checking: "checking it is better" };
  return (
    <div className="space-y-1">
      <div className="flex items-center gap-2 text-xs">
        <span className="w-20 shrink-0 text-neutral-400" title="Render this frame through a learned style instead of its own camera's look. The style brings its colour and tone; the frame keeps the brightness its camera chose.">Style</span>
        <select value={value} onChange={(e) => onChange(e.target.value)}
          className="flex-1 min-w-0 bg-neutral-900 border border-neutral-800 rounded px-1 py-0.5 text-xs">
          <option value="">Camera's own look</option>
          {list.map((s) => (
            <option key={s.id} value={s.id} title={s.source}>{s.name} ({s.frames})</option>
          ))}
          {value && !known && <option value={value}>{value} (not on this computer)</option>}
        </select>
        {value && (
          <button onClick={() => onChange("")} data-testid="style-clear"
            title="Take the style off this photo (back to the camera's own look). The style stays for other photos."
            className="text-neutral-600 hover:text-neutral-300">✕</button>
        )}
        {list.length > 0 && (
          <button onClick={() => setGallery((g) => !g)} data-testid="style-gallery-toggle"
            title="See this frame through every style at once, and click one to use it"
            className={`text-[10px] px-1 rounded border ${gallery ? "border-sky-500 text-sky-300" : "border-neutral-800 text-neutral-500 hover:text-neutral-300"}`}>
            ▦ Try each
          </button>
        )}
      </div>
      {gallery && list.length > 0 && (
        <StyleGallery styles={list} value={value} onChange={onChange} filepath={filepath} edit={edit} />
      )}
      <div className="text-[10px] text-neutral-600">
        {running ? (
          job.kind === "improve"
            ? <span>Improving “{job.name}”: {phase[job.phase] || "working"}{job.phase !== "checking" ? ` ${job.done}/${job.total || "?"}` : "…"}</span>
            : <span>Learning “{job.name}”… {job.done}/{job.total || "?"}</span>
        ) : naming ? (
          <span className="flex items-center gap-1">
            <input autoFocus value={name} onChange={(e) => setName(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && name.trim()) learn(); if (e.key === "Escape") setNaming(false); e.stopPropagation(); }}
              placeholder="Name, e.g. GR Positive Film"
              className="flex-1 min-w-0 bg-neutral-900 border border-neutral-800 rounded px-1 py-0.5 text-[11px] text-neutral-200" />
            <button disabled={!name.trim()} onClick={learn}
              className="px-1.5 rounded bg-neutral-800 text-neutral-200 disabled:opacity-40">learn</button>
          </span>
        ) : folder ? (
          <button onClick={() => setNaming(true)}
            title="Learn a style from this folder's RAW frames shot with this camera and its settings — each against the JPEG beside it (your edited export, or the camera's own) or inside it."
            className="underline decoration-neutral-700 hover:text-neutral-300">
            + New style from this folder
          </button>
        ) : null}
        {naming && !running && <span className="block mt-0.5 leading-snug" data-testid="style-preview">
          {stylePreviewText(preview)}
        </span>}
        {value && known && !running && !naming && (
          <button onClick={improve} disabled={!chosen.improvable} data-testid="style-improve"
            title={chosen.improvable
              ? "Look in other folders for frames shot with this style's camera and settings, and add them only if the style gets no worse on its own frames. Runs in the background."
              : "This style was learned before styles remembered their camera settings: learn it again to improve it from other folders."}
            className="ml-2 underline decoration-neutral-800 hover:text-neutral-300 disabled:opacity-40 disabled:no-underline">
            improve from other folders
          </button>
        )}
        {value && known && !running && !naming && chosen.has_previous && (
          <button onClick={undo} data-testid="style-undo"
            title="Go back to the version of this style before its last improvement"
            className="ml-2 underline decoration-neutral-800 hover:text-neutral-300">undo improvement</button>
        )}
        {value && known && !running && !naming && (
          <button onClick={remove} data-testid="style-delete"
            title="Delete this style from this computer, for every photo and folder"
            className="ml-2 underline decoration-neutral-800 hover:text-red-400">delete style…</button>
        )}
        {note && !running && <span className="block mt-0.5 leading-snug text-neutral-400" data-testid="style-note">{note}</span>}
        {error && <span className="block text-red-400/80">{error}</span>}
      </div>
    </div>
  );
}

// ── Colour mixer ─────────────────────────────────────────────────────

function ColourMixer({ settings, onChange, onReset }) {
  const [kind, setKind] = useState("sat");
  const active = HSL_KINDS.find((k) => k.id === kind);
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-1">
        {HSL_KINDS.map((k) => {
          const touched = HSL_BANDS.some((b) => Number(settings[`${k.id}_${b.id}`]) !== 0);
          return (
            <button key={k.id} onClick={() => setKind(k.id)} title={k.title}
              className={`text-[10px] px-1.5 py-0.5 rounded border ${kind === k.id
                ? "border-neutral-500 text-white" : "border-neutral-800 text-neutral-500 hover:text-neutral-300"}
                ${touched ? "text-amber-300" : ""}`}>
              {k.label}{touched ? " •" : ""}
            </button>
          );
        })}
      </div>
      <Controls>
      {HSL_BANDS.map((band) => {
        const key = `${kind}_${band.id}`;
        return (
          <Slider key={key} value={settings[key]} onChange={onChange} onReset={() => onReset(key)}
            def={{
              key, label: band.label, min: -100, max: 100, step: 1, format: signed,
              swatch: `hsl(${band.hue} 75% 55%)`,
              title: `${active.label} of the ${band.label.toLowerCase()}s. ${active.title}`,
            }} />
        );
      })}
      </Controls>
    </div>
  );
}

export default function DevelopView({ filepath, photos, onClose, onNavigate, onSaved }) {
  const [settings, setSettings] = useState(() => ZERO);
  const [controlStyle, setControlStyle] = useState(() => {
    try { return localStorage.getItem(CONTROLS_KEY) === "slider" ? "slider" : "bar"; } catch { return "bar"; }
  });
  const pickControlStyle = (id) => {
    setControlStyle(id);
    try { localStorage.setItem(CONTROLS_KEY, id); } catch { /* this visit only */ }
  };
  const [loading, setLoading] = useState(true);
  const [preview, setPreview] = useState(null);
  const [original, setOriginal] = useState(null);
  const [originalHist, setOriginalHist] = useState(null);
  const [editedHist, setEditedHist] = useState(null);
  const [editedChannels, setEditedChannels] = useState(null);
  // The pixel under the cursor: where it is in the photograph and its colour in
  // the edited picture. The picture's pixels are read once per render.
  const [probe, setProbe] = useState(null);
  const [readout, setReadoutState] = useState(lastReadout);
  const setReadout = useCallback((id) => { lastReadout = id; setReadoutState(id); }, []);
  const probePixels = useRef(null);
  const [photoSize, setPhotoSize] = useState(null);
  const [clip, setClip] = useState({ high: 0, low: 0 });
  const [clipUrl, setClipUrl] = useState(null);
  const [showClip, setShowClip] = useState(false);
  const [showOriginal, setShowOriginal] = useState(false);
  const [split, setSplit] = useState(0);          // 0 = off, else 0..100 percent
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [rendering, setRendering] = useState(false);
  const [hasLook, setHasLook] = useState(false);   // this frame has a camera look to dial
  // How established that look is (none | learning | own | ready), and from how
  // many frames: said once, quietly, under the Look heading.
  const [lookNote, setLookNote] = useState({ state: "", frames: 0, hint: "" });
  // Where the frame's noise reduction stands (off | working | ready | failed |
  // raw-only | unavailable), from the last preview drawn.
  const [noiseState, setNoiseState] = useState("off");
  // Learning the folder's frames, when the look asks for more (see LOOK_HINTS).
  const [learnJob, setLearnJob] = useState(null);
  const [lookRefresh, setLookRefresh] = useState(0);
  const [presets, setPresets] = useState([]);
  const [activePreset, setActivePreset] = useState("");
  const [presetName, setPresetName] = useState("");

  const clipboard = useCopiedEdit();
  const [copyPicker, setCopyPicker] = useState(false);
  const pasteMode = usePasteAs();
  const [pastePicker, setPastePicker] = useState(false);
  const [copyTicks, setCopyTicks] = useState(() => chosenSections().map((s) => s.id));
  const [cropMode, setCropMode] = useState(false);
  // Drawing gradients: the whole frame is shown, unturned, with their handles.
  const [maskMode, setMaskMode] = useState(false);
  const [maskSel, setMaskSel] = useState(-1);
  // Removing spots: like drawing masks, on the whole frame.
  const [spotMode, setSpotMode] = useState(false);
  const [spotSel, setSpotSel] = useState(-1);
  const [spotSize, setSpotSize] = useState(0.012);
  const [spotKind, setSpotKind] = useState("heal");
  const [spotBusy, setSpotBusy] = useState(false);
  const [brush, setBrushState] = useState(lastBrush);
  const [showShade, setShowShade] = useState(false);
  const brushRef = useRef(brush);
  brushRef.current = brush;
  const fullView = cropMode || maskMode || spotMode;
  const [aspect, setAspect] = useState("free");
  const [draft, setDraft] = useState(null);          // the box while it is being dragged
  const [frameDims, setFrameDims] = useState({ w: 3, h: 2 });
  const [previewFull, setPreviewFull] = useState(false); // the preview on screen is uncropped
  const [geoOriginal, setGeoOriginal] = useState({ key: "", url: null });
  const [version, setVersion] = useState(0);         // 0 is the photo's own edit, 1.. its copies
  const [copies, setCopies] = useState([]);          // virtual copies of this frame
  const [loupe, setLoupe] = useState(() => lastLoupe);          // {u, v} on the picture, or null
  const [loupeUrl, setLoupeUrl] = useState(null);
  const [loupeZoom, setLoupeZoomState] = useState(lastLoupeZoom);
  const setLoupeZoom = useCallback((z) => { lastLoupeZoom = z; setLoupeZoomState(z); }, []);
  const [loupeState, setLoupeState] = useState("");  // "", "loading", "failed"
  const imgRef = useRef(null);
  const geoOriginalUrl = useRef(null);
  const loupeObjectUrl = useRef(null);
  const previewUrl = useRef(null);
  const originalUrl = useRef(null);
  const saveTimer = useRef(null);
  const previewTimer = useRef(null);
  const pendingSave = useRef(null);          // the edit waiting on the debounce
  const lastBitmap = useRef(null);           // the picture on screen, for the clipping map
  // When this view opened. The keypress that opened it is still travelling up
  // the page when this view mounts and registers its own key handler, and that
  // handler would receive the very key that opened it.
  const openedAt = useRef(typeof performance !== "undefined" ? performance.now() : 0);
  const showClipRef = useRef(false);
  const settingsRef = useRef(settings);
  settingsRef.current = settings;
  showClipRef.current = showClip;

  const position = useMemo(
    () => (photos || []).findIndex((p) => p.filepath === filepath),
    [photos, filepath],
  );

  const render = useCallback(
    async (target, body) => {
      const res = await fetch(`${API}/export/preview`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          filepath: target,
          // The same pipeline with and without an edit, so the original and an
          // edit sitting at zero are one picture and nothing jumps when the
          // first slider moves.
          linear: true,
          size: PREVIEW_SIZE,
          ...body,
        }),
      });
      if (!res.ok) return null;
      setHasLook(res.headers.get("X-Camera-Look") === "1");
      setLookNote({
        state: res.headers.get("X-Camera-Look-State") || "none",
        frames: Number(res.headers.get("X-Camera-Look-Frames")) || 0,
        hint: res.headers.get("X-Camera-Look-Hint") || "",
      });
      const noise = res.headers.get("X-Noise-Reduction") || "off";
      const [pw, ph] = (res.headers.get("X-Photo-Size") || "").split("x").map(Number);
      const blob = await res.blob();
      const bitmap = await createImageBitmap(blob);
      return { url: URL.createObjectURL(blob), bitmap, noise, photoSize: pw && ph ? { w: pw, h: ph } : null };
    },
    [],
  );

  const learnFolder = useCallback(async (folder) => {
    try {
      const res = await fetch(`${API}/camera-look/learn`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ folder }),
      });
      if (res.ok) setLearnJob({ folder, ...(await res.json()) });
    } catch { /* the hint simply stays */ }
  }, []);

  // While a folder is being learned, ask how far it has got; when it is done the
  // frame on screen is drawn again from the finished pool.
  useEffect(() => {
    if (learnJob?.status !== "running") return undefined;
    const timer = setInterval(async () => {
      try {
        const res = await fetch(`${API}/camera-look/learn?folder=${encodeURIComponent(learnJob.folder)}`);
        if (!res.ok) return;
        const state = await res.json();
        setLearnJob({ folder: learnJob.folder, ...state });
        if (state.status !== "running") setLookRefresh((n) => n + 1);
      } catch { /* try again on the next tick */ }
    }, 2000);
    return () => clearInterval(timer);
  }, [learnJob]);

  // The frame itself: the unedited rendering is fetched once per photo and kept,
  // because every comparison needs it and it never changes while the frame is
  // open.
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setSettings(ZERO);
    // Nothing to undo until the frame's edit is read (its kept history is not
    // touched: it resumes then).
    historyRef.current = { entries: [], index: -1 };
    setHistory(historyRef.current);
    setActivePreset("");
    setNote("");
    setEditedHist(null);
    setClipUrl(null);
    setShowOriginal(false);
    setCropMode(false);
    setAspect("free");
    setDraft(null);

    (async () => {
      try {
        fetch(`${API}/photos/versions?filepath=${encodeURIComponent(filepath)}`)
          .then((r) => (r.ok ? r.json() : { copies: [] }))
          .then((d) => { if (!cancelled) setCopies(d.copies || []); })
          .catch(() => {});
        const [stored, baseline] = await Promise.all([
          fetch(`${API}/photos/edit?filepath=${encodeURIComponent(filepath)}`)
            .then((r) => (r.ok ? r.json() : { settings: {} }))
            .then((d) => d.settings || {})
            .catch(() => ({})),
          render(filepath, { adjustment: {} }),
        ]);
        if (cancelled) return;
        const opened = asSettings(stored);
        setSettings(opened);
        settingsRef.current = opened;
        startHistory(opened);
        if (baseline) {
          if (originalUrl.current) URL.revokeObjectURL(originalUrl.current);
          originalUrl.current = baseline.url;
          setOriginal(baseline.url);
          setFrameDims({ w: baseline.bitmap.width, h: baseline.bitmap.height });
          setOriginalHist(histogramOf(baseline.bitmap).bins);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
      // With this one on screen, the server gets the photographs either side of
      // it ready, and this one at full size for the loupe, so stepping or
      // zooming does not wait for a decode.
      if (cancelled) return;
      const at = (photos || []).findIndex((p) => p.filepath === filepath);
      const near = at < 0 ? [] : [photos[at + 1], photos[at - 1]].filter(Boolean).map((p) => p.filepath);
      fetch(`${API}/export/develop/prefetch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filepaths: near, size: PREVIEW_SIZE, full: filepath }),
      }).catch(() => {});
    })();

    return () => { cancelled = true; };
  }, [filepath, render]);  // eslint-disable-line react-hooks/exhaustive-deps

  // The unedited frame is kept once per photo; learning the folder changes the look it
  // starts from, so it is drawn again then.
  useEffect(() => {
    if (!lookRefresh) return undefined;
    let cancelled = false;
    (async () => {
      const baseline = await render(filepath, { adjustment: {} });
      if (cancelled || !baseline) return;
      if (originalUrl.current) URL.revokeObjectURL(originalUrl.current);
      originalUrl.current = baseline.url;
      setOriginal(baseline.url);
      setOriginalHist(histogramOf(baseline.bitmap).bins);
    })();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lookRefresh]);

  // The edit as it would export. One request at a time: while a render is on
  // its way, each change only replaces the one waiting to go next, and when the
  // render lands the newest waiting edit is sent at once. A drag then updates as
  // fast as the server can draw, and the server never has a queue of renders
  // nobody will see — which is what made a curve drag lag before, since a drag
  // pauses often enough to start a request at every pause.
  const previewBusy = useRef(false);
  const previewNext = useRef(null);
  const previewAsked = useRef(null);         // the last edit sent to be drawn
  // Whether a pointer is held down: a slider, the curve, a mask's handle, the brush.
  const held = useRef(false);
  const mounted = useRef(true);
  // Set on every mount, not only the first: React's StrictMode mounts, unmounts
  // and mounts again in development, and a flag cleared by that first unmount
  // would stop previews for good.
  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);

  // While the noise reduction is being made the preview is drawn without it, and
  // asked for again every so often until it is there (for a minute at most).
  const noisePoll = useRef({ timer: null, tries: 0 });
  useEffect(() => {
    setNoiseState("off");
    return () => {
      clearTimeout(noisePoll.current.timer);
      noisePoll.current.tries = 0;
    };
  }, [filepath]);

  const drawPreviews = useCallback(async () => {
    if (previewBusy.current) return;
    previewBusy.current = true;
    setRendering(true);
    let waiting = null;
    try {
      while (previewNext.current && mounted.current) {
        const want = previewNext.current;
        previewNext.current = null;
        previewAsked.current = want;
        const result = await render(filepath, want.body);
        if (!result || !mounted.current) continue;
        waiting = result.noise === "working" ? want : null;
        setNoiseState(result.noise);
        if (previewUrl.current) URL.revokeObjectURL(previewUrl.current);
        previewUrl.current = result.url;
        setPreview(result.url);
        setPreviewFull(want.full);
        const stats = histogramOf(result.bitmap);
        setEditedHist(stats.bins);
        setEditedChannels(stats.rgb);
        setClip({ high: stats.clipHigh, low: stats.clipLow });
        setClipUrl(showClipRef.current ? clipOverlayOf(result.bitmap) : null);
        lastBitmap.current = result.bitmap;
        probePixels.current = null;
        if (result.photoSize) setPhotoSize(result.photoSize);
      }
    } finally {
      previewBusy.current = false;
      if (mounted.current) setRendering(false);
    }
    clearTimeout(noisePoll.current.timer);
    if (waiting && mounted.current && noisePoll.current.tries < 60) {
      noisePoll.current.tries += 1;
      noisePoll.current.timer = setTimeout(() => {
        if (!previewNext.current) previewNext.current = waiting;
        drawPreviews();
      }, 1000);
    } else if (!waiting) {
      noisePoll.current.tries = 0;
    }
  }, [filepath, render]);

  useEffect(() => {
    if (loading) return undefined;
    // While the crop is being drawn the whole frame is shown, so the box has
    // something to be drawn on; the framing is applied the moment it is done.
    previewNext.current = {
      full: fullView,
      body: {
        adjustment: isEdited(settings) ? activeOnly(settings, !fullView) : {},
        // While the hand is still moving, a draft (a quarter of the pixels, so
        // about a quarter of the time) keeps the picture up with it.
        draft: held.current,
      },
    };
    // A breath, so a burst of changes from one gesture leaves as one request.
    if (previewTimer.current) clearTimeout(previewTimer.current);
    previewTimer.current = setTimeout(drawPreviews, 40);
    return undefined;
  }, [settings, loading, fullView, lookRefresh, drawPreviews]);

  // Letting go draws the last draft again at full size.
  useEffect(() => {
    const down = () => { held.current = true; };
    const up = () => {
      held.current = false;
      const want = previewNext.current || previewAsked.current;
      if (!want?.body.draft) return;
      previewNext.current = { ...want, body: { ...want.body, draft: false } };
      if (previewTimer.current) clearTimeout(previewTimer.current);
      drawPreviews();
    };
    window.addEventListener("pointerdown", down, true);
    window.addEventListener("pointerup", up, true);
    window.addEventListener("pointercancel", up, true);
    return () => {
      window.removeEventListener("pointerdown", down, true);
      window.removeEventListener("pointerup", up, true);
      window.removeEventListener("pointercancel", up, true);
    };
  }, [drawPreviews]);

  // The "original" shown beside a cropped edit is cropped the same way, so the
  // split and the backslash key compare like with like instead of two different shapes.
  const framingKey = JSON.stringify(framingOnly(settings));
  useEffect(() => {
    if (loading || cropMode || !hasFraming(settings)) return undefined;
    if (geoOriginal.key === framingKey) return undefined;
    let cancelled = false;
    const timer = setTimeout(async () => {
      const result = await render(filepath, { adjustment: framingOnly(settings) });
      if (cancelled || !result) return;
      if (geoOriginalUrl.current) URL.revokeObjectURL(geoOriginalUrl.current);
      geoOriginalUrl.current = result.url;
      setGeoOriginal({ key: framingKey, url: result.url });
    }, 250);
    return () => { cancelled = true; clearTimeout(timer); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [framingKey, cropMode, loading, filepath, render]);

  // The overlay is built on demand from the picture on screen, so turning the
  // warning on does not wait for the next slider movement.
  useEffect(() => {
    if (!showClip) { setClipUrl(null); return; }
    if (lastBitmap.current) setClipUrl(clipOverlayOf(lastBitmap.current));
  }, [showClip]);

  // The loupe: a 1:1 patch of the edited frame around the clicked point. It is
  // redrawn when the edit changes — the point of it is to watch a face or a
  // highlight while the sliders move — on a slightly longer debounce than the
  // preview because it reads the full-resolution frame.
  useEffect(() => {
    if (!loupe || loading || cropMode) return undefined;
    let cancelled = false;
    setLoupeState("loading");
    const timer = setTimeout(async () => {
      try {
        const res = await fetch(`${API}/export/develop/loupe`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            filepath, u: loupe.u, v: loupe.v, size: Math.round(LOUPE_SIDE / loupeZoom), preview_size: PREVIEW_SIZE,
            adjustment: showOriginal ? framingOnly(settings) : activeOnly(settings),
          }),
        });
        if (!res.ok) throw new Error("loupe");
        const blob = await res.blob();
        if (cancelled) return;
        const url = URL.createObjectURL(blob);
        if (loupeObjectUrl.current) URL.revokeObjectURL(loupeObjectUrl.current);
        loupeObjectUrl.current = url;
        setLoupeUrl(url);
        setLoupeState("");
      } catch {
        if (!cancelled) setLoupeState("failed");
      }
    }, 250);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [loupe, settings, showOriginal, cropMode, loading, filepath, loupeZoom]);

  const placeLoupe = useCallback((point) => {
    lastLoupe = point;
    setLoupe(point);
    if (!point) setLoupeUrl(null);
  }, []);

  // Where the cursor is over the picture, and the colour there.
  const onPictureMove = useCallback((e) => {
    const bitmap = lastBitmap.current;
    if (!imgRef.current || !bitmap) return;
    const rect = imgRef.current.getBoundingClientRect();
    const u = (e.clientX - rect.left) / rect.width;
    const v = (e.clientY - rect.top) / rect.height;
    if (u < 0 || u >= 1 || v < 0 || v >= 1) { setProbe(null); return; }
    if (!probePixels.current) {
      const canvas = document.createElement("canvas");
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      ctx.drawImage(bitmap, 0, 0);
      probePixels.current = { w: bitmap.width, h: bitmap.height, data: ctx.getImageData(0, 0, bitmap.width, bitmap.height).data };
    }
    const { w, h, data } = probePixels.current;
    const i = (Math.min(h - 1, Math.floor(v * h)) * w + Math.min(w - 1, Math.floor(u * w))) * 4;
    setProbe({ u, v, rgb: [data[i], data[i + 1], data[i + 2]] });
  }, []);

  const onPictureClick = useCallback((e) => {
    if (fullView || !imgRef.current) return;
    const rect = imgRef.current.getBoundingClientRect();
    const u = (e.clientX - rect.left) / rect.width;
    const v = (e.clientY - rect.top) / rect.height;
    if (u < 0 || u > 1 || v < 0 || v > 1) return;
    placeLoupe({ u, v });
  }, [fullView, placeLoupe]);

  const persist = useCallback(
    async (next, keepalive = false) => {
      try {
        const res = await fetch(`${API}/photos/edit`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ filepath, version, ...activeOnly(next) }),
          keepalive,
        });
        if (!res.ok) { if (!keepalive) setNote("Could not save this edit"); return; }
        const data = await res.json();
        // The library shows the photo's own edit; a copy is not what the grid draws.
        if (version === 0) onSaved?.(filepath, data.settings || {});
      } catch {
        if (!keepalive) setNote("Could not save this edit");
      }
    },
    [filepath, onSaved, version],
  );
  const persistRef = useRef(persist);
  persistRef.current = persist;

  // Saved on a short debounce rather than on every step of a drag: the write
  // goes to the catalog and to the sidecar, and a drag is a hundred steps.
  const scheduleSave = useCallback((next) => {
    pendingSave.current = next;
    if (saveTimer.current) clearTimeout(saveTimer.current);
    saveTimer.current = setTimeout(() => {
      saveTimer.current = null;
      pendingSave.current = null;
      persistRef.current(next);
    }, 500);
  }, []);

  // ── History ─────────────────────────────────────────────────────
  const [history, setHistory] = useState({ entries: [], index: -1 });
  const historyRef = useRef(history);
  const historyKey = `${filepath}#${version}`;
  const historyKeyRef = useRef(historyKey);
  historyKeyRef.current = historyKey;
  const putHistory = useCallback((h) => {
    historyRef.current = h;
    HISTORIES.set(historyKeyRef.current, h);
    setHistory(h);
  }, []);
  // A frame or copy as it is read: its kept history when that ends where the
  // edit now is, a new one starting here when not (it was changed elsewhere).
  const startHistory = useCallback((now) => {
    const kept = HISTORIES.get(historyKeyRef.current);
    if (kept && kept.index >= 0 && canonical(kept.entries[kept.index].settings) === canonical(now)) {
      putHistory(kept);
      return;
    }
    putHistory({ entries: [{ settings: now, label: "As opened", group: null, at: 0 }], index: 0 });
  }, [putHistory]);
  const record = useCallback((next, step) => {
    const h = historyRef.current;
    if (h.index < 0) return;
    const top = h.entries[h.index];
    if (JSON.stringify(top.settings) === JSON.stringify(next)) return;
    const at = Date.now();
    if (step.group && top.group === step.group && at - top.at < COALESCE_MS && h.index > 0) {
      const entries = [...h.entries.slice(0, h.index), { ...step, settings: next, at }];
      putHistory({ entries, index: h.index });
      return;
    }
    const entries = [...h.entries.slice(0, h.index + 1), { ...step, settings: next, at }].slice(-MAX_HISTORY);
    putHistory({ entries, index: entries.length - 1 });
  }, [putHistory]);

  const change = useCallback((key, value) => {
    setActivePreset("");
    const prev = settingsRef.current;
    const next = { ...prev, [key]: value };
    settingsRef.current = next;
    setSettings(next);
    scheduleSave(next);
    record(next, stepOf(key, value, prev));
  }, [scheduleSave, record]);

  const resetOne = useCallback((key) => change(key, ZERO[key]), [change]);

  const applySettings = useCallback((next, noteText = "", recorded = true) => {
    if (recorded) record(next, { label: noteText || diffLabel(settingsRef.current, next), group: null });
    setSettings(next);
    settingsRef.current = next;
    if (saveTimer.current) clearTimeout(saveTimer.current);
    saveTimer.current = null;
    pendingSave.current = null;
    persist(next);
    setNote(noteText);
  }, [persist, record]);

  // Going back or forward is not itself a step: it moves along the list, and
  // the next change from there drops what was ahead.
  const jumpTo = useCallback((index) => {
    const h = historyRef.current;
    if (index < 0 || index >= h.entries.length || index === h.index) return;
    const target = h.entries[index];
    const undoing = index < h.index;
    putHistory({ ...h, index, entries: h.entries.map((e) => ({ ...e, group: null })) });
    setActivePreset("");
    applySettings(target.settings,
      undoing ? `Undone: ${h.entries[index + 1].label}` : `Redone: ${target.label}`, false);
  }, [applySettings, putHistory]);
  const undo = useCallback(() => jumpTo(historyRef.current.index - 1), [jumpTo]);
  const redo = useCallback(() => jumpTo(historyRef.current.index + 1), [jumpTo]);

  // Reset undoes the tone and leaves the framing: a crop is a decision about the
  // picture, not a slider, and has its own button.
  const reset = useCallback(() => {
    setActivePreset("");
    applySettings(withFramingOf(ZERO, settingsRef.current), "Reset");
  }, [applySettings]);

  // ── Crop and level ──────────────────────────────────────────────
  const currentBox = useCallback(
    () => draft || boxOf(settingsRef.current) || FULL_BOX,
    [draft],
  );

  const commitFraming = useCallback((patch, noteText = "", debounced = false) => {
    const next = { ...settingsRef.current, ...patch };
    // A box that is the whole frame is no crop.
    const box = boxOf(next);
    if (box && box.x < 1e-4 && box.y < 1e-4 && box.w > 0.9999 && box.h > 0.9999) {
      Object.assign(next, { crop_x: null, crop_y: null, crop_w: null, crop_h: null });
    }
    setActivePreset("");
    if (debounced) {
      // A drag of the level slider is a hundred steps; the write is one.
      record(next, { label: diffLabel(settingsRef.current, next), group: "framing" });
      setSettings(next);
      settingsRef.current = next;
      scheduleSave(next);
      return;
    }
    applySettings(next, noteText);
  }, [applySettings, scheduleSave, record]);

  const commitBox = useCallback((box) => {
    commitFraming({ crop_x: box.x, crop_y: box.y, crop_w: box.w, crop_h: box.h });
  }, [commitFraming]);

  // A loupe point is a spot on the picture as it is framed now, so reframing it
  // would leave the marker on a different part of the scene.
  const enterCrop = useCallback(() => {
    lastLoupe = null;
    setLoupe(null);
    setLoupeUrl(null);
    setCropMode(true);
  }, []);
  const startCrop = useCallback(() => { setMaskMode(false); setSpotMode(false); enterCrop(); }, [enterCrop]);
  const endCrop = useCallback(() => { setCropMode(false); setDraft(null); }, []);

  // ── Gradients ───────────────────────────────────────────────────
  const enterMasks = useCallback(() => {
    setSpotMode(false);
    lastLoupe = null;
    setLoupe(null);
    setLoupeUrl(null);
    setCropMode(false);
    setDraft(null);
    setMaskMode(true);
  }, []);
  const endMasks = useCallback(() => setMaskMode(false), []);

  // ── Spots ───────────────────────────────────────────────────────
  const toggleSpots = useCallback(() => {
    setSpotMode((on) => {
      if (!on) {
        lastLoupe = null;
        setLoupe(null);
        setLoupeUrl(null);
        setCropMode(false);
        setDraft(null);
        setMaskMode(false);
      }
      return !on;
    });
  }, []);
  const setSpots = useCallback((spots) => change("spots", spots), [change]);
  const addSpot = useCallback(async (x, y) => {
    const r = spotSize;
    const current = settingsRef.current.spots || [];
    setSpotBusy(true);
    let source = null;
    try {
      const res = await fetch(`${API}/export/develop/spot-source`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filepath, x, y, r, spots: current, size: PREVIEW_SIZE }),
      });
      if (res.ok) source = await res.json();
    } catch { /* placed beside it instead */ }
    setSpotBusy(false);
    const sx = source?.sx ?? (x + 3 * r <= 1 ? x + 3 * r : x - 3 * r);
    const sy = source?.sy ?? y;
    const spots = [...(settingsRef.current.spots || []), { x, y, r, sx, sy, mode: spotKind }];
    setSpots(spots);
    setSpotSel(spots.length - 1);
  }, [spotSize, spotKind, filepath, setSpots]);
  const updateSpot = useCallback((index, spot) => {
    setSpots((settingsRef.current.spots || []).map((sp, i) => (i === index ? spot : sp)));
  }, [setSpots]);
  const removeSpot = useCallback((index) => {
    setSpots((settingsRef.current.spots || []).filter((_, i) => i !== index));
    setSpotSel(-1);
  }, [setSpots]);
  const clearSpots = useCallback(() => { setSpots([]); setSpotSel(-1); }, [setSpots]);
  const setMasks = useCallback((masks) => change("masks", masks), [change]);
  const setBrush = useCallback((patch) => {
    setBrushState((b) => (lastBrush = { ...b, ...patch }));
  }, []);
  const [choosing, setChoosing] = useState({ busy: null, error: "" });
  const addMask = useCallback(async (kind) => {
    if (CHOSEN_NAMES[kind]) {
      // The model is fetched the first time (a few MB); the mask is added once
      // it can select anything.
      setChoosing({ busy: kind, error: "" });
      try {
        const res = await fetch(`${API}/export/develop/mask-model`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ kind }),
        });
        if (!res.ok) {
          const body = await res.json().catch(() => ({}));
          setChoosing({ busy: null, error: body.detail || `Could not get the ${kind} model` });
          return;
        }
      } catch {
        setChoosing({ busy: null, error: "The server did not answer" });
        return;
      }
      setChoosing({ busy: null, error: "" });
    }
    const masks = [...(settingsRef.current.masks || []), newMask(kind, frameDims)];
    setMasks(masks);
    setMaskSel(masks.length - 1);
    // A chosen mask is shown shaded: what the model chose is the first question.
    if (CHOSEN_NAMES[kind]) setShowShade(true);
    // A brush starts with nothing painted, so it starts with the brush in hand.
    setBrush({ tool: kind === "brush" ? "paint" : null });
    enterMasks();
  }, [setMasks, frameDims, enterMasks, setBrush]);
  const updateMask = useCallback((index, mask) => {
    setMasks((settingsRef.current.masks || []).map((m, i) => (i === index ? mask : m)));
  }, [setMasks]);
  const removeMask = useCallback((index) => {
    const masks = (settingsRef.current.masks || []).filter((_, i) => i !== index);
    setMasks(masks);
    setMaskSel((sel) => (sel === index ? -1 : sel > index ? sel - 1 : sel));
    if (!masks.length) setMaskMode(false);
  }, [setMasks]);
  const selectMask = useCallback((index) => {
    setMaskSel(index);
    const m = settingsRef.current.masks?.[index];
    setBrush({ tool: m?.kind === "brush" ? "paint" : null });
    enterMasks();
  }, [enterMasks, setBrush]);
  // What the subject and sky selections are on this frame, as pictures for the
  // shading; asked for once per frame, when a mask of that kind is drawn on.
  const [mattes, setMattes] = useState({});
  const [people, setPeople] = useState(null);
  // Picking a colour for the selected mask's range: the next click on the picture.
  const [picking, setPicking] = useState(false);
  const [rangeImage, setRangeImage] = useState(null);
  const mattesFor = useRef({ path: null, asked: new Set() });
  useEffect(() => {
    const lensBody = frameGeometry(settings);
    const framed = `${filepath}|${JSON.stringify(lensBody)}`;
    if (mattesFor.current.path !== framed) {
      Object.values(mattesFor.current.urls || {}).forEach((u) => URL.revokeObjectURL(u));
      mattesFor.current = { path: framed, asked: new Set(), urls: {} };
      setMattes({});
      setPeople(null);
    }
    if (!maskMode) return;
    const chosen = (settings.masks || []).filter((m) => CHOSEN_NAMES[m.kind]);
    const keys = [...new Map(chosen.map((m) => [matteKey(m), m])).entries()];
    const state = mattesFor.current;
    keys.filter(([key]) => !state.asked.has(key)).forEach(async ([key, m]) => {
      state.asked.add(key);
      try {
        const res = await fetch(`${API}/export/develop/mask-image`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ filepath, kind: m.kind, person: m.person || 0, size: PREVIEW_SIZE, ...lensBody }),
        });
        if (!res.ok || mattesFor.current !== state) return;
        const url = URL.createObjectURL(await res.blob());
        state.urls[key] = url;
        setMattes((had) => ({ ...had, [key]: url }));
      } catch {
        state.asked.delete(key);
      }
    });
    // The faces a people or faces mask can be narrowed to, asked for once.
    if (chosen.some((m) => OF_PEOPLE.includes(m.kind)) && !state.asked.has("_people")) {
      state.asked.add("_people");
      fetch(`${API}/export/develop/people`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filepath, size: PREVIEW_SIZE, ...lensBody }),
      }).then((r) => (r.ok ? r.json() : null)).then((body) => {
        if (body && mattesFor.current === state) setPeople(body.faces);
      }).catch(() => state.asked.delete("_people"));
    }
  }, [filepath, maskMode, settings.masks, settings.lens_profile, settings.distortion,
    settings.tf_vertical, settings.tf_horizontal, settings.tf_aspect]);
  useEffect(() => () => {
    Object.values(mattesFor.current.urls || {}).forEach((u) => URL.revokeObjectURL(u));
  }, []);

  // The selected mask's range, as a picture the shading is multiplied by: asked
  // for a moment after its sliders stop.
  const selectedRange = maskMode ? settings.masks?.[maskSel]?.range : null;
  const rangeKey = selectedRange
    ? JSON.stringify([filepath, selectedRange, frameGeometry(settings)]) : "";
  useEffect(() => {
    if (!rangeKey) { setRangeImage(null); return undefined; }
    let stop = false;
    let url = null;
    const timer = setTimeout(async () => {
      try {
        const res = await fetch(`${API}/export/develop/range-image`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ filepath, range: selectedRange, size: PREVIEW_SIZE, ...frameGeometry(settings) }),
        });
        if (!res.ok || stop) return;
        url = URL.createObjectURL(await res.blob());
        if (!stop) setRangeImage(url);
      } catch { /* the shading is drawn without it */ }
    }, 250);
    return () => { stop = true; clearTimeout(timer); if (url) URL.revokeObjectURL(url); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rangeKey]);

  const pickColour = useCallback(async (x, y) => {
    setPicking(false);
    const m = settingsRef.current.masks?.[maskSel];
    if (!m) return;
    try {
      const res = await fetch(`${API}/export/develop/sample`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filepath, x, y, size: PREVIEW_SIZE, ...frameGeometry(settingsRef.current) }),
      });
      if (!res.ok) return;
      const c = await res.json();
      const width = m.range?.kind === "color" ? m.range.width : 40;
      updateMask(maskSel, { ...m, range: { kind: "color", cb: c.cb, cr: c.cr, lum: Math.round(c.lum), width } });
      setShowShade(true);
    } catch { /* nothing picked */ }
  }, [filepath, maskSel, updateMask]);

  const addStroke = useCallback((index, stroke) => {
    const m = settingsRef.current.masks?.[index];
    if (m) updateMask(index, { ...m, strokes: [...(m.strokes || []), stroke] });
  }, [updateMask]);

  const chooseAspect = useCallback((id) => {
    const def = ASPECTS.find((a) => a.id === id);
    setAspect(id);
    enterCrop();
    if (!def || def.ratio === null) return;
    const base = boxOf(settingsRef.current) || FULL_BOX;
    const angle = Number(settingsRef.current.angle) || 0;
    const box = fitBox(boxOfRatio(def.ratio, base, frameDims.w, frameDims.h), angle, frameDims.w, frameDims.h);
    commitBox(box);
  }, [commitBox, frameDims, enterCrop]);

  const setAngle = useCallback((value) => {
    const angle = Math.max(-45, Math.min(45, Number(value) || 0));
    enterCrop();
    const box = boxOf(settingsRef.current) || FULL_BOX;
    const fitted = fitBox(box, angle, frameDims.w, frameDims.h);
    commitFraming({
      angle,
      crop_x: fitted.x, crop_y: fitted.y, crop_w: fitted.w, crop_h: fitted.h,
    }, "", true);
  }, [commitFraming, frameDims, enterCrop]);

  const resetFraming = useCallback(() => {
    setAspect("free");
    applySettings({
      ...settingsRef.current,
      angle: 0, crop_x: null, crop_y: null, crop_w: null, crop_h: null,
    }, "Crop and level removed");
  }, [applySettings]);

  // A rescue or a look sets its own group and leaves the rest as it was.
  const applyGroup = useCallback((owned, values, name) => {
    setActivePreset("");
    const next = { ...settingsRef.current };
    for (const key of owned) next[key] = ZERO[key];
    for (const [key, value] of Object.entries(values)) next[key] = value;
    applySettings(next, name);
  }, [applySettings]);

  // Upright: the transform and level that straighten the photo's own lines, put
  // on the sliders (backend adjust/upright.py). It says why when it changes nothing.
  const [upright, setUpright] = useState({ busy: null, note: "" });
  useEffect(() => { setUpright({ busy: null, note: "" }); }, [filepath]);
  const runUpright = useCallback(async (mode) => {
    setUpright({ busy: mode, note: "" });
    try {
      const res = await fetch(`${API}/export/develop/upright`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filepath, mode, size: PREVIEW_SIZE,
          lens_profile: Number(settingsRef.current.lens_profile) || 0,
          distortion: Number(settingsRef.current.distortion) || 0 }),
      });
      const found = res.ok ? await res.json() : null;
      if (!found) { setUpright({ busy: null, note: "Could not read this photo" }); return; }
      if (!found.found) { setUpright({ busy: null, note: `Left as it is: ${found.reason}.` }); return; }
      if (mode === "vertical") {
        applyGroup(["tf_vertical", "angle"], { tf_vertical: found.tf_vertical, angle: found.angle }, "Upright: verticals");
      } else {
        applyGroup(["angle"], { angle: found.angle }, "Upright: level");
      }
      setUpright({ busy: null, note: `From ${found.lines} lines.` });
    } catch {
      setUpright({ busy: null, note: "The server did not answer" });
    }
  }, [filepath, applyGroup]);


  // ── Match a look ────────────────────────────────────────────────
  const [reference, setReferenceState] = useState(lookReference);
  const [matched, setMatched] = useState(null);       // { found, amount, note, name } on this frame
  const [matchBusy, setMatchBusy] = useState(false);
  const [matchError, setMatchError] = useState("");
  useEffect(() => { setMatched(null); setMatchError(""); }, [filepath, version]);
  const setReference = useCallback((ref) => { lookReference = ref; setReferenceState(ref); }, []);
  const pickThisAsReference = useCallback(() => {
    setReference({ filepath, name: filepath.split("/").pop() });
    setMatchError("");
  }, [filepath, setReference]);
  const referenceFromFile = useCallback((file) => {
    const reader = new FileReader();
    reader.onload = () => { setReference({ image: reader.result, name: file.name }); setMatchError(""); };
    reader.onerror = () => setMatchError("Could not read that file");
    reader.readAsDataURL(file);
  }, [setReference]);
  const runMatch = useCallback(async () => {
    if (!reference) return;
    setMatchBusy(true);
    setMatchError("");
    try {
      const res = await fetch(`${API}/export/develop/match-look`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filepath, adjustment: activeOnly(settingsRef.current),
          reference: reference.filepath || "", reference_image: reference.image || "" }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) { setMatchError(data.detail || "Could not match that look"); return; }
      const amount = 75;
      const shoot = (p) => p.split("/").slice(0, -1).join("/");
      setMatched({ found: data.settings, amount, note: data.note, name: reference.name,
        base: matchBase(settingsRef.current),
        elsewhere: !!reference.filepath && shoot(reference.filepath) !== shoot(filepath) });
      setActivePreset("");
      applySettings({ ...settingsRef.current, ...matchAt(data.settings, amount / 100) },
        `Look matched to ${reference.name}`);
    } catch {
      setMatchError("The server did not answer");
    } finally {
      setMatchBusy(false);
    }
  }, [reference, filepath, applySettings]);
  const setMatchAmount = useCallback((amount) => {
    setMatched((m) => (m ? { ...m, amount } : m));
    const m = matched;
    if (!m) return;
    // A drag: shown at once, saved when the hand stops, one step in the history.
    const next = { ...settingsRef.current, ...matchAt(m.found, amount / 100) };
    record(next, { label: `Match ${m.name} ${amount}%`, group: "match" });
    setSettings(next);
    settingsRef.current = next;
    scheduleSave(next);
  }, [matched, record, scheduleSave]);

  const runAuto = useCallback(async () => {
    setBusy(true);
    setNote("Reading this frame…");
    try {
      const res = await fetch(`${API}/export/develop/auto`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // The same size the preview is made at, so the measurement reuses the
        // decode already held for it instead of reading the file again.
        body: JSON.stringify({ filepath, size: PREVIEW_SIZE }),
      });
      const data = res.ok ? await res.json() : null;
      const found = data?.settings || {};
      if (!Object.keys(found).length) {
        setNote("Auto found nothing worth changing on this frame");
        return;
      }
      // Auto decides the light — exposure included — and leaves the colour and
      // the look alone. It replaces the light controls rather than adding to
      // them, so pressing it twice is the same as pressing it once.
      const next = { ...settingsRef.current };
      for (const key of LIGHT_KEYS) next[key] = 0;
      Object.assign(next, found);
      const moved = Object.keys(found)
        .map((k) => `${k} ${k === "exposure" ? Number(found[k]).toFixed(2) : signed(found[k])}`)
        .join(", ");
      applySettings(next, `Auto: ${moved}`);
    } catch {
      setNote("Auto could not read this frame");
    } finally {
      setBusy(false);
    }
  }, [filepath, applySettings]);

  const loadPresets = useCallback(async () => {
    try {
      const res = await fetch(`${API}/export/develop/presets`);
      const data = await res.json();
      setPresets(data.presets || []);
    } catch { /* the menu is a convenience */ }
  }, []);

  useEffect(() => { loadPresets(); }, [loadPresets]);

  const applyPreset = useCallback((name) => {
    const found = presets.find((p) => p.name === name);
    if (!found) return;
    setActivePreset(name);
    applySettings(withFramingOf(asSettings(found.settings), settingsRef.current), `Preset “${name}”`);
  }, [presets, applySettings]);

  const savePreset = useCallback(async () => {
    const name = presetName.trim();
    if (!name || !isToneEdited(settingsRef.current)) return;
    const res = await fetch(`${API}/export/develop/presets`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, settings: activeOnly(settingsRef.current, false) }),
    });
    if (!res.ok) { setNote("Could not save the preset"); return; }
    const data = await res.json();
    setPresets(data.presets || []);
    setActivePreset(name);
    setPresetName("");
    setNote(`Saved preset “${name}”`);
  }, [presetName]);

  const deletePreset = useCallback(async () => {
    if (!activePreset) return;
    await fetch(`${API}/export/develop/presets/${encodeURIComponent(activePreset)}`, {
      method: "DELETE",
    });
    setActivePreset("");
    loadPresets();
    setNote("Preset forgotten");
  }, [activePreset, loadPresets]);

  // Copy the ticked sections (copyChoice). Each is carried as it is on this
  // frame, untouched ones too: pasting Light from a frame with no Light edit
  // takes the target's back to zero, as copying a setting does anywhere.
  const copySettings = useCallback(() => {
    const sections = chosenSections();
    const keys = sections.flatMap((s) => s.keys);
    const active = activeOnly(settingsRef.current);
    if (!sections.length) { setNote("Nothing is ticked to copy"); return; }
    if (!keys.some((k) => k in active)) { setNote("None of what is ticked is set on this frame"); return; }
    copyEdit({
      settings: active, keys, source: filepath, landscape: frameDims.w >= frameDims.h,
      sections: sectionNames(sections),
    });
    setNote(`Copied ${sectionNames(sections).join(" + ")}. Ctrl+V on another frame, or Paste edit on a selection in the grid`);
  }, [filepath, frameDims]);

  // The copied sections onto this frame; the rest of its edit stays its own.
  const pasteSettings = useCallback(async () => {
    const copied = copiedEdit();
    if (!copied) { setNote("Nothing copied yet"); return; }
    const current = settingsRef.current;
    const turned = (frameDims.w >= frameDims.h) !== copied.landscape;
    const keys = turned ? copied.keys.filter((k) => !PLACED_KEYS.includes(k)) : copied.keys;
    const next = { ...current };
    for (const key of keys) next[key] = key in copied.settings ? copied.settings[key] : ZERO[key];
    // A level on its own turns this frame's own crop, which has to stay on picture.
    const own = boxOf(current);
    if (keys.includes("angle") && !keys.includes("crop_x") && own) {
      const fitted = fitBox(own, Number(next.angle) || 0, frameDims.w, frameDims.h);
      Object.assign(next, { crop_x: fitted.x, crop_y: fitted.y, crop_w: fitted.w, crop_h: fitted.h });
    }
    setActivePreset("");
    const skipped = turned && keys.length < copied.keys.length;
    const label = skipped ? "Edit pasted (crop and spots kept: this frame is turned)" : "Edit pasted";
    const mode = pasteAs();
    if (mode === "keep_light") {
      for (const key of ["exposure", "temperature", "tint"]) next[key] = key in current ? current[key] : ZERO[key];
      applySettings(next, `${label}, keeping this frame's exposure and white balance`);
      return;
    }
    if (mode === "copied" || !copied.source || copied.source === filepath) {
      applySettings(next, label);
      return;
    }
    // Matching measures this frame against the one copied from (backend
    // export/paste_match.py); the result goes on as one step, like any paste.
    setNote(mode === "match_exposure" ? "Matching the light…" : "Matching the look…");
    try {
      const res = await fetch(`${API}/photos/edit/paste-adjust`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filepath, source: copied.source, copied: copied.settings,
          edit: activeOnly(next), own: activeOnly(current), paste_as: mode }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) { setNote(data.detail || "Could not match: pasted as copied"); applySettings(next, label); return; }
      const got = data.settings || {};
      const changed = mode === "match_exposure" ? ["exposure", "temperature", "tint"] : [...CURVE_KEYS, "saturation", "mono"];
      for (const key of changed) next[key] = key in got ? got[key] : ZERO[key];
      applySettings(next, data.note ? `${label} · ${data.note}` : label);
    } catch {
      setNote("The server did not answer: pasted as copied");
      applySettings(next, label);
    }
  }, [applySettings, frameDims, filepath]);

  const tickCopy = useCallback((ids) => {
    setCopyTicks(ids);
    keepCopyChoice(ids);
  }, []);

  const applyToBurst = useCallback(async () => {
    setBusy(true);
    try {
      const res = await fetch(`${API}/photos/edit/burst`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filepath, ...activeOnly(settingsRef.current),
          keys: chosenSections().flatMap((s) => s.keys), paste_as: pasteAs() }),
      });
      const data = res.ok ? await res.json() : null;
      const copied = data?.copied?.length || 0;
      const how = copied ? pasteSummary(pasteAs(), data.notes) : "";
      setNote(copied
        ? `Copied to ${copied} other frame${copied === 1 ? "" : "s"} of this burst${how ? ` · ${how}` : ""}`
        : "No other frame in this burst");
      if (copied) onSaved?.(null, null);
    } catch {
      setNote("Could not copy to the burst");
    } finally {
      setBusy(false);
    }
  }, [filepath, onSaved]);

  // ── Virtual copies ─────────────────────────────────────────────
  // Another edit of this same file: a colour and a black-and-white, a square for
  // the grid and a 9:16 for Stories. Each has its own tone and its own crop.
  const flushSave = useCallback(() => {
    if (saveTimer.current) clearTimeout(saveTimer.current);
    saveTimer.current = null;
    if (pendingSave.current) {
      persistRef.current(pendingSave.current);
      pendingSave.current = null;
    }
  }, []);

  const switchVersion = useCallback(async (target) => {
    if (target === version) return;
    flushSave();
    try {
      const res = await fetch(
        `${API}/photos/edit?filepath=${encodeURIComponent(filepath)}&version=${target}`);
      if (!res.ok) throw new Error("copy");
      const data = await res.json();
      const next = asSettings(data.settings || {});
      historyKeyRef.current = `${filepath}#${target}`;
      setVersion(target);
      setSettings(next);
      settingsRef.current = next;
      startHistory(next);
      setActivePreset("");
      setCropMode(false);
      setDraft(null);
      setNote("");
    } catch {
      setNote("Could not open that copy");
    }
  }, [version, filepath, flushSave, startHistory]);

  const newCopy = useCallback(async () => {
    flushSave();
    setBusy(true);
    try {
      const res = await fetch(`${API}/photos/versions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filepath, from_version: version }),
      });
      if (!res.ok) throw new Error("copy");
      const data = await res.json();
      setCopies(data.copies || []);
      await switchVersion(data.version);
      setNote("New copy made from this edit — change it freely; the other stays as it was");
    } catch {
      setNote("Could not make a copy");
    } finally {
      setBusy(false);
    }
  }, [filepath, version, flushSave, switchVersion]);

  const patchCopy = useCallback(async (patch) => {
    const res = await fetch(`${API}/photos/versions`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filepath, version, ...patch }),
    });
    if (res.ok) setCopies((await res.json()).copies || []);
  }, [filepath, version]);

  const renameCopy = useCallback(() => {
    const found = copies.find((c) => c.version === version);
    const name = window.prompt("Name this copy", found?.name || "");
    if (name !== null) patchCopy({ name });
  }, [copies, version, patchCopy]);

  const deleteCopy = useCallback(async () => {
    if (!version || !window.confirm("Delete this copy? The photo and its other copies are untouched.")) return;
    if (saveTimer.current) clearTimeout(saveTimer.current);
    saveTimer.current = null;
    pendingSave.current = null;
    const gone = version;
    const res = await fetch(
      `${API}/photos/versions?filepath=${encodeURIComponent(filepath)}&version=${gone}`,
      { method: "DELETE" });
    if (!res.ok) { setNote("Could not delete this copy"); return; }
    setCopies((await res.json()).copies || []);
    // Back to the photo's own edit; version 0 is read afresh from the server.
    const back = await fetch(`${API}/photos/edit?filepath=${encodeURIComponent(filepath)}`)
      .then((r) => r.json()).catch(() => ({ settings: {} }));
    const next = asSettings(back.settings || {});
    HISTORIES.delete(`${filepath}#${gone}`);
    historyKeyRef.current = `${filepath}#0`;
    setVersion(0);
    setSettings(next);
    settingsRef.current = next;
    startHistory(next);
    setCropMode(false);
    setNote("Copy deleted");
  }, [version, filepath]);

  const step = useCallback((delta) => {
    if (!photos?.length) return;
    const next = position + delta;
    if (next < 0 || next >= photos.length) return;
    onNavigate?.(photos[next].filepath);
  }, [photos, position, onNavigate]);

  // Keyboard is the whole point of a develop view: the sliders are for the
  // fine decision, and moving between frames should not need the mouse.
  useEffect(() => {
    const onKey = (e) => {
      if (e.timeStamp < openedAt.current) return;
      const tag = e.target?.tagName;
      // A range input keeps focus after it is dragged, and its arrow keys are
      // the slider's own: they nudge the value, they do not change frame.
      const typing = tag === "INPUT" && e.target.type !== "range";
      // Escape backs out one layer at a time: the crop, then the loupe, then the view.
      if (e.key === "Escape") {
        if (cropMode) endCrop();
        else if (maskMode) endMasks();
        else if (spotMode) setSpotMode(false);
        else if (loupe) placeLoupe(null);
        else onClose();
        return;
      }
      if (typing || tag === "SELECT") return;
      if (e.key === "Enter" && cropMode) { endCrop(); return; }

      // The edit clipboard is the page's own (editClipboard.js), never the
      // system's: nothing copied elsewhere can reach an edit. Text selected on
      // the page is still copied the usual way, and a held key does not paste
      // over and over.
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "c") {
        if (!e.shiftKey && String(window.getSelection?.() || "").trim()) return;
        e.preventDefault();
        if (e.repeat) return;
        if (e.shiftKey) setCopyPicker((open) => !open); else copySettings();
        return;
      }
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "v") {
        e.preventDefault();
        if (!e.repeat) pasteSettings();
        return;
      }
      if ((e.ctrlKey || e.metaKey) && (e.key.toLowerCase() === "y"
          || (e.key.toLowerCase() === "z" && e.shiftKey))) {
        e.preventDefault(); redo(); return;
      }
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
        e.preventDefault(); undo(); return;
      }
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (tag === "INPUT" && (e.key === "ArrowLeft" || e.key === "ArrowRight")) return;

      switch (e.key) {
        case "ArrowLeft": e.preventDefault(); step(-1); break;
        case "ArrowRight": e.preventDefault(); step(1); break;
        case "\\": setShowOriginal(true); break;
        case "0": reset(); break;
        case "a": case "A": runAuto(); break;
        case "b": case "B": if (version === 0) applyToBurst(); break;
        case "j": case "J": setShowClip((v) => !v); break;
        case "c": case "C": if (cropMode) endCrop(); else startCrop(); break;
        case "l": case "L":
          if (cropMode) break;
          placeLoupe(loupe ? null : { u: 0.5, v: 0.5 });
          break;
        // D opens this view, so D closes it: the same key goes there and back.
        case "d": case "D": if (!e.repeat) onClose(); break;
        // The brush's size, as in Lightroom.
        case "[":
          if (maskMode) setBrush({ size: Math.max(0.0005, brushRef.current.size / 1.2) });
          else if (spotMode) setSpotSize((v) => Math.max(SPOT_SIZE.min, v / 1.2));
          break;
        case "]":
          if (maskMode) setBrush({ size: Math.min(0.25, brushRef.current.size * 1.2) });
          else if (spotMode) setSpotSize((v) => Math.min(SPOT_SIZE.max, v * 1.2));
          break;
        // Q for spots, as in Lightroom.
        case "q": case "Q": toggleSpots(); break;
        case "Delete": case "Backspace":
          if (spotMode && spotSel >= 0) { e.preventDefault(); removeSpot(spotSel); }
          break;
        default: break;
      }
    };
    const onUp = (e) => { if (e.key === "\\") setShowOriginal(false); };
    window.addEventListener("keydown", onKey);
    window.addEventListener("keyup", onUp);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("keyup", onUp);
    };
  }, [onClose, copySettings, pasteSettings, step, reset, runAuto, applyToBurst,
    cropMode, endCrop, startCrop, maskMode, endMasks, loupe, placeLoupe, version, setBrush, undo, redo,
    spotMode, spotSel, toggleSpots, removeSpot]);

  // Object URLs are the one thing here that leaks if it is forgotten. A save
  // still waiting on its debounce is sent now: closing the view or stepping to
  // the next frame within half a second of a drag must not lose the edit.
  useEffect(() => () => {
    if (previewUrl.current) URL.revokeObjectURL(previewUrl.current);
    if (originalUrl.current) URL.revokeObjectURL(originalUrl.current);
    if (geoOriginalUrl.current) URL.revokeObjectURL(geoOriginalUrl.current);
    if (loupeObjectUrl.current) URL.revokeObjectURL(loupeObjectUrl.current);
    if (previewTimer.current) clearTimeout(previewTimer.current);
    if (saveTimer.current) clearTimeout(saveTimer.current);
    if (pendingSave.current) {
      persistRef.current(pendingSave.current, true);
      pendingSave.current = null;
    }
  }, []);

  const framed = hasFraming(settings);
  const originalShown = fullView || !framed
    ? original
    : geoOriginal.key === framingKey
      ? geoOriginal.url
      : (preview && !previewFull ? preview : original);
  // The preview on screen is only usable if it was drawn for the same mode: a
  // cropped picture under the crop box would put the box on the wrong pixels.
  const previewReady = preview && (previewFull === fullView || !framed);
  const shown = showOriginal || !previewReady ? originalShown : preview;
  const current = photos?.[position];
  const edited = isEdited(settings);
  const toneEdited = isToneEdited(settings);
  const angle = Number(settings.angle) || 0;
  const activeAspect = ASPECTS.find((a) => a.id === aspect);
  const ratioFrac = activeAspect && activeAspect.ratio
    ? ((activeAspect.ratio === "orig" ? frameDims.w / frameDims.h : activeAspect.ratio) * frameDims.h) / frameDims.w
    : null;
  const box = draft || boxOf(settings) || FULL_BOX;
  const shape = (box.w * frameDims.w) / (box.h * frameDims.h);

  // One group of sliders by its id, with what belongs to it: the treatment,
  // the camera-look note and the style under Profile, the rescues under Light.
  const section = (id) => {
    const group = GROUPS.find((g) => g.id === id);
    return (
      <div className="space-y-2">
        <div className="flex items-center justify-between text-[10px] uppercase tracking-wide text-neutral-600 pt-1">
          <span title={group.hint}>{group.title}</span>
          {[...group.sliders.map((s) => s.key), ...(group.switches || [])].some((k) => Number(settings[k]) !== 0) && (
            <button
              onClick={() => applyGroup([...group.sliders.map((s) => s.key), ...(group.switches || [])], {}, `${group.title} reset`)}
              className="normal-case tracking-normal text-neutral-600 hover:text-neutral-300"
              title={`Reset ${group.title.toLowerCase()} controls`}>reset</button>
          )}
        </div>
        {id === "profile" && (
          <div className="flex items-center gap-2 text-xs" data-testid="treatment">
            <span className="w-20 shrink-0 text-neutral-400" title="Develop this photo in colour or in black and white. The black-and-white mix and toning are kept when you switch back.">Treatment</span>
            {[{ value: 0, label: "Colour" }, { value: 1, label: "Black & white" }].map((t) => (
              <button key={t.value} onClick={() => change("mono", t.value)}
                className={`px-2 py-0.5 rounded border ${Number(settings.mono) === t.value
                  ? "border-neutral-400 text-white bg-neutral-800" : "border-neutral-800 text-neutral-500 hover:text-neutral-300"}`}>
                {t.label}
              </button>
            ))}
          </div>
        )}
        {id === "transform" && (
          <div className="space-y-1" data-testid="upright">
            <div className="flex flex-wrap gap-1">
              {[["vertical", "Auto: verticals", "Make the photo's vertical lines vertical: sets Vertical and the Level"],
                ["level", "Auto: level", "Make the photo's level lines (a horizon, a stage edge) level: sets the Level"]].map(([mode, label, title]) => (
                <button key={mode} onClick={() => runUpright(mode)} title={title} disabled={!!upright.busy}
                  className="text-[11px] px-2 py-0.5 rounded border border-neutral-700 text-neutral-300 hover:text-white hover:border-neutral-500 disabled:opacity-50">
                  {upright.busy === mode ? "looking…" : label}
                </button>
              ))}
            </div>
            {upright.note && <p className="text-[10px] text-neutral-500" data-testid="upright-note">{upright.note}</p>}
          </div>
        )}
        {id === "lens" && (
          <div className="flex items-center gap-2 text-xs" data-testid="lens-profile">
            <span className="w-20 shrink-0 text-neutral-400"
              title="Straighten the lens's distortion as the camera straightened its own JPEG of this photo. RAW files only.">Distortion</span>
            {[{ value: 0, label: "As shot" }, { value: 1, label: "Camera's correction" }].map((t) => (
              <button key={t.value} onClick={() => change("lens_profile", t.value)}
                className={`px-2 py-0.5 rounded border ${Number(settings.lens_profile) === t.value
                  ? "border-neutral-400 text-white bg-neutral-800" : "border-neutral-800 text-neutral-500 hover:text-neutral-300"}`}>
                {t.label}
              </button>
            ))}
          </div>
        )}
        {id === "profile" && LOOK_NOTES[lookNote.state] && (() => {
          const note = LOOK_NOTES[lookNote.state](lookNote.frames);
          const hint = LOOK_HINTS[lookNote.hint];
          const folder = current?.folder;
          const running = learnJob?.status === "running";
          return (
            <div className="text-[10px] text-neutral-600 space-y-0.5">
              <div title={note.title} data-testid="look-note">Camera look: {note.text}</div>
              {(hint || running) && (
                <div data-testid="look-hint">
                  {running ? `Learning from this folder… ${learnJob.done}/${learnJob.total}` : hint}
                  {!running && folder && (
                    <button onClick={() => learnFolder(folder)}
                      title="Open every RAW in this folder so the camera's look is learned from all of them. Runs in the background."
                      className="ml-1 underline decoration-neutral-700 hover:text-neutral-300">
                      Learn from this folder
                    </button>
                  )}
                </div>
              )}
            </div>
          );
        })()}
        {id === "profile" && (
          <StylePicker value={settings.style || ""} folder={current?.folder} filepath={filepath}
            edit={activeOnly(settings)} onChange={(style) => change("style", style)}
            onRefitted={() => setLookRefresh((n) => n + 1)} />
        )}
        {id === "light" && (
          <div className="flex flex-wrap gap-1">
            {RESCUES.map((r) => (
              <button key={r.name} title={`${r.note}. Sets the Light controls.`}
                onClick={() => applyGroup(LIGHT_KEYS, r.settings, r.name)}
                className="text-[11px] px-2 py-0.5 rounded border border-neutral-700 text-neutral-300 hover:text-white hover:border-neutral-500">
                {r.name}
              </button>
            ))}
          </div>
        )}
        {id === "noise" && NOISE_NOTES[noiseState] && (
          <div className="text-[10px] text-neutral-600 -mt-1" data-testid="noise-note">
            {NOISE_NOTES[noiseState]}
          </div>
        )}
        <Controls>
          {group.sliders.filter((def) => (!def.needsLook || hasLook)
            && (!def.needsStyle || settings.style)
            && (!def.needsSharpen || Number(settings.sharpen) > 0)
            && !(Number(settings.mono) && COLOUR_ONLY_KEYS.includes(def.key))).map((def) => (
            <Slider key={def.key} def={def} value={settings[def.key]}
              onChange={change} onReset={() => resetOne(def.key)} />
          ))}
        </Controls>
      </div>
    );
  };

  return (
    <div data-modal="develop" className="fixed inset-0 z-[60] bg-neutral-950 text-neutral-100 flex flex-col">
      <header className="flex items-center justify-between px-5 py-2 border-b border-neutral-800 shrink-0">
        <div className="flex items-baseline gap-3 min-w-0">
          <span className="font-medium truncate">{current?.filename || filepath}</span>
          <span className="text-xs text-neutral-500 truncate">{current?.folder}</span>
          {version !== 0 && (
            <span className="text-xs text-sky-400" title="You are editing a virtual copy of this photo">
              {copies.find((c) => c.version === version)?.name || `Copy ${version}`}
            </span>
          )}
          {edited && (
            <span className="text-xs text-amber-400" title="This frame carries an edit">edited</span>
          )}
          {rendering && <span className="text-xs text-neutral-600">rendering…</span>}
        </div>
        <div className="flex items-center gap-3 text-xs text-neutral-400">
          <span>{position >= 0 ? `${position + 1} / ${photos.length}` : ""}</span>
          <button onClick={onClose} className="px-2 py-1 rounded border border-neutral-700 hover:text-white"
            title="Close (Esc or D)">Close ✕</button>
        </div>
      </header>

      <div className="flex-1 flex min-h-0">
        {/* The picture, big enough to judge. Everything below exists to make a
            subtle change visible: the compare keys, the split, the histogram. */}
        <div className="flex-1 relative bg-black flex items-center justify-center min-w-0">
          {loading && <span className="text-neutral-500 text-sm">Loading…</span>}
          {!loading && !original && (
            <span className="text-neutral-500 text-sm">Could not render this frame</span>
          )}
          {original && (
            <div className="relative max-h-full max-w-full overflow-hidden"
              style={{ lineHeight: 0, cursor: fullView ? "default" : "crosshair" }}
              onClick={onPictureClick}
              onPointerMove={onPictureMove} onPointerLeave={() => setProbe(null)}
              title={fullView ? undefined : "Click to look at this spot at 1:1"}>
              <img ref={imgRef} src={shown} alt="" draggable={false}
                className="max-h-[calc(100vh-9rem)] max-w-full object-contain"
                style={cropMode && angle ? { transform: `rotate(${angle}deg)` } : undefined} />
              {split > 0 && preview && !fullView && (
                <div className="absolute inset-0 overflow-hidden pointer-events-none"
                  style={{ width: `${split}%` }}>
                  <img src={originalShown} alt=""
                    className="max-h-[calc(100vh-9rem)] max-w-full object-contain" />
                </div>
              )}
              {cropMode && (
                <CropOverlay box={box} ratio={ratioFrac} angle={angle} dims={frameDims}
                  onDraft={setDraft}
                  onCommit={(final) => { setDraft(null); commitBox(final); }} />
              )}
              {spotMode && (
                <SpotOverlay spots={settings.spots || []} selected={spotSel} dims={frameDims} size={spotSize}
                  onSelect={setSpotSel} onAdd={addSpot} onChange={updateSpot} />
              )}
              {maskMode && (
                <MaskOverlay masks={settings.masks || []} selected={maskSel} dims={frameDims}
                  brush={brush} showShade={showShade} mattes={mattes} people={people}
                  range={rangeImage} picking={picking} onPick={pickColour}
                  onSelect={setMaskSel} onChange={updateMask} onStroke={addStroke} />
              )}
              {loupe && !fullView && (
                <span className="absolute w-5 h-5 -ml-2.5 -mt-2.5 rounded-full border-2 border-amber-400 pointer-events-none"
                  style={{ left: `${loupe.u * 100}%`, top: `${loupe.v * 100}%`,
                    boxShadow: "0 0 0 1px rgba(0,0,0,0.6)" }} />
              )}
              {showClip && clipUrl && !showOriginal && (
                <img src={clipUrl} alt="" className="absolute inset-0 w-full h-full pointer-events-none" />
              )}
            </div>
          )}

          {loupe && !fullView && (
            <div className={`absolute top-3 ${loupe.u > 0.5 ? "left-3" : "right-3"} z-10 bg-neutral-900/95 border border-neutral-700 rounded shadow-xl`}
              style={{ maxWidth: "46%" }}>
              {/* One line of fixed height, whatever it says: a header that grew a
                  second line as a patch was read moved the picture under it. */}
              <div className="flex items-center justify-between gap-3 px-2 h-6 text-[10px] text-neutral-400 whitespace-nowrap">
                <span className="flex items-center gap-1" data-testid="loupe-zoom">
                  {[1, 2].map((z) => (
                    <button key={z} onClick={() => setLoupeZoom(z)}
                      title={z === 1 ? "The photograph's pixels one for one" : "Each of the photograph's pixels as four on the screen"}
                      className={`px-1 rounded ${loupeZoom === z ? "bg-neutral-700 text-white" : "hover:text-white"}`}>{z}:1</button>
                  ))}
                  {showOriginal ? <span className="ml-1">original</span> : null}
                  <span className={`ml-1 w-1.5 h-1.5 rounded-full ${loupeState === "loading" ? "bg-amber-400 animate-pulse"
                    : loupeState === "failed" ? "bg-red-500" : "bg-transparent"}`}
                    title={loupeState === "loading" ? "Reading the full-size frame" : loupeState === "failed" ? "Could not read this spot" : undefined}
                    data-testid="loupe-busy" />
                </span>
                <button onClick={() => placeLoupe(null)} className="hover:text-white" title="Close the loupe (Esc)">✕</button>
              </div>
              {loupeUrl ? (
                <img src={loupeUrl} alt="" draggable={false} data-testid="loupe-image"
                  style={{ display: "block", width: `${LOUPE_SIDE / (window.devicePixelRatio || 1)}px`, maxWidth: "100%",
                    // Enlarged, each pixel stays a pixel: a soft upscale would pass for soft focus.
                    imageRendering: loupeZoom > 1 ? "pixelated" : "auto" }} />
              ) : (
                <div className="w-64 h-40 flex items-center justify-center text-xs text-neutral-600">
                  {loupeState === "failed" ? "—" : "Reading full-size frame…"}
                </div>
              )}
            </div>
          )}

          <div className="absolute bottom-3 left-1/2 -translate-x-1/2 flex items-center gap-2 text-xs">
            <button onClick={() => (cropMode ? endCrop() : startCrop())}
              className={`px-2 py-1 rounded border ${cropMode ? "border-amber-500 text-amber-400" : "border-neutral-700 text-neutral-400 hover:text-white"}`}
              title="Crop and level (C)">
              {cropMode ? "Done" : "Crop"}
            </button>
            <button onClick={() => placeLoupe(loupe ? null : { u: 0.5, v: 0.5 })} disabled={cropMode}
              className={`px-2 py-1 rounded border disabled:opacity-40 ${loupe ? "border-amber-500 text-amber-400" : "border-neutral-700 text-neutral-400 hover:text-white"}`}
              title="1:1 loupe: click the picture to look at that spot at full size (L)">
              Loupe
            </button>
            <button onClick={() => setSplit(split > 0 ? 0 : 50)}
              className={`px-2 py-1 rounded border ${split > 0 ? "border-amber-500 text-amber-400" : "border-neutral-700 text-neutral-400 hover:text-white"}`}
              title="Show the original on the left, the edit on the right">
              Split
            </button>
            <button onClick={() => setShowClip((v) => !v)}
              className={`px-2 py-1 rounded border ${showClip ? "border-red-500 text-red-400" : "border-neutral-700 text-neutral-400 hover:text-white"}`}
              title="Mark clipped areas: red is pure white, blue is pure black (J)">
              Clipping
            </button>
            <span className="text-neutral-600">hold \ for the original</span>
          </div>
        </div>

        <ControlStyle.Provider value={controlStyle}>
        <aside className="w-80 2xl:w-96 shrink-0 border-l border-neutral-800 overflow-y-auto p-4 space-y-4">
          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-[10px] uppercase tracking-wide text-neutral-600">
              <span title="Virtual copies: more than one edit of this same file — say colour and black-and-white, or a square and a 9:16. No second file is made; an export writes one picture per copy.">
                Versions
              </span>
            </div>
            <div className="flex flex-wrap gap-1">
              <button onClick={() => switchVersion(0)}
                title="This photograph's own edit"
                className={`text-[11px] px-2 py-0.5 rounded border ${version === 0
                  ? "border-amber-500 text-amber-400"
                  : "border-neutral-700 text-neutral-300 hover:text-white"}`}>Main</button>
              {copies.map((c) => (
                <button key={c.version} onClick={() => switchVersion(c.version)}
                  title={c.export ? "Exported with this photo" : "Not exported"}
                  className={`text-[11px] px-2 py-0.5 rounded border max-w-[8rem] truncate ${version === c.version
                    ? "border-amber-500 text-amber-400"
                    : "border-neutral-700 text-neutral-300 hover:text-white"} ${c.export ? "" : "opacity-60"}`}>
                  {c.name || `Copy ${c.version}`}
                </button>
              ))}
              <button onClick={newCopy} disabled={busy}
                title={`Make another version of this photo, starting from ${version === 0 ? "the main edit" : "this copy"}`}
                className="text-[11px] px-2 py-0.5 rounded border border-dashed border-neutral-600 text-neutral-400 hover:text-white disabled:opacity-40">
                + Copy
              </button>
            </div>
            {version !== 0 && (
              <div className="flex items-center gap-3 text-[11px] text-neutral-400">
                <button onClick={renameCopy} className="hover:text-white">Rename</button>
                <label className="flex items-center gap-1 cursor-pointer"
                  title="Write this copy when the photo is exported">
                  <input type="checkbox" className="accent-amber-400"
                    checked={copies.find((c) => c.version === version)?.export ?? true}
                    onChange={(e) => patchCopy({ export: e.target.checked })} />
                  export
                </label>
                <button onClick={deleteCopy} className="hover:text-red-400 ml-auto">Delete</button>
              </div>
            )}
          </div>

          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs uppercase tracking-wide text-neutral-500">Histogram</span>
              <span className="text-[10px] text-neutral-500"
                title="The grey line is the original's brightness, the white line the edit's; the red, green and blue the edit's channels.">grey original · white edit · RGB</span>
            </div>
            <Histogram original={originalHist} edited={editedHist} channels={editedChannels} />
            {/* The pixel under the cursor: where it is in the whole photograph and
                its colour in the edited picture, always in this one place. */}
            <div className="flex items-center gap-2 mt-1 h-5 text-[10px] font-mono text-neutral-300" data-testid="pixel-readout">
              {(() => {
                if (!probe) return <span className="text-neutral-600 font-sans">Point at the picture for its pixel values</span>;
                const b = boxOf(settings) || FULL_BOX;
                const framed = !maskMode && !spotMode && !cropMode;
                const fx = framed ? b.x + probe.u * b.w : probe.u;
                const fy = framed ? b.y + probe.v * b.h : probe.v;
                const fmt = READOUTS.find((r) => r.id === readout) || READOUTS[0];
                return (
                  <>
                    <span className="w-3 h-3 rounded-sm border border-neutral-600 shrink-0"
                      style={{ background: `rgb(${probe.rgb.join(",")})` }} />
                    <span className="whitespace-nowrap">{fmt.show(probe.rgb)}</span>
                    {photoSize && (
                      <span className="text-neutral-500 whitespace-nowrap ml-auto">
                        {Math.round(fx * photoSize.w)}, {Math.round(fy * photoSize.h)}
                      </span>
                    )}
                  </>
                );
              })()}
            </div>
            <label className="flex items-center justify-end gap-1 mt-0.5 text-[10px] text-neutral-500"
              title="What the readout shows for the pixel under the cursor, in the edited picture">
              Values in
              <select value={readout} onChange={(e) => setReadout(e.target.value)} data-testid="readout-space"
                className="bg-neutral-900 border border-neutral-800 rounded px-1 text-[10px] text-neutral-300">
                {READOUTS.map((r) => <option key={r.id} value={r.id}>{r.label}</option>)}
              </select>
            </label>
            <div className="flex justify-between mt-1 text-[10px] font-mono">
              <span className={clip.low > 0.005 ? "text-blue-400" : "text-neutral-600"}
                title="Share of the picture that is pure black">▮ black {percent(clip.low)}</span>
              <span className={clip.high > 0.005 ? "text-red-400" : "text-neutral-600"}
                title="Share of the picture that is pure white">white {percent(clip.high)} ▮</span>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button onClick={runAuto} disabled={busy}
              className="text-xs px-2 py-1 rounded border border-neutral-700 text-amber-400 hover:text-amber-300 disabled:opacity-40"
              title="Set exposure, highlights, shadows and squeeze from this frame's own histogram (A)">
              {busy ? "Reading…" : "Auto"}
            </button>
            <button onClick={reset} disabled={!toneEdited}
              className="text-xs px-2 py-1 rounded border border-neutral-700 text-neutral-300 hover:text-white disabled:opacity-40"
              title="Undo the tone and colour changes and the gradients, keeping the crop (0)">Reset</button>
            <span className="flex-1" />
            <button onClick={undo} disabled={history.index < 1}
              className="text-sm px-1.5 text-neutral-400 hover:text-white disabled:opacity-30"
              title={history.index >= 1 ? `Undo ${history.entries[history.index].label} (Ctrl+Z)` : "Nothing to undo"}>↶</button>
            <button onClick={redo} disabled={history.index >= history.entries.length - 1}
              className="text-sm px-1.5 text-neutral-400 hover:text-white disabled:opacity-30"
              title={history.index < history.entries.length - 1
                ? `Redo ${history.entries[history.index + 1].label} (Ctrl+Shift+Z)` : "Nothing to redo"}>↷</button>
          </div>
          <HistoryPanel history={history} onJump={jumpTo} />

          <div className="space-y-2">
            <div className="flex items-center justify-between text-[10px] uppercase tracking-wide text-neutral-600">
              <span title="Crop to a shape and straighten the horizon. Nothing is cut from the file; the crop is part of the edit.">Crop &amp; level</span>
              {framed && (
                <button onClick={resetFraming}
                  className="normal-case tracking-normal text-neutral-600 hover:text-neutral-300"
                  title="Back to the whole frame, unturned">remove</button>
              )}
            </div>
            <div className="flex flex-wrap gap-1">
              {ASPECTS.map((a) => (
                <button key={a.id} title={a.note} onClick={() => chooseAspect(a.id)}
                  className={`text-[11px] px-2 py-0.5 rounded border ${cropMode && aspect === a.id
                    ? "border-amber-500 text-amber-400"
                    : "border-neutral-700 text-neutral-300 hover:text-white hover:border-neutral-500"}`}>
                  {a.label}
                </button>
              ))}
            </div>
            <label className="flex items-center gap-2 text-xs">
              <span className={`w-20 shrink-0 select-none ${angle ? "text-amber-300" : "text-neutral-400"}`}
                title="Turn the picture to level a horizon. Double-click to reset."
                onDoubleClick={() => setAngle(0)}>Level</span>
              <input type="range" min={-15} max={15} step={0.1} value={clamp(angle, -15, 15)}
                onChange={(e) => setAngle(+e.target.value)} onDoubleClick={() => setAngle(0)}
                className="flex-1 accent-amber-400" />
              <input type="number" min={-45} max={45} step={0.1} value={angle}
                onChange={(e) => setAngle(e.target.value)}
                className="w-14 bg-neutral-900 border border-neutral-800 rounded px-1 py-0.5 text-right font-mono text-xs" />
            </label>
            {framed && boxOf(settings) && (
              <p className="text-[10px] text-neutral-500 font-mono">
                {Math.round(box.w * 100)}% × {Math.round(box.h * 100)}% of the frame · {shape.toFixed(2)}:1
              </p>
            )}
          </div>

          {section("profile")}
          {section("wb")}
          {section("light")}
          {section("noise")}
          {section("sharpen")}
          {section("lens")}
          {section("transform")}

          <div className="space-y-2">
            {/* Before Presence on purpose: a look sets contrast, colour, toning,
                grain and vignette, so it is the start of the styling, and picked
                after them it would wipe the fine-tuning out. */}
            <div className="text-[10px] uppercase tracking-wide text-neutral-600 pt-1"
              title="Starting points for the styling. Each sets contrast, colour, toning, grain and vignette, so pick one before fine-tuning those.">Looks</div>
            <div className="flex flex-wrap gap-1">
              {LOOKS.map((r) => (
                <button key={r.name} title={r.note}
                  onClick={() => applyGroup(LOOK_KEYS, r.settings, r.name)}
                  className="text-[11px] px-2 py-0.5 rounded border border-neutral-700 text-neutral-300 hover:text-white hover:border-neutral-500">
                  {r.name}
                </button>
              ))}
            </div>
            <MatchPanel filepath={filepath} reference={reference} match={matched} busy={matchBusy}
              stale={!!matched && matchBase(settings) !== matched.base}
              error={matchError} onPickThis={pickThisAsReference} onFile={referenceFromFile}
              onForget={() => { setReference(null); setMatched(null); }} onMatch={runMatch}
              onAmount={setMatchAmount} />
          </div>

          {section("presence")}

          <div className="space-y-2">
            <div className="flex items-center justify-between text-[10px] uppercase tracking-wide text-neutral-600 pt-1">
              <span title="A point curve on the finished picture, for all channels or one at a time.">Tone curve</span>
              {CURVE_KEYS.some((k) => curveEdited(settings[k])) && (
                <button onClick={() => applyGroup(CURVE_KEYS, {}, "Tone curve reset")}
                  className="normal-case tracking-normal text-neutral-600 hover:text-neutral-300"
                  title="Reset every curve">reset</button>
              )}
            </div>
            <CurveEditor settings={settings} onChange={change} />
          </div>

          {Number(settings.mono) ? (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-[10px] uppercase tracking-wide text-neutral-600 pt-1">
                <span title="How light each colour turns in grey, and how the grey is toned.">B&amp;W mix</span>
                {BW_MIX_KEYS.some((k) => Number(settings[k]) !== 0) && (
                  <button onClick={() => applyGroup(BW_MIX_KEYS, {}, "B&W mix reset")}
                    className="normal-case tracking-normal text-neutral-600 hover:text-neutral-300"
                    title="Back to a neutral conversion">reset</button>
                )}
              </div>
              <BwMix settings={settings} onChange={change} onReset={resetOne} onPreset={applyGroup} />
            </div>
          ) : (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-[10px] uppercase tracking-wide text-neutral-600 pt-1">
                <span title="Hue, saturation and luminance, colour by colour.">Colour mixer</span>
                {HSL_KEYS.some((k) => Number(settings[k]) !== 0) && (
                  <button onClick={() => applyGroup(HSL_KEYS, {}, "Colour mixer reset")}
                    className="normal-case tracking-normal text-neutral-600 hover:text-neutral-300"
                    title="Reset the colour mixer">reset</button>
                )}
              </div>
              <ColourMixer settings={settings} onChange={change} onReset={resetOne} />
            </div>
          )}

          {section("toning")}

          <SpotPanel spots={settings.spots || []} selected={spotSel} spotMode={spotMode} size={spotSize}
            mode={spotKind} busy={spotBusy} onSize={setSpotSize} onMode={setSpotKind} onToggle={toggleSpots}
            onSelect={setSpotSel} onRemove={removeSpot} onClear={clearSpots} onChange={updateSpot} />

          <MaskPanel masks={settings.masks || []} selected={maskSel} maskMode={maskMode}
            brush={brush} onBrush={setBrush} showShade={showShade} onShowShade={setShowShade}
            choosing={choosing} people={people} picking={picking} onPicking={setPicking}
            onSelect={selectMask} onAdd={addMask} onRemove={removeMask} onChange={updateMask}
            onDone={endMasks} />

          {section("effects")}

          <div className="border-t border-neutral-800 pt-3 space-y-2">
            <div className="flex items-center gap-2">
              <select value={activePreset} onChange={(e) => applyPreset(e.target.value)}
                title="Put a saved edit back on this photo"
                className="flex-1 bg-neutral-900 border border-neutral-800 rounded px-2 py-1 text-xs">
                <option value="">My presets…</option>
                {presets.map((p) => (
                  <option key={p.name} value={p.name}>{p.name}</option>
                ))}
              </select>
              {activePreset && (
                <button onClick={deletePreset} title="Forget this preset"
                  className="text-xs text-neutral-500 hover:text-red-400">Delete</button>
              )}
            </div>
            <div className="flex items-center gap-2">
              <input value={presetName} onChange={(e) => setPresetName(e.target.value)}
                placeholder="Name this edit" onKeyDown={(e) => e.key === "Enter" && savePreset()}
                className="flex-1 bg-neutral-900 border border-neutral-800 rounded px-2 py-1 text-xs" />
              <button onClick={savePreset} disabled={!presetName.trim() || !toneEdited}
                className="text-xs text-neutral-400 hover:text-white disabled:opacity-40">Save</button>
            </div>
            <div className="flex items-center gap-2">
              <span className="flex">
                <button onClick={copySettings} data-testid="copy-edit"
                  className="text-xs px-2 py-1 rounded-l border border-neutral-700 text-neutral-300 hover:text-white"
                  title={`Copy ${sectionNames(COPY_SECTIONS.filter((s) => copyTicks.includes(s.id))).join(" + ")} (Ctrl+C)`}>Copy</button>
                <button onClick={() => setCopyPicker((open) => !open)} data-testid="copy-picker"
                  className={`text-xs px-1.5 py-1 rounded-r border border-l-0 border-neutral-700 hover:text-white ${copyPicker ? "text-white bg-neutral-800" : "text-neutral-400"}`}
                  title="Choose what Copy and To burst carry (Shift+Ctrl+C)">▾</button>
              </span>
              <span className="flex">
                <button onClick={pasteSettings} disabled={!clipboard} data-testid="paste-edit-here"
                  className="text-xs px-2 py-1 rounded-l border border-neutral-700 text-neutral-300 hover:text-white disabled:opacity-40"
                  title={clipboard
                    ? `Paste ${clipboard.sections.join(" + ")} onto this frame, ${PASTE_AS.find((p) => p.id === pasteMode).label.toLowerCase()} (Ctrl+V)`
                    : "Nothing copied yet"}>
                  Paste{pasteMode !== "copied" && <span className="text-amber-300"> ·</span>}
                </button>
                <button onClick={() => setPastePicker((open) => !open)} data-testid="paste-picker"
                  className={`text-xs px-1.5 py-1 rounded-r border border-l-0 border-neutral-700 hover:text-white ${pastePicker ? "text-white bg-neutral-800" : "text-neutral-400"}`}
                  title="Choose how Paste, To burst and the grid's Paste edit adjust each frame">▾</button>
              </span>
              <button onClick={applyToBurst} disabled={busy || version !== 0}
                className="text-xs px-2 py-1 rounded border border-neutral-700 text-neutral-300 hover:text-white disabled:opacity-40"
                title={version !== 0 ? "Burst frames take the main edit, not a copy" : "Put this edit on the rest of the burst (B)"}>To burst</button>
            </div>
            {pastePicker && (
              <div className="rounded border border-neutral-800 bg-neutral-900/60 p-2 space-y-1" data-testid="paste-as">
                <div className="text-[10px] text-neutral-500">How Paste, To burst and the grid's Paste edit adjust each frame</div>
                {PASTE_AS.map((opt) => (
                  <label key={opt.id} title={opt.title}
                    className="flex items-start gap-1.5 text-[11px] text-neutral-300 cursor-pointer">
                    <input type="radio" name="paste-as" className="mt-0.5 accent-amber-400"
                      checked={pasteMode === opt.id} onChange={() => keepPasteAs(opt.id)} />
                    <span>{opt.label}</span>
                  </label>
                ))}
                <p className="text-[10px] leading-relaxed text-neutral-600">
                  {PASTE_AS.find((p) => p.id === pasteMode).title}.
                </p>
              </div>
            )}
            {copyPicker && (
              <div className="rounded border border-neutral-800 bg-neutral-900/60 p-2 space-y-1.5" data-testid="copy-sections">
                <div className="flex items-center justify-between text-[10px] text-neutral-500">
                  <span>What Copy and To burst carry</span>
                  <span className="flex gap-2">
                    <button className="hover:text-white" onClick={() => tickCopy(TONE_SECTIONS)}>tone only</button>
                    <button className="hover:text-white" onClick={() => tickCopy(COPY_SECTIONS.map((s) => s.id))}>all</button>
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-x-2 gap-y-0.5">
                  {COPY_SECTIONS.map((sec) => (
                    <label key={sec.id} className={`flex items-center gap-1.5 text-[11px] cursor-pointer ${sec.framing ? "text-sky-200/80" : "text-neutral-300"}`}
                      title={sec.framing ? "Belongs to the frame it was made on: tick it when the camera did not move between frames (a tripod). Not put on a frame turned the other way." : undefined}>
                      <input type="checkbox" className="accent-amber-400" checked={copyTicks.includes(sec.id)}
                        onChange={(e) => tickCopy(e.target.checked
                          ? COPY_SECTIONS.filter((s) => s.id === sec.id || copyTicks.includes(s.id)).map((s) => s.id)
                          : copyTicks.filter((id) => id !== sec.id))} />
                      {sec.label}
                    </label>
                  ))}
                </div>
                <div className="flex justify-end">
                  <button onClick={() => { copySettings(); setCopyPicker(false); }}
                    className="text-xs px-2 py-0.5 rounded border border-amber-600/70 text-amber-300 hover:text-amber-200">Copy these</button>
                </div>
              </div>
            )}
            {note && <p className="text-xs text-neutral-500">{note}</p>}
          </div>

          <ControlStylePicker value={controlStyle} onChange={pickControlStyle} />

          <p className="text-[10px] leading-relaxed text-neutral-600">
            Saved to the catalog and to the photo's <span className="font-mono">.xmp</span> sidecar as
            you go. Nothing is written to the picture itself, so the edit can be changed or
            removed at any time and the original is untouched. RAW frames are developed from the
            sensor data, so exposure and highlights can recover detail a JPEG has already lost.
          </p>
        </aside>
        </ControlStyle.Provider>
      </div>

      <footer className="px-5 py-2 border-t border-neutral-800 text-[11px] text-neutral-600 flex gap-4 shrink-0 flex-wrap">
        <span>← → move</span>
        <span>\ original</span>
        <span>J clipping</span>
        <span>C crop</span>
        <span>L / click loupe</span>
        <span>0 reset</span>
        <span>A auto</span>
        <span>Ctrl+C / Ctrl+V edit</span>
        <span>B to burst</span>
        <span>Esc back out / close</span>
      </footer>
    </div>
  );
}
