# Photool

**A local, free photo library for photographers who shoot RAW: import, cull,
develop and export, the parts of Lightroom you actually use, with no
subscription and no cloud.**

Photool runs on the computer that holds your photographs and opens in your
browser. Your files stay where they are, in ordinary folders. Ratings and edits
go into a local catalog and into standard XMP sidecars, which Lightroom can read
too.

It's built for the everyday work of someone who shoots a lot of events and
family days: get a card onto the disk, get through two thousand frames quickly,
make the keepers look right, and share them.

> **Shared as-is.** Photool is a personal project published in case it's useful
> to others. It's used every day and well tested, but there's no support and no
> roadmap. Bug reports with clear steps are welcome; pull requests may not be
> reviewed.

## What it does

- **Import** from a memory card into date folders, with checksums and
  duplicate detection.
- **Cull fast, by keyboard:** stars, pick/reject, colour labels and tags. Bursts
  are grouped automatically, with "keep the sharpest" and side-by-side compare.
  Soft-focus and blown-highlight warnings, and a 1:1 loupe for checking focus.
- **Develop RAW files in linear light,** starting from the camera's own look,
  which Photool learns from each camera's JPEGs:
  - exposure (±5 EV), highlights and shadows, white balance, tone curve, colour
    mixer, black & white
  - local masks: gradients, brushes, subject, sky, people and faces, and any object you point at or name
    (a box or clicks, chosen by Segment Anything), with zoom and pan to work on small things
  - spot healing, noise reduction (AI), sharpening, lens correction, upright
  - virtual copies, presets, copy/paste and "match this look"

  Edits are non-destructive and written to XMP.
- **People:** faces are found and grouped on your own computer. Name a group
  once and search by name.
- **Search** the whole library in one line (`rating:>=3 iso:>3200 date:2025-06`),
  or by what's in the picture ("kids at the beach").
- **Places, a year in review, collages and a slideshow.**
- **Export** sized JPEGs with your credit in the EXIF, and with location
  stripped if you want.
- **Other devices at home:** browse and cull from a laptop or phone you've
  paired. Nobody else on the Wi-Fi sees anything.
- **AI agent control (optional):** the library is also available as
  [MCP](https://modelcontextprotocol.io) tools, so an agent like Claude Code can
  search, propose a cull with reasons, develop and export. Proposals wait in the
  grid for you to accept.

**Cameras:** Canon CR3 and CR2, DNG (Ricoh GR and others), camera JPEGs, and
most other makes' RAW files through LibRaw (NEF, ARW, ORF, RW2, RAF, PEF). The
camera-look learning works best on cameras that write RAW+JPEG.

## Platforms

| | Status |
|---|---|
| **Windows 10/11** | Supported. What it's developed and used on daily. |
| **macOS** (Apple Silicon or Intel) | Expected to work; less tested. No tray icon or start-at-login yet. |
| **Linux** | Expected to work; untested. |

**No NVIDIA card needed.** Everything runs on the CPU. The one GPU-accelerated
step, noise reduction, uses any graphics card on Windows (DirectML) and the Apple
Silicon GPU and Neural Engine on a Mac (Core ML).

## Getting started

You need Python 3.11+, Node.js 18+, and ideally [ExifTool](https://exiftool.org).

```sh
git clone https://github.com/FoRJu/Photool.git
cd Photool
setup.bat              # macOS / Linux: ./setup.sh
# edit backend/config.yaml: set library_root to your photo folder
serve.bat              # macOS / Linux: ./serve.sh
```

Then open **http://localhost:8000**.

The full walk-through, covering every setting, starting at login, other devices,
updating and moving to a new computer, is in **[docs/setup.md](docs/setup.md)**.

## Documentation

- **[docs/setup.md](docs/setup.md):** installing and configuring it on a new
  computer.
- **[docs/guide.md](docs/guide.md):** how each feature works and why, with the
  measurements behind the design and the keyboard shortcuts.

## Privacy

Faces, descriptions and edits are computed and kept on your computer, and no
photograph or catalog data leaves it. Photool goes online for only two things:

- the optional models below, downloaded the first time you use a feature that
  needs one
- map tiles from OpenStreetMap, when you open a map of where your folders were
  shot

## Models and their licences

Optional models are downloaded on first use. They are not part of this
repository, and each has its own licence. **Check them before any commercial
use.** In particular, the ArcFace weights (InsightFace) and SegFormer (NVIDIA)
are released for non-commercial use only. SAM 3 is under Meta's SAM License,
which is not an open-source licence; it is fetched only if you choose it and agree to it.

| Used for | Model | Source |
|---|---|---|
| Finding faces | YuNet | OpenCV model zoo |
| Grouping faces | ArcFace (InsightFace `buffalo_l` / `buffalo_sc`), or SFace | InsightFace releases, OpenCV model zoo |
| Smart search | CLIP ViT-B/32 (ONNX) | Hugging Face, `Xenova/clip-vit-base-patch32` |
| Noise reduction | TreeNet denoise (RawForge) | github.com/rymuelle/RawForge |
| Subject mask | U²-Net-p | rembg releases |
| Sky and people masks | SegFormer-B0 (ADE20K) | Hugging Face, `Xenova/segformer-b0-finetuned-ade-512-512` |
| Object masks | SAM 2.1 Hiera base+ (Apache-2.0); SAM 3 if chosen (SAM License), our ONNX export of Meta's weights | Hugging Face, `onnx-community/sam2.1-hiera-base-plus-ONNX`, `Photool/sam3-ONNX` |
| Place names | GeoNames `cities5000` | geonames.org (CC BY 4.0) |

## Built with

Python, FastAPI, rawpy/LibRaw, OpenCV, ONNX Runtime, SQLite, React, Vite and
Tailwind CSS. Developed with a lot of help from Claude Code.

## Support and thanks

Photool is free. If it saves you time and you'd like to say thanks, you can
[buy me a coffee on Ko-fi](https://ko-fi.com/forju).

Thanks to [@antVRC](https://github.com/antVRC) for reports and ideas that
improved the gradient tool, the slideshow's look, and masking.

## Licence

MIT. See [LICENSE](LICENSE).
