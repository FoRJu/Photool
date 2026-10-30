# Setting up Photool on a new computer

Photool runs on the computer that holds your photographs. It's a Python server
with a web page as its window: you open it in a browser at
`http://localhost:8000`. Nothing is uploaded anywhere.

- **Windows 10/11:** supported, and what Photool is used on every day.
- **macOS (Apple Silicon or Intel):** expected to work, and less tested. Start-up
  at login and the tray icon are Windows-only for now.
- **Linux:** expected to work like macOS. Untested.

## 1. Install what it needs

| What | Version | Why |
|---|---|---|
| [Python](https://www.python.org/downloads/) | 3.11 or newer (3.12 recommended) | the server |
| [Node.js](https://nodejs.org/) | 18 or newer (the LTS) | builds the web page |
| [ExifTool](https://exiftool.org/) | any recent | fast RAW thumbnails, camera settings, lens data. Optional, strongly recommended |
| [Git](https://git-scm.com/) | any | to download Photool and update it |
| [ffmpeg](https://ffmpeg.org/) | any | thumbnails for videos (GoPro, DJI, phone clips). Optional: without it videos import but show no picture |

**Windows:** install Python from python.org and tick *Add python.exe to PATH*.
Install Node.js from nodejs.org. `setup.bat` installs ExifTool with `winget` if
it's missing.

**macOS:** with [Homebrew](https://brew.sh):

```sh
brew install python@3.12 node exiftool git ffmpeg   # ffmpeg only for videos
```

## 2. Download Photool

```sh
git clone https://github.com/FoRJu/Photool.git
cd Photool
```

## 3. Run the setup

**Windows:** double-click `setup.bat`, or run it from a terminal.
**macOS / Linux:** `./setup.sh`

It does three things:

1. Makes a Python environment **outside the project folder**, at
   `~/photool-env/venv` (`%USERPROFILE%\photool-env\venv` on Windows), and
   installs the Python packages into it. Outside, because a sandboxed coding tool
   that has the project folder open confines programs that run from inside it,
   and the server then can't write its catalog.
2. Installs the web page's packages (`npm install` in `frontend`).
3. Copies `backend/config.example.yaml` to `backend/config.yaml`, if you don't have
   one yet.

## 4. Point it at your photos

Edit `backend/config.yaml`. The one setting you must change is where your library is:

```yaml
library_root: "D:/Photos"                          # Windows
# library_root: "/Users/you/Pictures/Photos"       # macOS, in your home folder
# library_root: "/Volumes/PhotoDrive/Photos"       # macOS, on an external drive
folder_pattern: "{year}/{year}-{month}-{day}"
```

- **`library_root`** is the folder that holds your photographs, the one with
  the year folders in it. Photool reads the folders under it as they are. It
  only adds sidecar files (`.xmp`) beside your photos, and it never changes a
  photograph. Use forward slashes, on Windows too.
  - **On a Mac**, an external drive is under `/Volumes/`, with the name Finder
    shows. To get a folder's exact path, drag it from Finder into a Terminal
    window.
  - Point it at ordinary folders of files, not at the Photos app's
    `Photos Library.photoslibrary`.
- **`folder_pattern`** is how imports from a memory card are filed. The default,
  `2025/2025-05-10`, matches Lightroom's date folders.
- **Memory cards are found on their own.** Photool looks for the `DCIM` folder
  every camera writes, on every drive letter on Windows and every volume under
  `/Volumes` on a Mac. So it doesn't matter what the card is called: a Canon
  names it `EOS_DIGITAL`, a Nikon usually its model, and many others `NO NAME`
  or `Untitled`. Only a card that appears somewhere unusual needs
  `sd_mount_paths`.

Photool keeps its catalog (ratings, edits, faces) and thumbnail cache in
`<library_root>/.photomanager/`. To keep them on another drive, set `db_path` and
`thumbnail_cache`. Every other setting is explained in `config.example.yaml`.

`config.yaml` is yours: git ignores it, so updating Photool never overwrites it.

## 5. Start it

**Windows:** `serve.bat`. **macOS / Linux:** `./serve.sh`.

The first start builds the web page, which takes a minute. Then open
**http://localhost:8000**. Stop it with Ctrl+C or by closing the window.

The first time you open a folder, Photool makes thumbnails and measures each photo
in the background, so a big library fills in over a while. You can use it
straight away.

### Starting at login

One command sets Photool to start in the background every time you log in, with
no window:

| | Windows | macOS |
|---|---|---|
| Start at every login | `startup.bat install` | `./startup.sh install` |
| Stop starting at login | `startup.bat remove` | `./startup.sh remove` |
| Is it set up? | `startup.bat` | `./startup.sh` |
| Stop the one running now | `stop.bat` | `./stop.sh` |

- **Windows:** the server gets a **Photool icon in the system tray**, with Open,
  Restart, Open log and Quit. It's listed under Task Manager → Startup apps,
  where you can also turn it off. No administrator rights are needed. If you
  made a Startup-folder shortcut by hand before, `startup.bat install` points it
  out: delete it, or Photool is started twice. The second one leaves at once, so
  this is harmless, just untidy.
- **macOS:** `install` starts it straight away as well. There's no menu-bar icon
  yet: open http://localhost:8000 in the browser (bookmark it). `./stop.sh` stops
  it until the next login.

The server's log is `serve.log` beside the catalog (`<library_root>/.photomanager/`).
On a Mac, anything that goes wrong before that log opens is in
`~/Library/Logs/Photool/launchd.log`.

If you move the project folder or remake the venv, run `install` again: the
login item names both.

## 6. Optional features and their downloads

These download a model the first time you use them, into `models/` in the project
folder. Each download is checked against a known hash.

| Feature | Where | Download |
|---|---|---|
| People (face grouping) | People tab | 290 MB (ArcFace), or 15 MB with `face_model: arcface_lite` |
| Smart search ("kids at the beach") | Search, *Describe* | 305 MB |
| Noise reduction | Develop, Detail | 22 MB |
| Subject, sky and people masks | Develop, Masks | 4.6 MB and 15 MB |
| Parts of people (hair, skin, eyes…) | Develop, Masks, People | 450 MB |
| Object masks (SAM 2.1; SAM 3 if chosen) | Develop, Masks | 330 MB; 1.8 GB |
| Place names | Library, a folder's About | ~10 MB (GeoNames town list) |

**Graphics cards.** Noise reduction and object masks use a graphics card.
On Windows they use any card through DirectML: NVIDIA, AMD or Intel. On a Mac
noise reduction uses Core ML, the Apple Silicon GPU and Neural Engine; object masks
run on the CPU there, which was faster than Core ML for them. Without a card they
run on the CPU, more slowly: SAM 2.1 takes about 2 s for the first click on a view
(a fifth of a second with a card), then a few hundredths for each click; SAM 3
takes 20 to 30 s for the first click or word on a view, then about 2 s a word,
so on the CPU it is for patient use. Everything else runs on the CPU and needs no
graphics card. SAM 3 holds about 6.5 GB of graphics memory while it is in use, and
lets go of it three minutes after the last click or word.

## 7. Other computers and phones

Photool can serve the album to the other devices in your home, but only the
ones you pair: see [Other computers](guide.md#other-computers) in the guide. To
reach it from outside, [Tailscale](https://tailscale.com) is the safest way.

## 8. An AI agent (optional)

The server also offers the library as [MCP](https://modelcontextprotocol.io)
tools at `http://127.0.0.1:8000/mcp`. An agent such as Claude Code can then
search, cull, develop and export through the same catalog. See
[Driving it from an agent](guide.md#driving-it-from-an-agent-mcp).

- **Claude Code** finds the server through `.mcp.json` when it's started in the
  project folder, and asks once whether to use it. In Claude Code, `/mcp` shows
  whether it's connected. Photool must already be running.
- **Anywhere else:** `claude mcp add --transport http photool http://127.0.0.1:8000/mcp`
- The address is `127.0.0.1` rather than `localhost` on purpose. On a Mac,
  `localhost` can resolve to IPv6 first, where Photool doesn't listen.

To have an agent cull the way you do, copy `calibration/culling.example.md` to
`<library_root>/.photomanager/culling.md` and grow it as you calibrate. It's the
file `cull.guide` hands to the agent.

## Updating

```sh
git pull
```

Then start it as usual. The server sees that the page's source is newer than the
built page and rebuilds it. If a new Python package was added, `serve.bat` /
`serve.sh` installs it.

## Moving to a new computer

Copy, or keep on the same drive:

- your photographs, with `.photomanager/` inside the library (the catalog, edits,
  faces, thumbnails). Edits are also in each photo's `.xmp` sidecar.
- `backend/config.yaml`, with `library_root` changed if the drive letter or path
  is different.
- `photool-profiles/` (if you made camera profiles with `tools.camera_profile`)
  and `models/` (or let them download again).

Then do steps 1–3 and 5 on the new computer.

## Development

- `dev.bat` / `./dev.sh` runs the backend with auto-reload on port 8000 and the
  page's development server on **http://localhost:5173**, which reloads as you
  edit. `stop.bat` / `./stop.sh` stops both.
- Tests: `~/photool-env/venv/bin/python -m pytest` (Windows:
  `%USERPROFILE%\photool-env\venv\Scripts\python.exe -m pytest`), from the
  project folder. Install `pytest` and `httpx` into the venv first. A few tests
  need real RAW files: set `PHOTOOL_SAMPLES` to a folder of your own frames, named
  as `tests/samples.py` lists. Without it those tests are skipped.

## Troubleshooting

- **"Storage is read-only" / a red banner:** the catalog's drive refused a write.
  Photool keeps browsing but saves nothing. Fix the drive, or move `db_path` and
  `thumbnail_cache` to a drive that accepts writes.
- **ExifTool isn't found** (an amber bar at the top of the page, an empty info
  panel, few camera settings): install it and restart Photool. On a Mac,
  `brew install exiftool`; `which exiftool` should then print a path. Photool
  also looks in Homebrew's folders (`/opt/homebrew/bin`, `/usr/local/bin`), so it
  doesn't matter if your shell's PATH lacks them.
- **"Camera look: none for this file":** the look is learned from the JPEG the
  camera made, embedded in the RAW or saved beside it. A JPEG-only photo has no
  look to learn, because it already *is* the camera's picture. A RAW whose camera
  embedded only a small preview may not have one either.
- **Noise reduction seems to do little:** it's conditioned on the photo's ISO,
  so on a low-ISO frame, with little noise to remove, the change is small by
  design. Try a high-ISO frame. The log says where the model runs
  (`Noise reduction model loaded on CoreMLExecutionProvider`, or `DmlExecutionProvider`
  on Windows).
- **A RAW shows no thumbnail:** install ExifTool and restart.
- **Port 8000 is taken:** set `port: 8010` (any free port) in `config.yaml`.
- **Windows asks about the firewall:** allow Python on **private** networks only,
  and only if you want other devices to reach it.
