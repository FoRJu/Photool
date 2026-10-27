# Culling calibration

How to teach an agent the way you cull, and to test it against a folder you have
already culled before letting it propose on folders you have not. The agent works
blind (it doesn't see your marks), proposes for every frame, `cull.evaluate`
compares its choices with yours, and the run is then dismissed, so the grid is left
as it was.

**This file is a template.** Your own copy lives with your library, at
`<library>/.photomanager/culling.md` (beside `catalog.db`), and `cull.guide` reads
that one when it is there. Copy this file there, fill in the folders you have culled,
and update "What we know about how you cull" after each run. It describes your
taste and the people in your photographs, so it stays out of the repository.

Do one **scene** at a time (20–60 frames). A whole 1,000-frame folder costs a lot of
contact sheets, and the disagreements are easier to read on one scene.

## Folders to calibrate against

List folders you have fully culled, with what kind of shoot each one is. For example:

- `2025/2025-05-10`: **event**, a concert, 12 scenes (scenes 2 and 5 done)
- `2025/2025-06-21`: **family**, a birthday party

Not every scene in a folder is always culled. Before a run, check that the scene has
at least some stars, flags or rejects, as a yes or no only, without looking at which
frames have them. A scene with no marks can't be calibrated against.

Choose a different scene or folder from earlier runs, and vary the kind of scene: a
performance, portraits, a wide event, a family day out.

## The prompt

Paste this into Claude Code (or another MCP client) with the Photool MCP server
connected. Fill in the `<...>` parts, and change the third line to say how you mark
your keepers.

```text
Run a blind culling calibration on <folder>, scene <N> only.

cull.measure the folder, then cull.signals with blind: true and keep the frames of scene <N>.
Look at them with render.contact_sheet (marks: false), one sheet per burst or run of similar frames. Use render.loupe only where focus is in doubt.
Cull it the way I would: I keep frames with stars (1–5) and leave the rest unrated. Propose for every frame in the scene: a rating of 1–5 for keepers, rating: 0 for frames you wouldn't keep, flag: -1 only for clear failures. Give a short reason for each. Use run: "calib-<MMDD>-s<N>" and agent set to your model name.
Run cull.evaluate with run: "calib-<MMDD>-s<N>" and summarise the agreement, precision, recall and the patterns in the disagreements.
Then cull.dismiss the run.

Read cull.guide first and use what earlier runs found.
Do not use photo.rate or cull.accept.
```

To test whether the agent has learned anything, leave out the line "Read
cull.guide first" on one run, and compare that run's numbers with a run that has it.

`cull.guide` returns the section "What we know about how you cull" below, so an agent
that reaches Photool only through the MCP reads the same lessons. With `full: true`
it returns this whole file.

## Steps, and what to watch for

1. **`cull.measure <folder>`**. On a large folder with faces, this can take longer
   than an MCP client's idle timeout, and the call then reports *failed*. The
   measuring still finishes on the server. Check with `cull.signals`: the top-level
   `measured` and no frame with `measured: false` mean it is done. Don't run it again.
2. **`cull.signals` with `blind: true`**. The answer for a whole folder can be a few
   hundred kB, too big to read inline. Save it to a file, parse it with a short
   script, print the `scenes` summary, and keep the frames whose `scene` is N.
   Signals are numbered with the default `scene_gap_seconds: 120`. Keep that value,
   or the scene numbers change.
3. **Contact sheets with `marks: false`**: one per burst or run of similar frames,
   8–15 frames a sheet, `cell` 448–512. Without `marks: false` the sheet prints
   your stars, and the run isn't blind any more.
4. **`render.loupe`** only on finalists where focus is in doubt (a low
   `sharpness_rank`, or a soft-looking face). Aim the patch at a face: `u` and `v`
   are fractions of the frame, so work them out from where the face sits in the
   contact-sheet cell, use `size` 600–800, and check the patch shows skin or eyes.
   A patch that lands on the sky doesn't tell you anything about focus.
5. **`cull.propose`**: every frame in the scene, one call, with `run` and `agent`
   set. `rating: 0` for a frame you wouldn't keep, so it still counts in the
   comparison.
6. **`cull.evaluate` with the run**. Then write up the results (see below), and add
   a row to the log.
7. **`cull.dismiss` the run**. Proposals that are pending or dismissed are both
   compared, so dismissing doesn't lose anything. It only clears the violet ✦ chips
   from the grid.

Never use `photo.rate`, `cull.accept` or `bursts.reject_duplicates` during a
calibration. They change your own marks, and the folder stops being a reference.

## Reading the result

| Number | Meaning | Where to aim |
|---|---|---|
| keep agreement | frames where the agent and you agree on kept or not kept | above 0.8 |
| precision | of the agent's keeps, how many you kept. Low means it keeps too much | above 0.6 |
| recall | of your keeps, how many the agent found. Low means it misses things you want | above 0.7 |
| rating exact / within one | how close the stars are on all compared frames, 0 counted too | |
| flag agreement | whether its rejects match yours | |

**These numbers compare exact frames, and that is too strict.** A choice between
near-identical twins is close to arbitrary: if the agent keeps frame 0150 and you
kept 0151 of the same smile, it counts as one false keep and one miss. So
`cull.evaluate` also counts **by moment** (`keep.by_moment`): a keep matches when the
other side kept a frame within 2 frame numbers and 3 seconds of it. Read the
moment-level numbers alongside the frame-level ones.

Low precision with fair recall means the agent is too generous: tell it to keep
fewer frames. Low recall means it judges by the wrong things: read which of your
keeps it dropped, and what those frames have in common.

In the write-up, cover:

- how many frames you kept and how many the agent kept, and the stars each used
- what kind of frame you kept that the agent didn't, and the reverse (wide or close,
  people or context, moment or technique)
- whether your choice within a burst matches the burst keeper, sharpness or eyes
  signals
- anything in the signals that turned out to mislead

## What we know about how you cull

Update this section after each run. An agent running a calibration reads it first.
Nothing has been learned yet. Until it has, the agent should cull conservatively
and say what it is unsure about.

- **What the stars mean.** (What is 1★, and what earns more?)
- **Picks and rejects.** (Do you use them, or leave failures unrated?)
- **Event shoots** (the folder's kind "school" in its About: an event culled to
  share). (How much do you keep? Group shots or candids? What never gets through?)
- **Family shoots** (kind "family"). (How much do you keep? What is the point of
  the day?)

What the signals are known to get wrong, whoever is culling:

- `sharpness_rank` and `soft` compare a frame only with its peers (same camera,
  within 30 s, similar focal length). A very dark frame can still read as soft.
  Loupe a frame before dropping it for being soft.
- `eyes_closed` judges only frontal faces whose eyes can be seen. Sunglasses and
  mirrored lenses can still fool it. Treat it as a hint: look before dropping a
  burst twin for it.
- **Within a burst, the suggested keeper is only a suggestion.** Choosing the right
  moment matters; choosing the right twin rarely does.
- **Kind of shoot.** `cull.signals` reports the folder's `kind` when it is set in
  the folder's About. Cull by that kind's rules above.

## Log

Precision and recall are given by frame, then by moment in brackets.

| Date | Folder, scene | Kind | Frames | You kept | Agent kept | Agreement | Precision | Recall | Agent | Main lesson |
|---|---|---|---|---|---|---|---|---|---|---|
