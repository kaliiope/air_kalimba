# Air Kalimba 🎵

## Two modes

| | `air_kalimba_hands.py` (recommended) | `air_kalimba.py` (sticker mode) |
|---|---|---|
| Needs | Just your hands | Coloured stickers + HSV calibration |
| Tracking | MediaPipe hand landmarks (21 pts/hand) | HSV colour segmentation + contours |
| Play by | Hover a fingertip over a tine, push that finger **down** | Flick a thumb down across a tine's orange line |
| Motion analysis | Fingertip velocity relative to palm, normalised by hand size | Sticker centroid velocity in px/s |

**Hands mode:** `python air_kalimba_hands.py`. The first run downloads an ~8 MB model.
Options: `--fingers thumb` (thumbs only, kalimba-style), `--von 2.0` (more sensitive), `--mute`.
Keys: `q` quit · `+`/`-` tap threshold · `l` toggle hand skeleton · `r` reset log.
Each fingertip dot turns orange as it moves down fast, which shows you how close you are to the threshold.

You can compare both modes in the report (accuracy, lighting robustness, latency). That gives you material for the Analysis section.

---

## Sticker mode (original)

A virtual 17-tine kalimba you play by flicking your thumbs in front of a webcam.
Put a **green sticker on your left thumbnail** and a **pink/magenta sticker on your right**.
Each tine has an orange "pluck line" at its tip. Flick your thumb **down across** that line, fast, to play the note. A harder flick plays a louder note.

## Setup (Windows, Git Bash)

Use Python 3.11. Wheels for OpenCV and pygame may not exist yet for 3.14.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1          # PowerShell  (Git Bash: source .venv/Scripts/activate)
python -m pip install -r requirements.txt
```

## Run

```bash
python synth.py                      # 1. check audio: plays a scale
python calibrate_hsv.py --name left  # 2. tune green sticker, press s to save
python calibrate_hsv.py --name right # 3. tune pink sticker, press s to save
python air_kalimba.py                # 4. play!
```

Options: `--cam 1` (other camera), `--vmin 400` (easier plucks), `--mute`, `--log run1.csv`.
Keys while playing: `q` quit · `m` show mask · `+`/`-` flick threshold · `r` reset log.

## How it works (map to the CV concepts / rubric)

| Stage | CV concept | Where |
|---|---|---|
| Mirror + blur | Image pre-processing (Gaussian smoothing) | `main()` |
| BGR → HSV + `inRange` | Colour spaces, colour-based segmentation | `StickerTracker.segment` |
| Open / close | Morphological operations (noise removal, hole filling) | `StickerTracker.segment` |
| Largest contour + moments | Contour analysis, image moments, centroid | `StickerTracker.locate` |
| EMA smoothing over frames | Temporal filtering / tracking | `StickerTracker.update` |
| Velocity from position history | **Motion analysis** (finite-difference velocity, px/s) | `StickerTracker.velocity` |
| Downward line crossing above a speed threshold | **Motion-based event (flick) detection** | `PluckDetector.check` |
| Speed → loudness | Motion features mapped to output | `PluckDetector.check` |

The pluck uses a line crossing, not "thumb is inside the box". A fast flick that jumps past the line between two frames still counts. The thumb also has to come back above the line before it can pluck the same tine again, which prevents double-triggers.

The sound is synthesised, so you don't need any sample files. The tone uses the inharmonic overtones of a clamped metal bar (1 : 6.27 : 17.55) with exponential decay. See `synth.py`.

## Results / testing ideas (15 + 15 pts)

Every pluck is logged to `events.csv` (time, thumb, tine, note, flick speed, volume, processing ms). Suggested experiments:

1. **Accuracy test.** Each member plays a fixed sequence, e.g. 20 target tines. Count correct hits, wrong-tine hits and misses. Report precision and recall.
2. **Threshold sweep.** Run the test at `--vmin 300 / 500 / 800`. Lower thresholds give more false triggers and higher ones give more misses. Plot both.
3. **Lighting test.** Repeat under room light, daylight and dim light. Calibrated HSV ranges vs the defaults.
4. **Latency.** Read `frame_ms` from the log and FPS from the HUD. Discuss the trade-off between EMA smoothing (`ema_alpha`) and lag.
5. **Failure analysis.** Look at motion blur at high flick speeds, skin or background with the same colour as a sticker, and the sticker turning edge-on to the camera.

## Possible extensions (for individual contribution, 5 pts)

Ideas for splitting the work three ways:

- **Member A: tracking robustness.** Kalman filter instead of EMA. Optional Lucas–Kanade optical flow (`cv2.calcOpticalFlowPyrLK`) around the sticker as a second velocity estimate. Back-projection with a hue histogram.
- **Member B: interaction / motion.** Tune the pluck model (acceleration spike, sustained notes). Glissando by sweeping across tines. Chord detection when both thumbs pluck together. Song-teaching mode that highlights the next tine.
- **Member C: evaluation + docs.** Testing harness (scripted target sequences that score themselves from `events.csv`), charts, demo video, presentation.

## Files

- `air_kalimba.py`: main app (tracking, flick detection, drawing, logging)
- `synth.py`: kalimba tone synthesis and playback
- `calibrate_hsv.py`: HSV slider tool, saves `hsv_ranges.json`
