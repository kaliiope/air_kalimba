"""
air_kalimba.py - Air Kalimba (Computer Vision group project starter)

Play a virtual 17-tine kalimba by flicking your thumbs (with coloured
stickers on the thumbnails) DOWN across the tips of on-screen tines.

Pipeline per frame
  1. Capture + mirror            (webcam frame, flipped so it feels like a mirror)
  2. Colour segmentation         (BGR -> HSV, inRange, morphological open/close)
  3. Blob localisation           (largest contour, image moments -> centroid)
  4. Temporal tracking           (EMA-smoothed position history per thumb)
  5. Motion analysis             (velocity from finite differences; flick = a
                                  downward crossing of a tine tip faster than
                                  a speed threshold)
  6. Output                      (note played, loudness ~ flick speed, visuals,
                                  CSV event log for the Results/Testing section)

Keys:  q quit | m toggle mask view | +/- change flick speed threshold | r reset log | n restart song
"""
import argparse
import csv
import json
import os
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from synth import KALIMBA_17, NOTE_NUMBERS

# --------------------------------------------------------------------------
# Default HSV ranges (OpenCV H is 0-179). Run calibrate_hsv.py to tune these
# for YOUR stickers and YOUR room - defaults are only a starting point.
# --------------------------------------------------------------------------
DEFAULT_RANGES = {
    "left":  {"lower": [40, 80, 70],   "upper": [85, 255, 255]},   # green sticker
    "right": {"lower": [140, 80, 70],  "upper": [175, 255, 255]},  # pink/magenta sticker
}
COLOURS_BGR = {"left": (80, 220, 80), "right": (200, 80, 230)}


# ==========================================================================
# 1-3. Segmentation + localisation
# ==========================================================================
class StickerTracker:
    """Finds one coloured sticker per frame and keeps a short motion history."""

    def __init__(self, name, lower, upper, min_area=60, ema_alpha=0.6, history=8):
        self.name = name
        self.lower = np.array(lower, dtype=np.uint8)
        self.upper = np.array(upper, dtype=np.uint8)
        self.min_area = min_area
        self.alpha = ema_alpha            # 1.0 = no smoothing, lower = smoother but laggier
        self.history = deque(maxlen=history)   # (t, x, y) smoothed
        self.lost_frames = 0
        self.mask = None
        self.raw = None                   # last raw centroid (before smoothing)

    def segment(self, hsv):
        mask = cv2.inRange(hsv, self.lower, self.upper)
        k = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)    # remove speckle noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)   # fill holes in sticker
        self.mask = mask
        return mask

    def locate(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < self.min_area:
            return None
        m = cv2.moments(c)
        if m["m00"] == 0:
            return None
        return (m["m10"] / m["m00"], m["m01"] / m["m00"])

    def update(self, hsv, t):
        pos = self.locate(self.segment(hsv))
        self.raw = pos
        if pos is None:
            self.lost_frames += 1
            if self.lost_frames > 5:       # lost too long -> don't fake velocity
                self.history.clear()
            return None
        self.lost_frames = 0
        if self.history:
            _, px, py = self.history[-1]
            pos = (self.alpha * pos[0] + (1 - self.alpha) * px,
                   self.alpha * pos[1] + (1 - self.alpha) * py)
        self.history.append((t, pos[0], pos[1]))
        return pos

    # ----------------------------------------------------------------------
    # 5. Motion analysis
    # ----------------------------------------------------------------------
    def velocity(self, span=3):
        """(vx, vy) in pixels/second over the last `span` samples. +vy = downward."""
        if len(self.history) < 2:
            return 0.0, 0.0
        span = min(span, len(self.history) - 1)
        t1, x1, y1 = self.history[-1]
        t0, x0, y0 = self.history[-1 - span]
        dt = max(t1 - t0, 1e-3)
        return (x1 - x0) / dt, (y1 - y0) / dt

    def last_segment(self):
        """Previous and current smoothed positions, for crossing tests."""
        if len(self.history) < 2:
            return None
        return self.history[-2][1:], self.history[-1][1:]


# ==========================================================================
# Kalimba geometry
# ==========================================================================
@dataclass
class Tine:
    index: int
    note: str
    x0: int
    x1: int
    y_top: int
    y_tip: int        # crossing this line downward = pluck
    last_hit: float = -1.0
    glow: float = 0.0


def build_tines(w, h, notes=KALIMBA_17, margin_frac=0.06, gap_frac=0.18):
    n = len(notes)
    margin = int(w * margin_frac)
    slot = (w - 2 * margin) / n
    gap = slot * gap_frac
    centre = n // 2
    y_top = int(h * 0.08)
    shortest, longest = h * 0.28, h * 0.50       # centre tine longest, like a real kalimba
    tines = []
    for i, note in enumerate(notes):
        d = abs(i - centre) / centre               # 0 at centre, 1 at edge
        length = longest - (longest - shortest) * d
        x0 = int(margin + i * slot + gap / 2)
        x1 = int(margin + (i + 1) * slot - gap / 2)
        tines.append(Tine(i, note, x0, x1, y_top, int(y_top + length)))
    return tines


# ==========================================================================
# Pluck (flick) detection
# ==========================================================================
class PluckDetector:
    """
    A pluck fires when a thumb's path crosses a tine's tip line moving DOWN,
    inside that tine's column, faster than `v_min` px/s.

    Using a line *crossing* (rather than "is the thumb inside the box") means:
      - a fast flick that jumps past the line between two frames still counts
      - the thumb must come back up above the line before it can pluck again
    """

    def __init__(self, tines, v_min=500.0, v_max=2500.0, cooldown=0.12):
        self.tines = tines
        self.v_min = v_min
        self.v_max = v_max
        self.cooldown = cooldown

    def check(self, tracker, t):
        seg = tracker.last_segment()
        if seg is None:
            return []
        (x0, y0), (x1, y1) = seg
        _, vy = tracker.velocity()
        if vy < self.v_min:
            return []
        hits = []
        for tine in self.tines:
            if y0 < tine.y_tip <= y1:                       # crossed downward
                a = (tine.y_tip - y0) / max(y1 - y0, 1e-6)  # where along the segment
                xc = x0 + a * (x1 - x0)                     # x at the moment of crossing
                if tine.x0 <= xc <= tine.x1 and t - tine.last_hit > self.cooldown:
                    vel = (vy - self.v_min) / (self.v_max - self.v_min)
                    vel = float(np.clip(0.3 + 0.7 * vel, 0.3, 1.0))
                    tine.last_hit, tine.glow = t, 1.0
                    hits.append((tine, vy, vel))
        return hits


# ==========================================================================
# Drawing
# ==========================================================================
def draw_kalimba(frame, tines):
    overlay = frame.copy()
    h, w = frame.shape[:2]
    # bridge bar
    cv2.rectangle(overlay, (tines[0].x0 - 10, tines[0].y_top - 14),
                  (tines[-1].x1 + 10, tines[0].y_top), (40, 40, 40), -1)
    for tine in tines:
        g = tine.glow
        base = np.array([200, 200, 210])
        hot = np.array([60, 220, 255])
        col = tuple(int(c) for c in (base * (1 - g) + hot * g))
        cv2.rectangle(overlay, (tine.x0, tine.y_top), (tine.x1, tine.y_tip), col, -1)
        # tip line = the "pluck line"
        cv2.line(overlay, (tine.x0, tine.y_tip), (tine.x1, tine.y_tip), (0, 140, 255), 2)
        tine.glow = max(0.0, tine.glow - 0.08)    # fade
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for tine in tines:
        draw_tab(frame, tine)


def draw_tab(frame, tine):
    """Kalimba tab label: number (1=do ... 7=ti) with octave dots above it,
    exactly like the engraving on a real 17-key kalimba. Note name under the tip."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    num = NOTE_NUMBERS[tine.note[0]]
    dots = int(tine.note[1:]) - 4              # C4-B4: none, C5-B5: one, C6+: two
    cx = (tine.x0 + tine.x1) // 2
    scale, thick = 0.75, 2
    (tw, th), _ = cv2.getTextSize(num, font, scale, thick)
    ty = tine.y_tip - 12
    cv2.putText(frame, num, (cx - tw // 2, ty), font, scale, (20, 20, 20), thick, cv2.LINE_AA)
    for k in range(dots):
        cv2.circle(frame, (cx, ty - th - 6 - k * 8), 3, (20, 20, 20), -1, cv2.LINE_AA)
    (nw, _), _ = cv2.getTextSize(tine.note, font, 0.42, 1)
    cv2.putText(frame, tine.note, (cx - nw // 2, tine.y_tip + 18), font, 0.42,
                (255, 255, 255), 1, cv2.LINE_AA)


def draw_tracker(frame, tr):
    if not tr.history:
        return
    col = COLOURS_BGR.get(tr.name, (255, 255, 255))
    pts = [(int(x), int(y)) for _, x, y in tr.history]
    for a, b in zip(pts, pts[1:]):
        cv2.line(frame, a, b, col, 2)
    x, y = pts[-1]
    cv2.circle(frame, (x, y), 10, col, 2)
    vx, vy = tr.velocity()
    cv2.arrowedLine(frame, (x, y), (int(x + vx * 0.05), int(y + vy * 0.05)), col, 2, tipLength=0.3)
    cv2.putText(frame, f"{tr.name} vy={vy:+.0f}px/s", (x + 14, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)


# ==========================================================================
# Main loop
# ==========================================================================
def load_ranges(path):
    ranges = {k: dict(v) for k, v in DEFAULT_RANGES.items()}
    if os.path.exists(path):
        with open(path) as f:
            ranges.update(json.load(f))
        print(f"Loaded HSV ranges from {path}")
    else:
        print("No hsv_ranges.json found - using defaults (run calibrate_hsv.py!)")
    return ranges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--vmin", type=float, default=500.0, help="min downward speed (px/s) for a pluck")
    ap.add_argument("--ranges", default="hsv_ranges.json")
    ap.add_argument("--log", default="events.csv")
    ap.add_argument("--song", help="song file for practice mode, e.g. songs/twinkle.txt")
    ap.add_argument("--mute", action="store_true", help="no audio (for testing)")
    args = ap.parse_args()

    synth = None
    if not args.mute:
        from synth import KalimbaSynth
        synth = KalimbaSynth()

    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Could not open camera. Try --cam 1")
    h, w = frame.shape[:2]

    ranges = load_ranges(args.ranges)
    trackers = [StickerTracker(name, ranges[name]["lower"], ranges[name]["upper"])
                for name in ("left", "right")]
    tines = build_tines(w, h)
    detector = PluckDetector(tines, v_min=args.vmin)
    song = None
    if args.song:
        from song import SongFollower
        song = SongFollower(args.song)
        print(f"Song mode: {song.title} ({len(song.notes)} notes)")

    log_f = open(args.log, "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["time_s", "thumb", "tine", "note", "vy_px_s", "volume", "frame_ms", "target", "correct"])

    show_mask = False
    t_start = time.perf_counter()
    fps_hist = deque(maxlen=30)
    prev = t_start
    n_plucks = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t_frame = time.perf_counter()
        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (7, 7), 0), cv2.COLOR_BGR2HSV)
        t = t_frame - t_start

        for tr in trackers:
            tr.update(hsv, t)
            for tine, vy, vol in detector.check(tr, t):
                if synth:
                    synth.play(tine.index, vol)
                n_plucks += 1
                proc_ms = (time.perf_counter() - t_frame) * 1000
                target = song.notes[song.pos] if song and not song.done else ""
                correct = song.on_hit(tine.index) if song else ""
                log.writerow([f"{t:.3f}", tr.name, tine.index, tine.note,
                              f"{vy:.0f}", f"{vol:.2f}", f"{proc_ms:.1f}", target, correct])

        # ---- visuals ----
        if show_mask:
            m = cv2.bitwise_or(trackers[0].mask, trackers[1].mask)
            frame = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
        draw_kalimba(frame, tines)
        if song:
            song.draw(frame, tines)
        for tr in trackers:
            draw_tracker(frame, tr)

        now = time.perf_counter()
        fps_hist.append(1.0 / max(now - prev, 1e-6))
        prev = now
        hud = f"FPS {np.mean(fps_hist):.0f} | flick threshold {detector.v_min:.0f}px/s | plucks {n_plucks}"
        cv2.putText(frame, hud, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("Air Kalimba", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("m"):
            show_mask = not show_mask
        elif key in (ord("+"), ord("=")):
            detector.v_min += 50
        elif key == ord("-"):
            detector.v_min = max(50, detector.v_min - 50)
        elif key == ord("r"):
            log_f.seek(0); log_f.truncate()
            log.writerow(["time_s", "thumb", "tine", "note", "vy_px_s", "volume", "frame_ms", "target", "correct"])
            n_plucks = 0
        elif key == ord("n") and song:
            song.reset()

    log_f.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"{n_plucks} plucks logged to {args.log}")


if __name__ == "__main__":
    main()
