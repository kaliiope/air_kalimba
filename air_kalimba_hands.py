"""
air_kalimba_hands.py - Air Kalimba, NO stickers, NO calibration.

Hover a fingertip over a tine and push that finger DOWN (like tapping a
piano key in the air). The tine under the fingertip plays.

Pipeline per frame
  1. Capture + mirror
  2. Hand landmark detection     (MediaPipe HandLandmarker, 21 points per hand)
  3. Fingertip tracking          (5 tips per hand, EMA smoothing)
  4. Motion analysis             (fingertip velocity RELATIVE to the palm,
                                  normalised by hand size -> "hand-lengths/s")
  5. Tap detection               (velocity spike above threshold, with
                                  hysteresis + cooldown so one tap = one note)
  6. Output                      (note, loudness ~ tap speed, visuals, CSV log)

Why relative-to-palm velocity?  If you move your whole hand, every finger
moves together and nothing should play. Only a finger bending down relative
to the rest of the hand counts as a tap.

Why normalise by hand size?  A hand close to the camera looks bigger and
moves more pixels for the same real motion. Dividing by the hand's own
length in pixels makes the threshold work at any distance.

Keys: q quit | +/- tap sensitivity | l toggle landmarks | r reset log | n restart song
"""
import argparse
import csv
import os
import time
import urllib.request
from collections import deque

import cv2
import numpy as np

from air_kalimba import build_tines, draw_kalimba
from synth import KALIMBA_17

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
MODEL_PATH = "hand_landmarker.task"

TIP_IDS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
WRIST, MIDDLE_MCP = 0, 9            # palm reference points
HAND_COLOURS = {"Left": (80, 220, 80), "Right": (200, 80, 230)}
HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
                    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand model (~8 MB, one time only)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


# ==========================================================================
# Fingertip motion tracking + tap detection
# ==========================================================================
class FingerTap:
    """
    Tracks one fingertip. A tap fires when the fingertip's DOWNWARD speed
    relative to the palm goes above `v_on` (hand-lengths per second).
    It must slow down / come back up below `v_off` before it can fire again.
    """

    def __init__(self, ema_alpha=0.6, history=6):
        self.alpha = ema_alpha
        self.hist = deque(maxlen=history)    # (t, tip_x, tip_y, rel_y_norm)
        self.armed = True
        self.last_tap = -1.0
        self.rel_vy = 0.0

    def update(self, t, tip, palm, hand_len):
        """tip/palm in pixels, hand_len = wrist->middle-knuckle distance in px."""
        rel_y = (tip[1] - palm[1]) / max(hand_len, 1.0)
        if self.hist:
            _, px, py, pr = self.hist[-1]
            a = self.alpha
            tip = (a * tip[0] + (1 - a) * px, a * tip[1] + (1 - a) * py)
            rel_y = a * rel_y + (1 - a) * pr
        self.hist.append((t, tip[0], tip[1], rel_y))
        if len(self.hist) >= 3:
            t0, *_, r0 = self.hist[-3]
            t1, *_, r1 = self.hist[-1]
            self.rel_vy = (r1 - r0) / max(t1 - t0, 1e-3)
        else:
            self.rel_vy = 0.0

    def check(self, t, v_on, v_off, cooldown=0.12):
        """Returns tap strength in [0,1] or None."""
        if not self.armed and self.rel_vy < v_off:
            self.armed = True                       # finger came back up / stopped
        if self.armed and self.rel_vy > v_on and t - self.last_tap > cooldown:
            self.armed = False
            self.last_tap = t
            return float(np.clip(0.3 + 0.7 * (self.rel_vy - v_on) / (2 * v_on), 0.3, 1.0))
        return None

    @property
    def pos(self):
        return (self.hist[-1][1], self.hist[-1][2]) if self.hist else None

    def reset(self):
        self.hist.clear()
        self.armed = True
        self.rel_vy = 0.0


def tine_under(x, y, tines):
    """Which tine column is the fingertip in? (anywhere below the bridge)"""
    for tine in tines:
        if tine.x0 - 4 <= x <= tine.x1 + 4 and y > tine.y_top:
            return tine
    return None


# ==========================================================================
# Main loop
# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--von", type=float, default=2.5,
                    help="tap threshold: downward speed in hand-lengths/s (lower = more sensitive)")
    ap.add_argument("--fingers", default="thumb,index,middle,ring,pinky",
                    help="which fingers can play, e.g. 'thumb' for kalimba-style, 'index,middle'")
    ap.add_argument("--log", default="events_hands.csv")
    ap.add_argument("--song", help="song file for practice mode, e.g. songs/twinkle.txt")
    ap.add_argument("--mute", action="store_true")
    args = ap.parse_args()

    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=ensure_model()),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    ))

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
    tines = build_tines(w, h)
    song = None
    if args.song:
        from song import SongFollower
        song = SongFollower(args.song)
        print(f"Song mode: {song.title} ({len(song.notes)} notes)")

    fingers = [f.strip() for f in args.fingers.split(",") if f.strip() in TIP_IDS]
    trackers = {(hand, f): FingerTap() for hand in ("Left", "Right") for f in fingers}
    v_on = args.von
    show_lm = True

    log_f = open(args.log, "w", newline="")
    log = csv.writer(log_f)
    header = ["time_s", "hand", "finger", "tine", "note", "rel_vy_hl_s", "volume", "frame_ms", "target", "correct"]
    log.writerow(header)

    t_start = time.perf_counter()
    last_ts = -1
    fps_hist, prev, n_taps = deque(maxlen=30), t_start, 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t_frame = time.perf_counter()
        t = t_frame - t_start
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts = max(int(t * 1000), last_ts + 1)       # MediaPipe needs increasing timestamps
        last_ts = ts
        result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

        seen = set()
        hits = []
        for lms, handed in zip(result.hand_landmarks, result.handedness):
            hand = handed[0].category_name                 # "Left" / "Right"
            pts = np.array([(p.x * w, p.y * h) for p in lms])
            palm = pts[MIDDLE_MCP]
            hand_len = np.linalg.norm(pts[MIDDLE_MCP] - pts[WRIST])

            if show_lm:
                col = HAND_COLOURS.get(hand, (255, 255, 255))
                for a, b in HAND_CONNECTIONS:
                    cv2.line(frame, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), col, 1)

            for f in fingers:
                key = (hand, f)
                if key not in trackers:
                    continue
                seen.add(key)
                tr = trackers[key]
                tr.update(t, tuple(pts[TIP_IDS[f]]), tuple(palm), hand_len)
                vol = tr.check(t, v_on, v_off=0.3 * v_on)
                x, y = tr.pos
                if vol is not None:
                    tine = tine_under(x, y, tines)
                    if tine is not None:
                        hits.append((hand, f, tine, tr.rel_vy, vol))
                # fingertip marker, turns orange when moving down fast
                hot = min(max(tr.rel_vy / v_on, 0), 1)
                cv2.circle(frame, (int(x), int(y)), 7,
                           (int(255 * (1 - hot)), int(255 - 115 * hot), 255), -1)

        for key, tr in trackers.items():          # hand left the frame -> reset
            if key not in seen:
                tr.reset()

        for hand, f, tine, vy, vol in hits:
            if synth:
                synth.play(tine.index, vol)
            tine.glow, tine.last_hit = 1.0, t
            n_taps += 1
            target = song.notes[song.pos] if song and not song.done else ""
            correct = song.on_hit(tine.index) if song else ""
            log.writerow([f"{t:.3f}", hand, f, tine.index, tine.note, f"{vy:.2f}",
                          f"{vol:.2f}", f"{(time.perf_counter() - t_frame) * 1000:.1f}", target, correct])

        draw_kalimba(frame, tines)
        if song:
            song.draw(frame, tines)
        now = time.perf_counter()
        fps_hist.append(1.0 / max(now - prev, 1e-6))
        prev = now
        cv2.putText(frame, f"FPS {np.mean(fps_hist):.0f} | tap threshold {v_on:.1f} hl/s | taps {n_taps}",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("Air Kalimba (hands)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key in (ord("+"), ord("=")):
            v_on += 0.25
        elif key == ord("-"):
            v_on = max(0.5, v_on - 0.25)
        elif key == ord("l"):
            show_lm = not show_lm
        elif key == ord("r"):
            log_f.seek(0); log_f.truncate(); log.writerow(header); n_taps = 0
        elif key == ord("n") and song:
            song.reset()

    log_f.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"{n_taps} taps logged to {args.log}")


if __name__ == "__main__":
    main()
