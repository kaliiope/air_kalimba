"""
calibrate_hsv.py - find the HSV range of your thumb stickers.

Usage:
    python calibrate_hsv.py --cam 0

Hold the sticker in front of the camera and drag the sliders until ONLY the
sticker is white in the mask window. Press 's' to print/save the range,
'q' to quit. Do this for each sticker colour, under the lighting you'll
present in (room lights change everything!).
"""
import argparse
import json
import cv2
import numpy as np


def nothing(_):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--name", default="sticker", help="label saved in hsv_ranges.json")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.cam)
    cv2.namedWindow("controls")
    for name, maxv, init in [("H min", 179, 0), ("H max", 179, 179),
                             ("S min", 255, 100), ("S max", 255, 255),
                             ("V min", 255, 80), ("V max", 255, 255)]:
        cv2.createTrackbar(name, "controls", init, maxv, nothing)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (7, 7), 0), cv2.COLOR_BGR2HSV)
        g = lambda n: cv2.getTrackbarPos(n, "controls")
        lo = np.array([g("H min"), g("S min"), g("V min")])
        hi = np.array([g("H max"), g("S max"), g("V max")])
        mask = cv2.inRange(hsv, lo, hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        cv2.imshow("frame", frame)
        cv2.imshow("mask", mask)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            rng = {"lower": lo.tolist(), "upper": hi.tolist()}
            print(args.name, rng)
            try:
                with open("hsv_ranges.json") as f:
                    data = json.load(f)
            except FileNotFoundError:
                data = {}
            data[args.name] = rng
            with open("hsv_ranges.json", "w") as f:
                json.dump(data, f, indent=2)
            print("saved to hsv_ranges.json")
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
