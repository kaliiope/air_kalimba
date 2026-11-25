"""
song.py - Song / practice mode for Air Kalimba.

Songs are plain text files in kalimba number tabs (see songs/README.txt):

    # Twinkle Twinkle Little Star
    1 1 5 5 6 6 5
    4 4 3 3 2 2 1

  1-7         = do re mi fa so la ti (C D E F G A B), middle octave (C4-B4)
  1' or 1*    = one octave up  (one dot over the number on a real kalimba)
  1'' or 1**  = two octaves up (two dots)
  C5, G4 ...  = note names also work
  - | ,       = ignored (use them for spacing / bars)

The next tine to play glows blue. Hitting it advances the song; hitting a
different tine still plays but counts as a miss. Score shown at the end.
"""
import re

import cv2

from synth import KALIMBA_17

_NUM_TO_LETTER = {"1": "C", "2": "D", "3": "E", "4": "F", "5": "G", "6": "A", "7": "B"}
_DOT_CHARS = "'*°•·^"


def parse_token(tok):
    """'5' -> 'G4', "1'" -> 'C5', '3**' -> 'E6', 'D5' -> 'D5'. None if not a note."""
    tok = tok.strip()
    if not tok or tok in "-|,":
        return None
    m = re.fullmatch(r"([A-Ga-g])([0-9])", tok)
    if m:
        return m.group(1).upper() + m.group(2)
    m = re.fullmatch(r"([1-7])([" + re.escape(_DOT_CHARS) + r"]*)", tok)
    if m:
        return _NUM_TO_LETTER[m.group(1)] + str(4 + len(m.group(2)))
    raise ValueError(f"Can't read tab '{tok}'")


def load_song(path):
    title, notes, skipped = None, [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                title = title or line.lstrip("# ").strip()
                continue
            for tok in re.split(r"[\s|,]+", line):
                note = parse_token(tok)
                if note is None:
                    continue
                if note in KALIMBA_17:
                    notes.append(note)
                else:
                    skipped.append(note)
    if skipped:
        print(f"Warning: notes not on a 17-key C kalimba were skipped: {sorted(set(skipped))}")
    if not notes:
        raise ValueError(f"No playable notes in {path}")
    return title or path, notes


class SongFollower:
    def __init__(self, path):
        self.title, self.notes = load_song(path)
        self.reset()

    def reset(self):
        self.pos = 0
        self.correct = 0
        self.wrong = 0

    @property
    def done(self):
        return self.pos >= len(self.notes)

    @property
    def target_index(self):
        return None if self.done else KALIMBA_17.index(self.notes[self.pos])

    def on_hit(self, tine_index):
        """Call for every played tine. Returns True if it was the right one."""
        if self.done:
            return False
        if tine_index == self.target_index:
            self.correct += 1
            self.pos += 1
            return True
        self.wrong += 1
        return False

    def accuracy(self):
        total = self.correct + self.wrong
        return 100.0 * self.correct / total if total else 0.0

    # ------------------------------------------------------------------
    def draw(self, frame, tines):
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        # highlight the target tine
        if not self.done:
            t = tines[self.target_index]
            cv2.rectangle(frame, (t.x0 - 3, t.y_top), (t.x1 + 3, t.y_tip + 3), (255, 160, 0), 3)
            cx = (t.x0 + t.x1) // 2
            cv2.arrowedLine(frame, (cx, t.y_tip + 60), (cx, t.y_tip + 26), (255, 160, 0), 3, tipLength=0.4)

        # tab strip: next 8 notes
        y = h - 60
        cv2.rectangle(frame, (0, y - 30), (w, y + 14), (30, 30, 30), -1)
        cv2.putText(frame, f"{self.title}   {self.pos}/{len(self.notes)}", (10, y - 12),
                    font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        x = 330
        for k, note in enumerate(self.notes[self.pos:self.pos + 8]):
            num = {v: k2 for k2, v in _NUM_TO_LETTER.items()}[note[0]]
            label = num + "'" * (int(note[1]) - 4)
            col = (255, 160, 0) if k == 0 else (220, 220, 220)
            cv2.putText(frame, label, (x, y + 4), font, 0.8 if k == 0 else 0.6, col, 2, cv2.LINE_AA)
            x += 60

        if self.done:
            msg = f"Song complete!  {self.correct} correct, {self.wrong} wrong  ({self.accuracy():.0f}%)  - press n to restart"
            (tw, _), _ = cv2.getTextSize(msg, font, 0.7, 2)
            cv2.rectangle(frame, (w // 2 - tw // 2 - 12, h // 2 - 30), (w // 2 + tw // 2 + 12, h // 2 + 14), (0, 0, 0), -1)
            cv2.putText(frame, msg, (w // 2 - tw // 2, h // 2), font, 0.7, (80, 255, 80), 2, cv2.LINE_AA)
