"""
synth.py - Kalimba tone synthesis (no sample files needed).

A kalimba tine behaves like a clamped-free metal bar (cantilever). Its
overtones are NOT harmonic: the mode ratios are roughly 1 : 6.27 : 17.55.
That inharmonic "ping" plus a fast attack and exponential decay is what
makes it sound like a kalimba rather than a sine beep.
"""
import numpy as np

SAMPLE_RATE = 44100

# Standard 17-key kalimba in C major, LEFT to RIGHT as the player sees it.
# Lowest note (C4) is the longest tine in the middle; notes alternate outward.
KALIMBA_17 = [
    "D6", "B5", "G5", "E5", "C5", "A4", "F4", "D4",
    "C4",
    "E4", "G4", "B4", "D5", "F5", "A5", "C6", "E6",
]

# Number printed on real kalimba tines (1 = do, 2 = re ...), dots = octave
NOTE_NUMBERS = {"C": "1", "D": "2", "E": "3", "F": "4", "G": "5", "A": "6", "B": "7"}

_SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def note_to_freq(note: str) -> float:
    """'A4' -> 440.0 Hz (equal temperament)."""
    name, octave = note[0], int(note[1:])
    midi = 12 * (octave + 1) + _SEMITONE[name]
    return 440.0 * 2 ** ((midi - 69) / 12)


def kalimba_wave(freq: float, duration: float = 2.0, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Return a mono float32 waveform in [-1, 1]."""
    t = np.arange(int(sr * duration)) / sr
    # (ratio to fundamental, amplitude, decay rate)  - higher modes die faster
    partials = [(1.0, 1.00, 3.0), (6.27, 0.25, 12.0), (17.55, 0.08, 30.0)]
    wave = np.zeros_like(t)
    for ratio, amp, decay in partials:
        f = freq * ratio
        if f < sr / 2:                      # skip anything above Nyquist
            wave += amp * np.exp(-decay * t) * np.sin(2 * np.pi * f * t)
    # small "click" of the thumb hitting metal
    click_len = int(0.004 * sr)
    wave[:click_len] += np.random.uniform(-0.3, 0.3, click_len) * np.linspace(1, 0, click_len)
    # 3 ms attack ramp to avoid a pop
    attack = int(0.003 * sr)
    wave[:attack] *= np.linspace(0, 1, attack)
    wave /= np.max(np.abs(wave))
    return wave.astype(np.float32)


def to_int16_stereo(wave: np.ndarray, gain: float = 0.8) -> np.ndarray:
    mono = (wave * gain * 32767).astype(np.int16)
    return np.column_stack([mono, mono])


class KalimbaSynth:
    """Pre-renders all 17 tines once, then plays them with pygame.mixer."""

    def __init__(self, notes=KALIMBA_17):
        import pygame
        pygame.mixer.pre_init(SAMPLE_RATE, -16, 2, 256)   # small buffer = low latency
        pygame.mixer.init()
        pygame.mixer.set_num_channels(32)                # let notes ring over each other
        self._pg = pygame
        self.notes = list(notes)
        self.sounds = [pygame.sndarray.make_sound(np.ascontiguousarray(
            to_int16_stereo(kalimba_wave(note_to_freq(n))))) for n in self.notes]

    def play(self, index: int, velocity: float = 1.0):
        """velocity in [0, 1] -> loudness (faster flick = louder note)."""
        ch = self._pg.mixer.find_channel(True)
        ch.set_volume(float(np.clip(velocity, 0.15, 1.0)))
        ch.play(self.sounds[index])


if __name__ == "__main__":
    # Quick audio test: play a C-major run up the kalimba
    import time
    synth = KalimbaSynth()
    order = sorted(range(17), key=lambda i: note_to_freq(KALIMBA_17[i]))
    for i in order:
        print(KALIMBA_17[i])
        synth.play(i, 0.8)
        time.sleep(0.2)
    time.sleep(2)
