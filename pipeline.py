"""Bounded audio segmentation; plain Python so lifecycle can be tested without Qt."""
from __future__ import annotations

from collections import deque
import math
import threading
import time

import numpy as np
from scipy.signal import resample_poly


class Segmenter:
    def __init__(self, on_segment, on_level=lambda _: None, max_seconds=4.0):
        self.on_segment = on_segment
        self.on_level = on_level
        self.max_seconds = max_seconds
        self.lock = threading.RLock()
        self.reset()

    def reset(self):
        with self.lock:
            self.blocks = []
            self.preroll = deque(maxlen=10)
            self.samples = 0
            self.silence = 0
            self.active = False
            self.rate = None
            self.last_voice_time = None
            self.overlap_seconds = 0.0

    def feed(self, audio: np.ndarray, rate: int):
        with self.lock:
            if self.rate not in (None, rate):
                self.reset()
            self.rate = rate
            audio = np.asarray(audio, dtype=np.float32)
            if not audio.size:
                return
            rms = float(np.sqrt(np.mean(audio ** 2)))
            self.on_level(min(1.0, rms * 12))
            speech_energy = rms > 0.003
            if not self.active:
                self.preroll.append(audio.copy())
                if not speech_energy:
                    return
                self.blocks = list(self.preroll)
                self.samples = sum(len(a) for a in self.blocks)
                self.active = True
                self.preroll.clear()
            else:
                self.blocks.append(audio.copy())
                self.samples += len(audio)
            if speech_energy:
                self.silence = 0
                self.last_voice_time = time.monotonic()
            else:
                self.silence += len(audio)
            timed_out = self.samples >= self.max_seconds * rate
            silence_end = self.silence >= 0.55 * rate
            if timed_out or silence_end:
                full = np.concatenate(self.blocks)
                stamp = self.last_voice_time or time.monotonic()
                overlap = self.overlap_seconds
                enough = self.samples - self.silence >= (0.45 if overlap else 0.30) * rate
                self.reset()
                if timed_out and not silence_end:
                    # Context around the cut is retained instead of cutting a word in half.
                    tail = full[-round(rate * 0.30):].copy()
                    self.blocks = [tail]
                    self.samples = len(tail)
                    self.active = True
                    self.rate = rate
                    self.overlap_seconds = 0.30
                    self.last_voice_time = stamp
                if enough:
                    divisor = math.gcd(rate, 16000)
                    pcm = resample_poly(full, 16000 // divisor, rate // divisor).astype(np.float32)
                    self.on_segment(pcm, stamp, overlap)
