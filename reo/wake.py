"""Always-on listening: wait for your name, then hand over what you said after it.

The microphone stays open, but nothing is transcribed until the room goes quiet
after someone speaks. Idle cost is one RMS calculation per 100ms block, and the
audio never leaves this machine.
"""
import collections

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

from . import config
from .text import normalize

BLOCK_SECONDS = 0.1
LEAD_IN_SECONDS = 0.5  # keep audio from just before the first loud block, or "Reo" gets clipped


class WakeListener:
    def __init__(self):
        print(f"Loading wake model ({config.WAKE_MODEL})...")
        self.model = WhisperModel(config.WAKE_MODEL, device="cpu", compute_type="int8")

    def _transcribe(self, frames) -> str:
        audio = np.concatenate(frames).flatten()
        segments, _ = self.model.transcribe(audio, language="en", beam_size=1, vad_filter=True)
        return " ".join(s.text for s in segments).strip()

    def _command_after_wake(self, text: str):
        """Return what was said after the wake word, "" if nothing, None if not addressed."""
        words = normalize(text).split()
        for i, word in enumerate(words):
            if word in config.WAKE_WORDS:
                return " ".join(words[i + 1:])
        return None

    def wait(self):
        """Block until addressed. Returns the rest of that sentence, possibly empty."""
        block = int(config.SAMPLE_RATE * BLOCK_SECONDS)
        silence_blocks = int(config.WAKE_SILENCE_SECONDS / BLOCK_SECONDS)
        max_blocks = int(config.WAKE_MAX_SECONDS / BLOCK_SECONDS)

        lead_in = collections.deque(maxlen=int(LEAD_IN_SECONDS / BLOCK_SECONDS))
        frames, speaking, silent = [], False, 0

        with sd.InputStream(samplerate=config.SAMPLE_RATE, channels=1,
                            dtype="float32", blocksize=block) as stream:
            while True:
                data, _ = stream.read(block)
                loud = float(np.sqrt(np.mean(data ** 2))) > config.SILENCE_THRESHOLD

                if not speaking:
                    lead_in.append(data.copy())
                    if loud:
                        speaking, silent = True, 0
                        frames = list(lead_in)
                    continue

                frames.append(data.copy())
                if loud:
                    silent = 0
                else:
                    silent += 1

                if silent >= silence_blocks or len(frames) >= max_blocks:
                    command = self._command_after_wake(self._transcribe(frames))
                    if command is not None:
                        return command
                    lead_in.clear()
                    frames, speaking, silent = [], False, 0
