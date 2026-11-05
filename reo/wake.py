"""Always-on listening: wait for your name, then hand over what you said after it.

The microphone stays open, but nothing is transcribed until the room goes quiet
after someone speaks. Idle cost is one RMS calculation per 100ms block, and the
audio never leaves this machine.
"""
import collections

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

from . import audio as au
from . import config
from .text import normalize

BLOCK_SECONDS = 0.1
LEAD_IN_SECONDS = 0.5  # keep audio from just before the first loud block, or "Reo" gets clipped


def _within_one_edit(a: str, b: str) -> bool:
    """True if one insertion, deletion or substitution turns a into b."""
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) <= 1
    short, long = (a, b) if len(a) < len(b) else (b, a)
    for i in range(len(long)):
        if short == long[:i] + long[i + 1:]:
            return True
    return False


class WakeListener:
    def __init__(self):
        print(f"Loading wake model ({config.WAKE_MODEL})...")
        self.model = WhisperModel(config.WAKE_MODEL, device="cpu", compute_type="int8")

    def _transcribe(self, frames):
        """Returns (text, note). note explains an empty text."""
        raw = np.concatenate(frames).flatten()
        peak = float(np.max(np.abs(raw)))
        clip = au.normalize(raw)
        secs = len(raw) / config.SAMPLE_RATE

        segments, _ = self.model.transcribe(clip, language=config.LANGUAGE, beam_size=1,
                                            vad_filter=True, **au.CAREFUL)
        text = " ".join(s.text for s in segments).strip()

        if not text:
            # strict thresholds, the VAD, or genuinely no speech
            return "", f"{secs:.1f}s at peak {peak:.3f} produced no words"
        if au.looks_hallucinated(text):
            return "", f'ignored a repeat: "{text[:45]}"'
        return text, ""

    @staticmethod
    def _is_name(word: str) -> bool:
        if word in config.WAKE_WORDS:
            return True
        if not config.WAKE_FUZZY or not 3 <= len(word) <= 5:
            return False
        # one wrong letter is the difference between hearing "reo" and "rio"
        return any(_within_one_edit(word, name) for name in config.WAKE_WORDS)

    @staticmethod
    def _command_after_wake(text: str):
        """Return what was said after the wake word, "" if nothing, None if not addressed."""
        words = normalize(text).split()
        for i, word in enumerate(words):
            if WakeListener._is_name(word):
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
                    heard, note = self._transcribe(frames)
                    if heard:
                        command = WakeListener._command_after_wake(heard)
                        if command is not None:
                            return command
                    # say exactly why nothing happened, or a deaf mic and an
                    # unrecognised name look identical from the outside
                    print(f'  (heard "{heard}")' if heard else f"  ({note})")
                    lead_in.clear()
                    frames, speaking, silent = [], False, 0
