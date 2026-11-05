"""Wake word detection with a model trained for the job.

The earlier approach transcribed every sound in the room with Whisper and looked
for a name in the text. That asks a general speech model to do a job it was not
built for, and on a quiet microphone it mostly returned hallucinations.

openWakeWord scores 80ms frames against a model trained on one phrase. It is far
cheaper to run continuously and far harder to fool: on synthesised speech here it
scored 0.998 for "Hey Jarvis" and 0.000 for unrelated sentences.
"""
import os
import re
from pathlib import Path

import numpy as np
import sounddevice as sd

from . import config

FRAME = 1280          # 80ms at 16kHz, the frame size the models expect
RATE = 16000


class PhraseListener:
    """Waits for the wake phrase, then hands over to the command listener."""

    def __init__(self, listener):
        from openwakeword.model import Model

        print(f'Loading wake word ("{pretty(config.WAKE_PHRASE)}")...')
        self.model = Model(inference_framework="onnx", wakeword_models=[config.WAKE_PHRASE])
        # A pretrained name keys on itself, a custom file keys on its filename,
        # so take whatever the model actually registered.
        self.key = next(iter(self.model.models))
        self.listener = listener

    def wait(self) -> str:
        """Block until the phrase is heard, then return what was said after it."""
        with sd.InputStream(samplerate=RATE, channels=1,
                            dtype="int16", blocksize=FRAME) as stream:
            while True:
                frame, _ = stream.read(FRAME)
                score = self.model.predict(frame.flatten())[self.key]
                if score >= config.WAKE_THRESHOLD:
                    print(f"  (heard the wake word, {score:.2f})")
                    self.model.reset()
                    break

        # They are most likely still mid-sentence, so record immediately rather
        # than answering "Yes?" over the top of them.
        return self.listener.listen()


def pretty(spec: str) -> str:
    """Turn "hey_jarvis", or a path to a trained model, into something sayable."""
    stem = Path(spec).stem if spec.endswith(".onnx") or os.sep in spec else spec
    return re.sub(r"[_-]v?[\d.]+$", "", stem).replace("_", " ").strip()


def available(name: str) -> bool:
    """Is this one of the phrases openWakeWord ships a model for?"""
    try:
        from openwakeword.model import Model
        return name in Model(inference_framework="onnx").models
    except Exception:
        return False
