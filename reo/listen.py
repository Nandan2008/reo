import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

from . import config

BLOCK_SECONDS = 0.1


class Listener:
    def __init__(self):
        print("Loading speech model (first run downloads it)...")
        self.model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")

    def _record(self):
        block = int(config.SAMPLE_RATE * BLOCK_SECONDS)
        max_blocks = int(config.MAX_RECORD_SECONDS / BLOCK_SECONDS)
        wait_blocks = int(config.WAIT_FOR_SPEECH_SECONDS / BLOCK_SECONDS)
        silence_blocks = int(config.SILENCE_SECONDS / BLOCK_SECONDS)

        frames, heard, silent = [], False, 0
        with sd.InputStream(samplerate=config.SAMPLE_RATE, channels=1,
                            dtype="float32", blocksize=block) as stream:
            for i in range(max_blocks):
                data, _ = stream.read(block)
                frames.append(data.copy())
                loud = float(np.sqrt(np.mean(data ** 2))) > config.SILENCE_THRESHOLD
                if loud:
                    heard, silent = True, 0
                elif heard:
                    silent += 1
                    if silent >= silence_blocks:
                        break
                elif i >= wait_blocks:
                    return None
            else:
                if heard:
                    print(f"  (stopped at the {config.MAX_RECORD_SECONDS} second limit)")
        return np.concatenate(frames).flatten() if heard else None

    def listen(self) -> str:
        audio = self._record()
        if audio is None:
            return ""
        segments, _ = self.model.transcribe(audio, language="en", beam_size=1, vad_filter=True)
        return " ".join(s.text for s in segments).strip()
