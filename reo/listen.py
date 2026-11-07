import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

from . import audio as au
from . import config

BLOCK_SECONDS = 0.1


class Listener:
    def __init__(self):
        print("Loading speech model (first run downloads it)...")
        self.model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")

    def _record(self, stop_now=None):
        block = int(config.SAMPLE_RATE * BLOCK_SECONDS)
        max_blocks = int(config.MAX_RECORD_SECONDS / BLOCK_SECONDS)
        wait_blocks = int(config.WAIT_FOR_SPEECH_SECONDS / BLOCK_SECONDS)
        silence_blocks = int(config.SILENCE_SECONDS / BLOCK_SECONDS)

        frames, heard, silent, asked_to_stop = [], False, 0, False
        with sd.InputStream(samplerate=config.SAMPLE_RATE, channels=1,
                            dtype="float32", blocksize=block) as stream:
            for i in range(max_blocks):
                if stop_now is not None and stop_now():
                    asked_to_stop = True
                    break
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

        if not frames:
            return None
        # When they pressed the key to stop, transcribe what was captured even if
        # it never crossed the loudness threshold. A quiet microphone kept "heard"
        # false, so every deliberate recording was binned as "didn't catch that".
        if heard or (asked_to_stop and len(frames) * BLOCK_SECONDS >= 0.4):
            return np.concatenate(frames).flatten()
        return None

    def listen(self, stop_now=None) -> str:
        audio = self._record(stop_now)
        if audio is None:
            return ""
        segments, _ = self.model.transcribe(au.normalize(audio), language=config.LANGUAGE,
                                            beam_size=1, vad_filter=True, **au.CAREFUL)
        text = " ".join(s.text for s in segments).strip()
        return "" if au.looks_hallucinated(text) else text
