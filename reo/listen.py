import time

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

from . import audio as au
from . import config

BLOCK_SECONDS = 0.1
# Below this, Whisper is guessing at the language rather than recognising it,
# so the guess is not worth keeping for the rest of the session.
LANGUAGE_CONFIDENCE = 0.75


def _shut(stream) -> None:
    """Close a stream without caring whether it is still alive.

    Handling the failed read was not enough: `with sd.InputStream(...)` stops
    and closes on the way out, and stopping a stream the device has already
    dropped raises "Stream is stopped" a second time. That one escaped and
    still ended the process, after the first had been caught and reported.
    """
    for shut in (getattr(stream, "stop", None), getattr(stream, "close", None)):
        if shut is None:
            continue
        try:
            shut()
        except Exception:
            pass


class Listener:
    def __init__(self):
        print("Loading speech model (first run downloads it)...")
        self.model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")
        # None means "work it out from what you hear, once". Detection costs
        # about as much again as the transcription itself - measured at 4.5s
        # against 2.2s on the same three second clip - and paying that on every
        # single utterance was the largest part of reo's lag.
        self.language = config.LANGUAGE
        self.detected = False
        self.transcribe_seconds = 0.0    # what the last transcription cost
        self._warm_up()

    def _warm_up(self):
        """Transcribe a moment of silence so the first real question does not
        also pay for allocating the model's working memory."""
        try:
            t0 = time.time()
            segments, _ = self.model.transcribe(
                np.zeros(config.SAMPLE_RATE // 2, dtype=np.float32),
                language=self.language or "en", beam_size=1)
            list(segments)
            print(f"Speech model ready ({time.time() - t0:.1f}s).")
        except Exception as e:                      # never block startup on this
            print(f"  [listen] warm-up skipped: {e}")

    def _record(self, stop_now=None):
        block = int(config.SAMPLE_RATE * BLOCK_SECONDS)
        max_blocks = int(config.MAX_RECORD_SECONDS / BLOCK_SECONDS)
        wait_blocks = int(config.WAIT_FOR_SPEECH_SECONDS / BLOCK_SECONDS)
        silence_blocks = int(config.SILENCE_SECONDS / BLOCK_SECONDS)

        frames, heard, silent, asked_to_stop, nothing_said = [], False, 0, False, False
        try:
            stream = sd.InputStream(samplerate=config.SAMPLE_RATE, channels=1,
                                    dtype="float32", blocksize=block)
            stream.start()
        except sd.PortAudioError as e:
            # No microphone, or something else holds it. Saying so and carrying
            # on beats taking the whole assistant down with us.
            print(f"  [listen] cannot open the microphone: {e}")
            return None

        try:
            for i in range(max_blocks):
                if stop_now is not None and stop_now():
                    asked_to_stop = True
                    break
                try:
                    data, _ = stream.read(block)
                except sd.PortAudioError as e:
                    # The device can go away mid-sentence: unplugged, grabbed by
                    # another program, or the laptop coming back from sleep.
                    print(f"  [listen] the microphone stopped mid-recording ({e})")
                    asked_to_stop = True
                    break
                frames.append(data.copy())
                loud = float(np.sqrt(np.mean(data ** 2))) > config.SILENCE_THRESHOLD
                if loud:
                    heard, silent = True, 0
                elif heard:
                    silent += 1
                    if silent >= silence_blocks:
                        break
                elif i >= wait_blocks:
                    nothing_said = True
                    break
            else:
                if heard:
                    print(f"  (stopped at the {config.MAX_RECORD_SECONDS} second limit)")
        finally:
            _shut(stream)

        if nothing_said or not frames:
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
        clean = au.normalize(audio)

        started = time.time()
        if self.language is None:
            self._choose_language(clean)
        segments, _ = self.model.transcribe(
            clean, language=self.language, beam_size=config.BEAM_SIZE,
            hotwords=config.VOCABULARY or None, vad_filter=True, **au.CAREFUL)
        text = " ".join(s.text for s in segments).strip()
        self.transcribe_seconds = time.time() - started

        if au.looks_hallucinated(text):
            text = ""
        if not text:
            # Nothing usable came back. If the language was a guess, it may have
            # been the wrong guess, so let the next utterance decide again.
            self._forget_language()
            return ""
        return text

    def _choose_language(self, audio) -> None:
        """Work out the language once, choosing only among the ones spoken here.

        Left to choose freely, Whisper read this accent as Russian and Spanish
        and wrote confident nonsense. Ranking only the configured languages
        cannot produce a language the user does not speak.
        """
        try:
            _, _, every = self.model.detect_language(audio, vad_filter=True)
        except Exception as e:
            print(f"  [listen] could not detect the language ({e}); assuming English")
            self.language, self.detected = "en", True
            return
        allowed = set(config.LANGUAGES)
        ranked = sorted(((p, code) for code, p in every if code in allowed), reverse=True)
        if not ranked:
            return      # nothing configured matched; let whisper decide this once
        confidence, best = ranked[0]
        if confidence >= LANGUAGE_CONFIDENCE:
            self.language, self.detected = best, True
            print(f"  [listen] language is {best} ({confidence:.0%}); reusing it")

    def _forget_language(self) -> None:
        if self.detected:
            self.language, self.detected = None, False
