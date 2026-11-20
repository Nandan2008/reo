"""Cleaning up what a quiet laptop microphone gives us before Whisper sees it."""
import numpy as np

TARGET_PEAK = 0.5
MAX_GAIN = 12.0      # beyond this we are only amplifying room noise
SILENT = 1e-4

# Tightening Whisper's own thresholds below their defaults threw away genuine
# speech along with the hallucinations. looks_hallucinated() and having to hear
# the name filter the noise well enough on their own.
CAREFUL = dict(
    condition_on_previous_text=False,   # stops one hallucination seeding the next
)


def normalize(audio: np.ndarray) -> np.ndarray:
    """Bring a quiet recording up to a level Whisper was trained on."""
    peak = float(np.max(np.abs(audio)))
    if peak < SILENT:
        return audio
    return audio * min(TARGET_PEAK / peak, MAX_GAIN)


def as_wav(audio: np.ndarray, rate: int = 16000) -> bytes:
    """A recording as a .wav file in memory, for sending somewhere."""
    import io
    import wave

    samples = (np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
               * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(rate)
        file.writeframes(samples.tobytes())
    return buffer.getvalue()


def has_speech(audio: np.ndarray, floor: float) -> bool:
    """Is there anything in here louder than a quiet room?

    Whisper does not return nothing for silence - it invents. Given a few
    seconds of room noise it produced "I've been on 30 YouTube videos again
    and again and again and again", and reo answered it. The surest cure is
    not to ask: a recording whose loudest moment never approaches speech is
    thrown away before transcription, which also saves the time.
    """
    if audio is None or not len(audio):
        return False
    return float(np.max(np.abs(audio))) >= floor


def _says_twice(words: list[str], size: int, times: int = 3) -> bool:
    """Is some run of `size` words repeated `times` times back to back?"""
    for start in range(len(words) - size * times + 1):
        chunk = words[start:start + size]
        if all(words[start + i * size:start + (i + 1) * size] == chunk
               for i in range(1, times)):
            return True
    return False


# What whisper emits when handed a quiet room. These are not guesses: they
# are the handful of phrases it falls back on, and two of them - "bye" and
# "goodbye" - are how reo is told to stop. One of them ended a session he was
# in the middle of using.
#
# Only ever applied to a clip with no real voice in it, because all of these
# are things a person genuinely says.
FROM_SILENCE = {
    "bye", "goodbye", "bye bye", "thank you", "thanks", "thank you very much",
    "thanks for watching", "thanks for watching!", "you", "oh", "um", "uh",
    "please subscribe", "subscribe", "the end", "okay", "ok", "so",
    "amara org", "subtitles by the amara org community", "transcription by",
    "i'm sorry", "music", "applause", "silence", "blank audio",
}


def from_silence(text: str) -> bool:
    """Is this one of whisper's stock answers to having heard nothing?"""
    plain = "".join(c if c.isalnum() or c in " '" else " " for c in text.lower())
    return " ".join(plain.split()) in FROM_SILENCE


def looks_hallucinated(text: str) -> bool:
    """Whisper looping rather than you talking.

    Two shapes. A whole sentence repeated back to back, which the first check
    has always caught. And a phrase stuttering inside one sentence - "again
    and again and again and again" - which it did not, because the sentence
    as a whole was unique.
    """
    sentences = [s.strip().lower() for s in text.replace("?", ".").replace("!", ".").split(".")]
    sentences = [s for s in sentences if len(s) > 3]
    if len(sentences) >= 2 and len(set(sentences)) == 1:
        return True

    words = [w for w in "".join(
        c if c.isalnum() else " " for c in text.lower()).split() if w]
    # How many repeats it takes to be suspicious depends on how long the
    # repeated run is. One word three times over is ordinary speech - "no no
    # no, stop" - so a single word needs four. A six word phrase said twice
    # back to back is never a person: that is Whisper's "Subtitles by the
    # Amara.org community Subtitles by the Amara.org community".
    return (_says_twice(words, 1, times=4)
            or any(_says_twice(words, size, times=3) for size in (2, 3))
            or any(_says_twice(words, size, times=2) for size in range(4, 9)))


# --- telling a voice from a room ------------------------------------------
#
# Loudness cannot do it. A door, a keyboard and a chair all peak higher than
# someone speaking quietly a few feet away, and reo answered all three. This
# is the model whisper itself uses to find speech, run live on each block as
# it arrives instead of once at the end.

FRAME = 512          # what the model takes at a time: 32ms at 16kHz
CONTEXT = 64         # samples of the previous frame it also wants
SPEECH_STARTS = 0.5  # how sure it has to be that someone began talking
SPEECH_CONTINUES = 0.25   # and how sure, once they have, that they still are


class VoiceDetector:
    """Is a person speaking in this block?

    Silero is meant to be fed a stream, carrying its memory from one frame to
    the next. Measured against running it over a whole recording at the end:
    carried forward it agrees on 98.9% of frames, reset every block only 93.6%.
    So this keeps the state itself rather than using the convenience wrapper.

    Costs 0.18ms per 64ms block, which is a fifth of one percent of a core.
    """

    def __init__(self):
        self._session = None
        self._tail = np.zeros(0, dtype=np.float32)
        try:
            from faster_whisper.vad import get_vad_model
            self._session = get_vad_model().session
        except Exception as e:      # no onnxruntime, or the internals moved
            print(f"  [listen] voice detection unavailable ({e}); using loudness")
        self.reset()

    @property
    def ready(self) -> bool:
        return self._session is not None

    def reset(self) -> None:
        """Forget the last sentence before listening for the next one."""
        self._h = np.zeros((1, 1, 128), dtype="float32")
        self._c = np.zeros((1, 1, 128), dtype="float32")
        self._context = np.zeros((1, CONTEXT), dtype="float32")
        self._tail = np.zeros(0, dtype=np.float32)

    def chance(self, block: np.ndarray) -> float:
        """How likely it is that this block holds a voice, 0 to 1.

        Returns -1 when the model is not there, so the caller can fall back to
        loudness rather than treating silence as speech or the other way round.
        """
        if self._session is None:
            return -1.0
        audio = np.concatenate([self._tail, np.asarray(block, dtype=np.float32).reshape(-1)])
        whole = len(audio) // FRAME * FRAME
        self._tail = audio[whole:]
        if not whole:
            return 0.0
        frames = audio[:whole].reshape(-1, FRAME)
        before = (self._context if len(frames) == 1
                  else np.vstack([self._context, frames[:-1, -CONTEXT:]]))
        try:
            probs, self._h, self._c = self._session.run(
                None, {"input": np.concatenate([before, frames], 1),
                       "h": self._h, "c": self._c})
        except Exception as e:
            print(f"  [listen] voice detection stopped ({e}); using loudness")
            self._session = None
            return -1.0
        self._context = frames[-1:, -CONTEXT:]
        return float(probs.max())
