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
