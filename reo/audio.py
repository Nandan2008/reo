"""Cleaning up what a quiet laptop microphone gives us before Whisper sees it."""
import numpy as np

TARGET_PEAK = 0.5
MAX_GAIN = 12.0      # beyond this we are only amplifying room noise
SILENT = 1e-4

# Whisper invents text when there is little real speech, and it gives itself away
# by repeating. These are stricter than the defaults, which assume clean audio.
CAREFUL = dict(
    condition_on_previous_text=False,   # stops one hallucination seeding the next
    compression_ratio_threshold=2.0,    # repeated phrases compress well; drop them
    log_prob_threshold=-0.8,            # drop segments the model is unsure of
    no_speech_threshold=0.5,
)


def normalize(audio: np.ndarray) -> np.ndarray:
    """Bring a quiet recording up to a level Whisper was trained on."""
    peak = float(np.max(np.abs(audio)))
    if peak < SILENT:
        return audio
    return audio * min(TARGET_PEAK / peak, MAX_GAIN)


def looks_hallucinated(text: str) -> bool:
    """A sentence repeated back to back is Whisper looping, not you talking."""
    sentences = [s.strip().lower() for s in text.replace("?", ".").replace("!", ".").split(".")]
    sentences = [s for s in sentences if len(s) > 3]
    return len(sentences) >= 2 and len(set(sentences)) == 1
