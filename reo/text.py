"""Tidying up what Whisper gives us, shared by the wake listener and the main loop."""


# Whisper uses the curly apostrophe about as often as the straight one.
APOSTROPHES = "'‘’ʼ´"


def normalize(text: str) -> str:
    """Strip Whisper's punctuation and spacing so phrases match.

    Apostrophes are deleted rather than turned into spaces: "what's" has to
    come out as "whats", not "what s", or every pattern written for the
    contraction people actually say misses it. "What's the battery" was
    reaching the model and costing a round trip to answer something this
    machine already knew.
    """
    stripped = "".join(c for c in text.lower() if c not in APOSTROPHES)
    kept = "".join(c if c.isalnum() else " " for c in stripped)
    return " ".join(kept.split())


def is_speech(text: str) -> bool:
    """Whisper emits bare punctuation on room noise; that is not worth an API call."""
    return any(c.isalnum() for c in text)
