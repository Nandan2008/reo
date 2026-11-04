"""Tidying up what Whisper gives us, shared by the wake listener and the main loop."""


def normalize(text: str) -> str:
    """Strip Whisper's punctuation and spacing so phrases match."""
    kept = "".join(c if c.isalnum() else " " for c in text.lower())
    return " ".join(kept.split())


def is_speech(text: str) -> bool:
    """Whisper emits bare punctuation on room noise; that is not worth an API call."""
    return any(c.isalnum() for c in text)
