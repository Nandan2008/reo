"""Speech output. Offline pyttsx3 by default, Groq's Orpheus when asked for."""
import io
import re
import threading
import time
import wave

import numpy as np
import pyttsx3
import sounddevice as sd

from . import config

# Emotion the brain tagged its reply with -> (words per minute, volume).
# pyttsx3 gives us no pitch control on Windows, so pace and loudness carry it.
VOICE_SETTINGS = {
    "neutral":    (185, 0.95),
    "happy":      (200, 1.00),
    "excited":    (215, 1.00),
    "amused":     (195, 1.00),
    "sad":        (160, 0.80),
    "concerned":  (170, 0.90),
    "apologetic": (165, 0.85),
}

TAG = re.compile(r"^\s*\[(\w+)]\s*")

PLAY_GRACE = 5.0         # seconds past the clip length before giving up
_orpheus_failed = False  # fall back quietly after the first failure


# Whatever bracketed tokens the reply opens with. Not just [word]: the model has
# emitted "[emotion?]" and "[...]", and reo read them out loud.
LEADING_TAGS = re.compile(r"^(?:\s*\[[^\[\]\s]{0,15}])+\s*")
# A named emotion anywhere, since the model sometimes repeats it mid-reply.
# Brackets elsewhere are left alone, or a range like [1-10] gets eaten.
NAMED_TAG = re.compile(r"\[(\w+)]")


def _undouble(text: str) -> str:
    """gpt-oss sometimes emits its whole answer twice. Say it once.

    It also sometimes tacks the bare answer onto the end of the sentence that
    already contained it - "The capital of Peru is Lima. Lima." - so a short
    last fragment that the sentence before already said is dropped too. The
    fragment has to be contained in the previous sentence, which leaves
    "I like Rome. Rome is warm." alone.
    """
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s*", text) if p.strip()]
    if len(parts) > 1 and len(set(parts)) == 1:
        return parts[0]
    while len(parts) > 1 and _is_echo(parts[-1], parts[-2]):
        parts.pop()
    return " ".join(parts)


def _is_echo(tail: str, before: str) -> bool:
    """Is this short last fragment just the end of the sentence before it?"""
    words = tail.rstrip(".!?").strip().lower()
    if not words or len(words.split()) > 4:
        return False
    return words in before.rstrip(".!?").strip().lower()


def split_emotion(text: str) -> tuple[str, str]:
    """Pull the brain's [emotion] tag off, so it is never spoken.

    The tag belongs at the front, but the model sometimes repeats it mid-reply,
    so every one is removed rather than just the first.
    """
    named = [t.lower() for t in NAMED_TAG.findall(text)]
    emotion = next((t for t in named if t in VOICE_SETTINGS), "neutral")

    spoken = LEADING_TAGS.sub("", text)
    # and any repeat of a real emotion name later in the reply
    spoken = NAMED_TAG.sub(
        lambda m: "" if m.group(1).lower() in VOICE_SETTINGS else m.group(0), spoken)
    spoken = _undouble(spoken.strip())
    return emotion, spoken


def _pick_voice(engine) -> None:
    if not config.VOICE_NAME:
        return
    wanted = config.VOICE_NAME.lower()
    for voice in engine.getProperty("voices"):
        if wanted in voice.name.lower() or wanted in voice.id.lower():
            engine.setProperty("voice", voice.id)
            return


def _speak_pyttsx3(text: str, emotion: str, until=None) -> bool:
    """Say it. True if it was cut off part way through.

    runAndWait blocks until the whole sentence is out, so the watching has to
    happen beside it on another thread. stop() from there really does cut SAPI
    off mid-word: measured 11.28s to say the whole of a long sentence against
    1.17s when stopped after one second.
    """
    rate, volume = VOICE_SETTINGS[emotion]
    # A fresh engine per call avoids a known pyttsx3 hang on Windows
    engine = pyttsx3.init()
    _pick_voice(engine)
    engine.setProperty("rate", rate)
    engine.setProperty("volume", volume)
    engine.say(text)

    if until is None:
        engine.runAndWait()
        engine.stop()
        return False

    over, cut = threading.Event(), threading.Event()

    def watch():
        while not over.is_set():
            if until():
                cut.set()
                engine.stop()
                return
            over.wait(0.05)

    threading.Thread(target=watch, daemon=True, name="barge-in").start()
    try:
        engine.runAndWait()
    finally:
        over.set()
    engine.stop()
    return cut.is_set()


def _speak_orpheus(text: str, until=None) -> bool:
    from openai import OpenAI

    from . import providers

    # Orpheus is hosted by Groq alone, so the voice needs the Groq key even when
    # the thinking has moved to another provider.
    voice_host = providers.hosting(config.TTS_MODEL)
    if voice_host is None:
        raise RuntimeError("Orpheus needs a Groq key; set REO_TTS=pyttsx3 or add one.")
    client = OpenAI(api_key=voice_host.key, base_url=voice_host.base_url,
                    timeout=config.REQUEST_TIMEOUT, max_retries=1)
    response = client.audio.speech.create(
        model=config.TTS_MODEL,
        voice=config.TTS_VOICE,
        input=text,
        response_format="wav",
    )
    with wave.open(io.BytesIO(response.read())) as w:
        # Groq sends a streaming header claiming 2147483647 frames, so the count
        # is meaningless; what readframes actually returns is the clip.
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        channels, rate = w.getnchannels(), w.getframerate()

    if channels > 1:
        audio = audio.reshape(-1, channels)
    else:
        audio, rate = _tighten(audio, rate)
    return _play(audio, rate, until)


def _tighten(audio, rate):
    """Cut the dead air off each end and speak a little faster.

    Orpheus pads generously: of a 2.56s clip for "Got it. Done.", 0.29s was
    silence at the front and 0.55s at the back, with 1.72s of actual speech.
    A third of the waiting was nothing at all.
    """
    if audio.ndim > 1 or len(audio) == 0:
        return audio, rate

    peak = np.abs(audio).max()
    if peak:
        loud = np.abs(audio) > peak * 0.02
        if loud.any():
            first = int(np.argmax(loud))
            last = len(loud) - int(np.argmax(loud[::-1]))
            pad = int(rate * 0.05)          # keep a little, or it clips the word
            audio = audio[max(0, first - pad):min(len(audio), last + pad)]

    # Claiming a higher sample rate plays it faster, with no resampling cost.
    return audio, int(rate * config.SPEECH_SPEED)


def _play(audio, rate, until=None) -> bool:
    """Play it, but never wait forever.

    sd.wait() blocks with no timeout. When something else holds the sound device
    - a second copy of reo, say - it never returns, and reo looks frozen. One
    measured hang ran to 72,304 seconds.
    """
    expected = len(audio) / rate
    sd.play(audio, rate)
    # Generous: under load the clip really can take longer than its own
    # length, and cutting a reply off mid-word is worse than waiting.
    deadline = time.monotonic() + expected * 2 + PLAY_GRACE
    while time.monotonic() < deadline:
        if until is not None and until():
            sd.stop()
            return True
        stream = sd.get_stream()
        if stream is None or not stream.active:
            return False
        time.sleep(0.05)
    sd.stop()
    print(f"  [voice] playback did not finish in {expected + PLAY_GRACE:.0f}s; moving on.")
    return False


def speak(text: str, ui=None, until=None) -> bool:
    """Say it out loud. True if it was cut off part way through.

    `until` is asked, every fiftieth of a second, whether to stop. Without it
    reo talks over anyone who tries to interrupt, and a reply to a question it
    misheard has to be sat through to the end.
    """
    global _orpheus_failed

    emotion, text = split_emotion(text)
    print(f"Reo ({emotion}): {text}" if emotion != "neutral" else f"Reo: {text}")
    if ui:
        ui.line("reo", text)
        ui.set_state("speaking")

    if config.TTS_BACKEND == "orpheus" and not _orpheus_failed:
        try:
            return _speak_orpheus(text, until)
        except Exception as e:
            _orpheus_failed = True
            print(f"  [voice] Orpheus unavailable ({e}); using the offline voice.")

    return _speak_pyttsx3(text, emotion, until)
