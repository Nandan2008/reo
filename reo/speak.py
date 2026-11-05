"""Speech output. Offline pyttsx3 by default, Groq's Orpheus when asked for."""
import io
import re
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

_orpheus_failed = False  # fall back quietly after the first failure


ANY_TAG = re.compile(r"\[(\w+)]")


def _undouble(text: str) -> str:
    """gpt-oss sometimes emits its whole answer twice. Say it once."""
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s*", text) if p.strip()]
    if len(parts) > 1 and len(set(parts)) == 1:
        return parts[0]
    return text


def split_emotion(text: str) -> tuple[str, str]:
    """Pull the brain's [emotion] tag off, so it is never spoken.

    The tag belongs at the front, but the model sometimes repeats it mid-reply,
    so every one is removed rather than just the first.
    """
    tags = ANY_TAG.findall(text)
    emotion = next((t.lower() for t in tags if t.lower() in VOICE_SETTINGS), "neutral")
    spoken = _undouble(ANY_TAG.sub("", text).strip())
    return emotion, spoken


def _pick_voice(engine) -> None:
    if not config.VOICE_NAME:
        return
    wanted = config.VOICE_NAME.lower()
    for voice in engine.getProperty("voices"):
        if wanted in voice.name.lower() or wanted in voice.id.lower():
            engine.setProperty("voice", voice.id)
            return


def _speak_pyttsx3(text: str, emotion: str) -> None:
    rate, volume = VOICE_SETTINGS[emotion]
    # A fresh engine per call avoids a known pyttsx3 hang on Windows
    engine = pyttsx3.init()
    _pick_voice(engine)
    engine.setProperty("rate", rate)
    engine.setProperty("volume", volume)
    engine.say(text)
    engine.runAndWait()
    engine.stop()


def _speak_orpheus(text: str) -> None:
    from openai import OpenAI

    client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
    response = client.audio.speech.create(
        model=config.TTS_MODEL,
        voice=config.TTS_VOICE,
        input=text,
        response_format="wav",
    )
    with wave.open(io.BytesIO(response.read())) as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if w.getnchannels() > 1:
            audio = audio.reshape(-1, w.getnchannels())
        sd.play(audio, w.getframerate())
        sd.wait()


def speak(text: str, ui=None) -> None:
    global _orpheus_failed

    emotion, text = split_emotion(text)
    print(f"Reo ({emotion}): {text}" if emotion != "neutral" else f"Reo: {text}")
    if ui:
        ui.line("reo", text)
        ui.set_state("speaking")

    if config.TTS_BACKEND == "orpheus" and not _orpheus_failed:
        try:
            _speak_orpheus(text)
            return
        except Exception as e:
            _orpheus_failed = True
            print(f"  [voice] Orpheus unavailable ({e}); using the offline voice.")

    _speak_pyttsx3(text, emotion)
