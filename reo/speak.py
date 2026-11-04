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


def split_emotion(text: str) -> tuple[str, str]:
    """Pull the brain's leading [emotion] tag off, so it is never spoken."""
    match = TAG.match(text)
    if not match:
        return "neutral", text.strip()
    emotion = match.group(1).lower()
    return (emotion if emotion in VOICE_SETTINGS else "neutral"), text[match.end():].strip()


def _speak_pyttsx3(text: str, emotion: str) -> None:
    rate, volume = VOICE_SETTINGS[emotion]
    # A fresh engine per call avoids a known pyttsx3 hang on Windows
    engine = pyttsx3.init()
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


def speak(text: str) -> None:
    global _orpheus_failed

    emotion, text = split_emotion(text)
    print(f"Reo ({emotion}): {text}" if emotion != "neutral" else f"Reo: {text}")

    if config.TTS_BACKEND == "orpheus" and not _orpheus_failed:
        try:
            _speak_orpheus(text)
            return
        except Exception as e:
            _orpheus_failed = True
            print(f"  [voice] Orpheus unavailable ({e}); using the offline voice.")

    _speak_pyttsx3(text, emotion)
