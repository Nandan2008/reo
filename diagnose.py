"""Find out why reo is not hearing you. Run: .venv\\Scripts\\python.exe diagnose.py"""
import sys

import numpy as np
import sounddevice as sd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from reo import config
from reo.text import normalize

BLOCK = 0.1
SECONDS = 8


def meter():
    print(f"Default input : {sd.query_devices(kind='input')['name']}")
    print(f"Threshold     : {config.SILENCE_THRESHOLD}")
    print()
    print(f"Speak normally for {SECONDS} seconds - say \"Reo, what time is it?\" a couple of times.")
    print("A bar means the audio is loud enough to wake reo.")
    print()

    block = int(config.SAMPLE_RATE * BLOCK)
    frames, levels = [], []
    with sd.InputStream(samplerate=config.SAMPLE_RATE, channels=1,
                        dtype="float32", blocksize=block) as stream:
        for _ in range(int(SECONDS / BLOCK)):
            data, _ = stream.read(block)
            frames.append(data.copy())
            rms = float(np.sqrt(np.mean(data ** 2)))
            levels.append(rms)
            over = rms > config.SILENCE_THRESHOLD
            bar = "#" * min(int(rms * 400), 50)
            print(f"  {rms:7.4f} {'OVER ' if over else '     '}{bar}")

    return np.concatenate(frames).flatten(), levels


def verdict(levels):
    peak, quiet = max(levels), float(np.median(levels))
    over = sum(1 for l in levels if l > config.SILENCE_THRESHOLD)
    print()
    print(f"peak level      : {peak:.4f}")
    print(f"room floor      : {quiet:.4f}  (median, i.e. you not talking)")
    print(f"blocks over     : {over} of {len(levels)}")
    print(f"threshold now   : {config.SILENCE_THRESHOLD}")

    if peak < config.SILENCE_THRESHOLD:
        suggested = max(peak / 3, 0.0005)
        print(f"\nPROBLEM: nothing you said reached the threshold.")
        print(f"Set REO_SILENCE_THRESHOLD={suggested:.4f} in .env")
    elif quiet > config.SILENCE_THRESHOLD:
        print(f"\nPROBLEM: the room alone is over the threshold, so reo never hears silence.")
        print(f"Set REO_SILENCE_THRESHOLD={(peak + quiet) / 2:.4f} in .env")
    else:
        midpoint = (peak + quiet) / 2
        print(f"\nLevels look usable. A good threshold sits between the two: ~{midpoint:.4f}")


def transcribe(audio):
    import wave
    from faster_whisper import WhisperModel

    peak = float(np.max(np.abs(audio)))
    gained = audio * (0.5 / peak) if peak > 0 else audio
    print()
    print(f"peak {peak:.4f} -> normalised to 0.5 (gain x{0.5 / peak:.1f})")

    with wave.open("diagnose.wav", "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(config.SAMPLE_RATE)
        w.writeframes((gained * 32767).astype(np.int16).tobytes())
    print("saved diagnose.wav so you can play back what the mic actually got")

    for name in (config.WAKE_MODEL, config.WHISPER_MODEL):
        model = WhisperModel(name, device="cpu", compute_type="int8")
        for label, clip, vad in [("raw   + vad", audio, True),
                                 ("raw   - vad", audio, False),
                                 ("gained+ vad", gained, True),
                                 ("gained- vad", gained, False)]:
            segments, _ = model.transcribe(clip, language="en", beam_size=1, vad_filter=vad)
            text = " ".join(s.text for s in segments).strip()
            words = set(normalize(text).split())
            hit = words & config.WAKE_WORDS
            flag = "WAKE" if hit else "    "
            print(f"  {name:9} {label}  {flag}  {text[:70]!r}")


if __name__ == "__main__":
    audio, levels = meter()
    verdict(levels)
    transcribe(audio)
