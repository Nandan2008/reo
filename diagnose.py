"""Find out why reo is not hearing you. Run: .venv\\Scripts\\python.exe diagnose.py"""
import sys

import numpy as np
import sounddevice as sd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from reo import config
from reo.text import normalize

BLOCK = 0.1
SECONDS = 6


PHRASE = "Reo, what time is it?"


def meter():
    print(f"Default input : {sd.query_devices(kind='input')['name']}")
    print(f"Threshold     : {config.SILENCE_THRESHOLD}")
    print()
    print("When you press Enter, this records for 6 seconds.")
    print(f'Say this, clearly, close to the laptop:  "{PHRASE}"')
    print("Say it twice if you have time. Do not speak before pressing Enter.")
    try:
        print()
        input("Press Enter when you are ready to speak  ")
    except (KeyboardInterrupt, EOFError):
        raise SystemExit("cancelled")
    print()
    print("RECORDING - speak now")

    block = int(config.SAMPLE_RATE * BLOCK)
    frames, levels = [], []
    with sd.InputStream(samplerate=config.SAMPLE_RATE, channels=1,
                        dtype="float32", blocksize=block) as stream:
        for i in range(int(SECONDS / BLOCK)):
            data, _ = stream.read(block)
            frames.append(data.copy())
            rms = float(np.sqrt(np.mean(data ** 2)))
            levels.append(rms)
            if i % 2 == 0:
                bar = "#" * min(int(rms * 400), 50)
                print(f"  {rms:7.4f} {'OVER ' if rms > config.SILENCE_THRESHOLD else '     '}{bar}")
    print("done")

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
    import time
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

    print("
What each model hears, and how long it takes. Pick the smallest one")
    print("that gets your words right, and put it in .env as REO_WHISPER_MODEL.
")
    print(f"  vocabulary hint: {config.VOCABULARY or '(none)'}
")

    seen = []
    for name in dict.fromkeys(["tiny", "base", "small", config.WHISPER_MODEL]):
        try:
            model = WhisperModel(name, device="cpu", compute_type="int8")
        except Exception as e:
            print(f"  {name:9} unavailable: {str(e)[:60]}")
            continue
        for label, hints in [("plain ", None), ("+vocab", config.VOCABULARY or None)]:
            started = time.time()
            segments, _ = model.transcribe(gained, language="en",
                                           beam_size=config.BEAM_SIZE,
                                           hotwords=hints, vad_filter=True)
            text = " ".join(s.text for s in segments).strip()
            took = time.time() - started
            hit = "WAKE" if set(normalize(text).split()) & config.WAKE_WORDS else "    "
            print(f"  {name:9} {label}  {took:5.2f}s  {hit}  {text[:64]!r}")
            seen.append((name, label.strip(), text))
        print()

    if len({t for _, _, t in seen}) == 1:
        print("Every model heard the same thing, so the recording is clear.")
    else:
        print("They disagree, so the accent or the microphone is the limit here,")
        print("not the model alone. Add the words it got wrong to REO_VOCABULARY.")


if __name__ == "__main__":
    audio, levels = meter()
    verdict(levels)
    transcribe(audio)
