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
from .reply import tighten as drop_offers

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
    """Is this short last fragment just the end of the sentence before it?

    Only the short-fragment case. The model also sometimes says the same
    thing twice in different words - "Wrote the note. The note is saved as
    the note." - and a rule loose enough to catch that was loose enough to
    eat a real second sentence, so that one is left alone and simply sounds
    a bit redundant.
    """
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
    # And the offer of further help its own prompt tells it not to make.
    # Here rather than only in the brain, so what gets printed, shown in the
    # window and kept in the conversation is what was actually said out loud.
    spoken = drop_offers(spoken)
    return emotion, spoken


def said_aloud(text: str) -> str:
    """What reo actually says, given what the model produced: the emotion tag
    off, the answer said once, and no offer of further help."""
    return split_emotion(text)[1]


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


_voice_client = None


def _client():
    """One client for the voice, built once and kept.

    It used to be built per reply. Measured on a cold one: 2.69s before a
    single byte came back, against 0.20s once it is warm.
    """
    global _voice_client
    if _voice_client is None:
        from openai import OpenAI

        from . import providers

        # Orpheus is hosted by Groq alone, so the voice needs the Groq key
        # even when the thinking has moved to another provider.
        voice_host = providers.hosting(config.TTS_MODEL)
        if voice_host is None:
            raise RuntimeError("Orpheus needs a Groq key; set REO_TTS=pyttsx3 or add one.")
        _voice_client = OpenAI(api_key=voice_host.key, base_url=voice_host.base_url,
                               timeout=config.REQUEST_TIMEOUT, max_retries=1)
    return _voice_client


def _speak_orpheus(text: str, until=None) -> bool:
    client = _client()
    try:
        with client.audio.speech.with_streaming_response.create(
                model=config.TTS_MODEL, voice=config.TTS_VOICE,
                input=text, response_format="wav") as response:
            return _play_arriving(response.iter_bytes(chunk_size=16384), until)
    except _NoAudio:
        pass            # nothing usable came back; fall through to the whole-file way
    response = client.audio.speech.create(
        model=config.TTS_MODEL, voice=config.TTS_VOICE,
        input=text, response_format="wav")
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


class _NoAudio(Exception):
    """The streamed response held no usable WAV."""


def _wav_shape(raw: bytes):
    """Where the samples start in a WAV, and at what rate. None if the header
    has not all arrived yet."""
    if len(raw) < 12 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        return None
    at, rate, channels, bits = 12, None, 1, 16
    while at + 8 <= len(raw):
        name = raw[at:at + 4]
        size = int.from_bytes(raw[at + 4:at + 8], "little")
        body = at + 8
        if name == b"fmt " and body + 16 <= len(raw):
            channels = int.from_bytes(raw[body + 2:body + 4], "little") or 1
            rate = int.from_bytes(raw[body + 4:body + 8], "little")
            bits = int.from_bytes(raw[body + 14:body + 16], "little") or 16
        elif name == b"data":
            # Groq claims 2147483647 bytes here, so the size is no use; what
            # matters is that everything after this point is samples.
            return (rate, channels, bits, body) if rate else None
        at = body + size + (size & 1)
    return None


def _first_sound(samples) -> int:
    """Where the speech starts, so the pause at the front is not played."""
    if not len(samples):
        return 0
    peak = int(np.abs(samples).max())
    if peak == 0:
        return 0
    loud = np.flatnonzero(np.abs(samples) > peak * 0.02)
    return int(loud[0]) if len(loud) else 0


# The speaker, opened once and left open. Opening one costs 287ms on this
# laptop - measured - and that was paid on every single reply, by sd.play()
# as much as by anything else. Starting an already-open stream costs nothing.
# It plays silence between clips, which is what a sound card does anyway.
_out = None
_out_rate = None
_clip = {"audio": np.zeros(0, dtype=np.int16), "at": 0, "end": 0, "final": True}
_clip_lock = threading.Lock()
_clip_over = threading.Event()


def _feed(out, frames, moment, status):
    """Hand the card whatever is ready. Never raises CallbackStop: the stream
    outlives the clip, so it fills with silence and waits for the next one."""
    with _clip_lock:
        audio, at, end = _clip["audio"], _clip["at"], _clip["end"]
        take = max(0, min(frames, end - at))
        _clip["at"] = at + take
        over = _clip["final"] and _clip["at"] >= end
        block = audio[at:at + take]
    if take:
        out[:take, 0] = block
    if take < frames:
        out[take:, 0] = 0
    if over:
        _clip_over.set()


def _speaker(rate: int):
    """The open stream, at this rate. Rebuilt only if the rate changes."""
    global _out, _out_rate
    if _out is not None and _out_rate == rate and _out.active:
        return _out
    close()
    _out = sd.OutputStream(samplerate=rate, channels=1, dtype="int16",
                           callback=_feed)
    _out_rate = rate
    _out.start()
    return _out


def close() -> None:
    """Hand the sound card back."""
    global _out, _out_rate
    stream, _out, _out_rate = _out, None, None
    if stream is None:
        return
    try:
        stream.stop()
        stream.close()
    except Exception:
        pass


def _begin(audio, rate: int, final: bool) -> None:
    _clip_over.clear()
    with _clip_lock:
        _clip["audio"] = audio
        _clip["at"] = _first_sound(audio) if final else 0
        _clip["end"] = len(audio)
        _clip["final"] = final
    _speaker(rate)


def _wait(seconds: float, until=None) -> bool:
    """Until it has finished, or he cuts in. True if he cut in."""
    deadline = time.monotonic() + seconds * 2 + PLAY_GRACE
    while not _clip_over.wait(0.04):
        if until is not None and until():
            with _clip_lock:
                _clip["end"] = _clip["at"]          # stop where it got to
                _clip["final"] = True
            return True
        if time.monotonic() > deadline:
            with _clip_lock:
                _clip["end"] = _clip["at"]
                _clip["final"] = True
            print(f"  [voice] playback did not finish in {seconds + PLAY_GRACE:.0f}s; "
                  "moving on.")
            return False
    return False


def _play_arriving(chunks, until=None) -> bool:
    """Speak from a buffer that is still filling.

    Measured: the first bytes arrive 0.20s in and the last at 0.74s, so two
    thirds of the wait was the transfer rather than groq thinking. The clip
    comes down about eight times faster than it plays, so once a fifth of a
    second is in hand the rest will always keep up.

    The silence on the end still gets trimmed. The download is long finished
    before playback reaches the end, so the end can simply be moved in once
    the last chunk has landed.
    """
    raw = bytearray()
    shape = None
    for chunk in chunks:
        raw.extend(chunk)
        shape = _wav_shape(raw)
        if shape:
            break
    if not shape:
        raise _NoAudio
    rate, channels, bits, start = shape
    if bits != 16 or channels != 1:
        raise _NoAudio                  # the whole-file path can cope; this cannot

    spare = bytearray(raw[start:])
    held = {"audio": np.zeros(0, dtype=np.int16)}

    def absorb():
        whole = len(spare) // 2 * 2
        if not whole:
            return
        block = np.frombuffer(bytes(spare[:whole]), dtype=np.int16)
        del spare[:whole]
        held["audio"] = np.concatenate([held["audio"], block])

    absorb()
    enough = int(rate * 0.2)
    for chunk in chunks:
        spare.extend(chunk)
        absorb()
        if len(held["audio"]) >= enough:
            break
    if not len(held["audio"]):
        raise _NoAudio

    # Claiming a higher sample rate plays it faster, with no resampling cost.
    speed = max(8000, int(rate * config.SPEECH_SPEED))
    _begin(held["audio"], speed, final=False)
    with _clip_lock:
        _clip["at"] = _first_sound(held["audio"])   # skip the pause at the front

    def rest():
        try:
            for chunk in chunks:
                spare.extend(chunk)
                absorb()
                with _clip_lock:
                    _clip["audio"] = held["audio"]
                    _clip["end"] = len(held["audio"])
        except Exception as e:
            print(f"  [voice] the rest of the clip did not arrive ({e})")
        whole = held["audio"]
        tail = len(whole)
        if tail:
            peak = int(np.abs(whole).max())
            if peak:
                loud = np.flatnonzero(np.abs(whole) > peak * 0.02)
                if len(loud):
                    tail = min(tail, int(loud[-1]) + int(rate * 0.05))
        with _clip_lock:
            _clip["audio"] = whole
            _clip["end"] = tail
            _clip["final"] = True
            if _clip["at"] >= tail:
                _clip_over.set()

    threading.Thread(target=rest, daemon=True, name="voice").start()
    return _wait(len(held["audio"]) / rate + 30, until)


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
    """Play a clip that is already in hand, through the same kept speaker.

    sd.play() opens the device itself, which is the 287ms this avoids, and
    sd.wait() blocks with no timeout at all - when something else held the
    sound card it never returned, and reo looked frozen. One measured hang
    ran to 72,304 seconds.
    """
    if getattr(audio, "ndim", 1) > 1:       # stereo: rare, and not worth a path
        audio = audio.mean(axis=1).astype(np.int16)
    _begin(np.ascontiguousarray(audio, dtype=np.int16), rate, final=True)
    return _wait(len(audio) / rate, until)


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

    try:
        return _speak_pyttsx3(text, emotion, until)
    except Exception as e:
        # pyttsx3 talks to SAPI over COM, which fails on Windows for its own
        # reasons, and this was unguarded all the way up: a voice that fell
        # over in the middle of answering ended the whole session, with the
        # traceback going to a stderr nobody was reading. Reo is allowed to
        # lose its voice; it is not allowed to die of it.
        print(f"  [voice] could not speak ({type(e).__name__}: {e})")
        if ui:
            ui.line("note", "(could not speak that out loud)")
        return False
