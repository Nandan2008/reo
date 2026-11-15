import queue
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

from . import audio as au
from . import config

# 1024 samples: 64ms at 16kHz, and exactly two of the frames the voice detector
# takes. Small enough that the end of a sentence is noticed promptly.
BLOCK_SAMPLES = 1024
BLOCK_SECONDS = BLOCK_SAMPLES / config.SAMPLE_RATE
# Below this, Whisper is guessing at the language rather than recognising it,
# so the guess is not worth keeping for the rest of the session.
LANGUAGE_CONFIDENCE = 0.75
# Below this the device is not quiet, it is handing back digital zero: muted,
# or taken by something else. Worth saying out loud, because "didn't catch
# that" sends him looking for the wrong problem. His Realtek array gates a
# silent room to exactly 0.0 and reads about 0.00003 when muted, against a
# peak near 0.06 when he speaks.
DEAD_DEVICE = 1e-4
# Roughly where his voice sits, measured on this laptop: the room reads about
# 0.0012 and ordinary speech peaks around 0.06.
CLEARLY = 0.030
FAINTLY = 0.012

# "Didn't catch that" covered a muted microphone, a silent room, a mumble and
# whisper inventing words, which need completely different things done about
# them. Each of these is a different problem with a different fix.
MISSED = {
    "silent": "the microphone is giving back silence - check it is not muted",
    "nothing": "nothing was said after the key",
    "noise": "I heard the room, but no voice in it",
    "quiet": "that was too quiet to make out",
    "garbled": "that came back as noise rather than words",
    "unclear": "(didn't catch that)",
}


def _shut(stream) -> None:
    """Close a stream without caring whether it is still alive.

    Handling the failed read was not enough: `with sd.InputStream(...)` stops
    and closes on the way out, and stopping a stream the device has already
    dropped raises "Stream is stopped" a second time. That one escaped and
    still ended the process, after the first had been caught and reported.
    """
    for shut in (getattr(stream, "stop", None), getattr(stream, "close", None)):
        if shut is None:
            continue
        try:
            shut()
        except Exception:
            pass


class Microphone:
    """The input device, opened once and left running.

    Opening one costs half a second on this laptop - 491ms, measured - and
    that half second used to sit between the key press and the first word.
    Whatever was said into it was simply gone, which is one of the ways
    "Hyderabad" came back as "Abad". So the stream now runs continuously and
    the moment *before* the key press is kept as well: by the time recording
    starts, the beginning of the sentence is already in hand.

    Nothing is stored. Outside a recording the audio goes into a ring that
    holds PRE_ROLL_SECONDS and overwrites itself, and it is emptied the
    moment a recording ends. REO_KEEP_MIC_OPEN=0 goes back to opening the
    device for each recording, at the cost of that half second.
    """

    def __init__(self, keep_open: bool = True):
        self.keep_open = keep_open
        self._lock = threading.Lock()
        self._recent = deque(maxlen=max(1, round(config.PRE_ROLL_SECONDS / BLOCK_SECONDS)))
        self._live = None           # a queue while recording, otherwise None
        self._stream = None
        self._reader = None
        self._closing = threading.Event()

    @property
    def running(self) -> bool:
        return self._stream is not None and self._reader is not None and self._reader.is_alive()

    def open(self) -> bool:
        """Start the stream if it is not already going. False means no device."""
        if self.running:
            return True
        _shut(self._stream)
        self._stream = None
        try:
            stream = sd.InputStream(samplerate=config.SAMPLE_RATE, channels=1,
                                    dtype="float32", blocksize=BLOCK_SAMPLES)
            stream.start()
        except Exception as e:
            # No microphone, or something else holds it. Saying so and carrying
            # on beats taking the whole assistant down with us.
            print(f"  [listen] cannot open the microphone: {e}")
            return False
        self._stream = stream
        self._closing.clear()
        self._reader = threading.Thread(target=self._pump, daemon=True,
                                        name="microphone")
        self._reader.start()
        return True

    def _pump(self) -> None:
        """Read the device forever, into the ring or into the recording."""
        stream = self._stream
        while not self._closing.is_set():
            try:
                data, _ = stream.read(BLOCK_SAMPLES)
            except Exception as e:
                # The device can go away mid-sentence: unplugged, grabbed by
                # another program, or the laptop coming back from sleep.
                if not self._closing.is_set():
                    print(f"  [listen] the microphone stopped ({e})")
                break
            block = data.copy()
            with self._lock:
                if self._live is not None:
                    self._live.put(block)
                else:
                    self._recent.append(block)
        _shut(stream)
        if self._stream is stream:
            self._stream = None

    def start(self):
        """Begin a recording. Returns the audio from just before it, or None
        when there is no microphone to record from."""
        if not self.open():
            return None
        with self._lock:
            before = list(self._recent)
            self._recent.clear()
            self._live = queue.Queue()
        return before

    def take(self, timeout: float):
        """The next block of the recording, or None if the device went quiet."""
        live = self._live
        if live is None:
            return None
        try:
            return live.get(timeout=timeout)
        except queue.Empty:
            return None

    def done(self) -> None:
        """End the recording and drop everything that is not already taken."""
        with self._lock:
            self._live = None
            self._recent.clear()
        if not self.keep_open:
            self.close()

    def close(self) -> None:
        self._closing.set()
        reader, self._reader = self._reader, None
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=1.0)
        _shut(self._stream)
        self._stream = None


class Attempt:
    """A transcription running on its own thread, which can be given up on.

    One 2.5 second clip of him took 853 seconds to come back. That did not
    reproduce - the same work measures 0.66s warm and 2.4s after the laptop
    has been sitting idle, so the cost of it swings by three and a half times
    for reasons that have nothing to do with reo - but a voice assistant that
    can wait indefinitely on anything is broken whatever the cause. A plain
    daemon thread rather than a pool because a pool's threads are joined when
    python exits, so one stuck transcription would stop reo closing.
    """

    def __init__(self, work):
        self._box = {}
        self._over = threading.Event()
        threading.Thread(target=self._run, args=(work,), daemon=True,
                         name="transcribe").start()

    def _run(self, work) -> None:
        try:
            self._box["text"] = work()
        except BaseException as e:               # noqa: BLE001 - re-raised below
            self._box["trouble"] = e
        finally:
            self._over.set()

    @property
    def finished(self) -> bool:
        return self._over.is_set()

    def wait(self, seconds: float):
        """The words, or None if it is still going after `seconds`."""
        if not self._over.wait(seconds):
            return None
        if "trouble" in self._box:
            raise self._box["trouble"]
        return self._box["text"]


class Listener:
    # Defaults on the class, so an instance built without __init__ - which the
    # tests do, to get at one method without a model or a microphone - still
    # has them.
    missed = ""              # why the last command came to nothing
    heard_by = ""            # "groq" or "here", so the log says which
    last_level = 0.0         # how loud he was, 0 to 1
    last_voice = 0.0         # seconds of it the detector called speech
    heard_nothing = False    # the device handed back digital zero

    def __init__(self):
        print("Loading speech model (first run downloads it)...")
        self.model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")
        # None means "work it out from what you hear, once". Detection costs
        # about as much again as the transcription itself - measured at 4.5s
        # against 2.2s on the same three second clip - and paying that on every
        # single utterance was the largest part of reo's lag.
        self.language = config.LANGUAGE
        self.detected = False
        self.transcribe_seconds = 0.0    # what the last transcription cost
        self.missed = ""                 # why the last one came to nothing
        self.last_level = 0.0            # how loud he was, 0 to 1
        self.last_voice = 0.0            # seconds the detector called speech
        self._warm_up()
        # Pay the half second for the device now, while he is not waiting on it.
        if self.mic.keep_open:
            self.mic.open()

    # Built on first use, so a Listener made without __init__ - which the tests
    # do, to get at one method without a model or a microphone - still works.
    @property
    def voice(self) -> au.VoiceDetector:
        if getattr(self, "_voice", None) is None:
            self._voice = au.VoiceDetector()
        return self._voice

    @property
    def mic(self) -> Microphone:
        if getattr(self, "_mic", None) is None:
            # In wake mode the wake word has a stream of its own on the same
            # device, so only the hotkey holds the microphone open.
            self._mic = Microphone(keep_open=config.KEEP_MIC_OPEN
                                   and config.MODE == "hotkey")
        return self._mic

    @property
    def _one_at_a_time(self) -> threading.Lock:
        """The model cannot be run twice at once, and a thrown-away attempt
        from the pause may still be finishing when the real one starts."""
        if getattr(self, "_model_lock", None) is None:
            self._model_lock = threading.Lock()
        return self._model_lock

    def _warm_up(self):
        """Transcribe a moment of silence so the first real question does not
        also pay for allocating the model's working memory."""
        try:
            t0 = time.time()
            segments, _ = self.model.transcribe(
                np.zeros(config.SAMPLE_RATE // 2, dtype=np.float32),
                language=self.language or "en", beam_size=1)
            list(segments)
            print(f"Speech model ready ({time.time() - t0:.1f}s).")
        except Exception as e:                      # never block startup on this
            print(f"  [listen] warm-up skipped: {e}")

    def _voiced(self, block, mid_sentence: bool) -> bool:
        """Is someone talking in this block?

        Hysteresis: it takes more to decide that a sentence has begun than to
        decide it is still going, so the quiet tail of a word is not mistaken
        for the end of the sentence.
        """
        chance = self.voice.chance(block)
        if chance < 0:          # no detector; loudness is all there is
            return float(np.sqrt(np.mean(np.asarray(block) ** 2))) > config.SILENCE_THRESHOLD
        return chance >= (au.SPEECH_CONTINUES if mid_sentence else au.SPEECH_STARTS)

    def _record(self, stop_now=None):
        """Capture one utterance. Leaves a head start on self._guess when the
        transcription could be begun during the pause at the end."""
        self._guess = None
        self.missed = "unclear"
        before = self.mic.start()
        if before is None:
            self.missed = "silent"
            return None

        max_blocks = int(config.MAX_RECORD_SECONDS / BLOCK_SECONDS)
        wait_blocks = int(config.WAIT_FOR_SPEECH_SECONDS / BLOCK_SECONDS)
        # A key press will end this, so silence need only catch the case
        # where nobody comes back to it.
        quiet_for = (config.HOTKEY_SILENCE_SECONDS if stop_now is not None
                     else config.SILENCE_SECONDS)
        silence_blocks = max(1, round(quiet_for / BLOCK_SECONDS))
        early_blocks = max(1, round(config.EARLY_GUESS_SECONDS / BLOCK_SECONDS))

        self.voice.reset()
        frames, heard, silent, voiced = [], False, 0, 0
        asked_to_stop = nothing_said = stale = False
        guess = None

        # The moment before the key press is kept, so the start of a word is
        # not lost, and it is fed through the detector so the detector has
        # some context. But it is NOT allowed to decide that he has started
        # talking. It used to be, and that was a bug: anything voice-like in
        # that 0.8s - a television, a throat clear, the tail of reo's own last
        # reply - started the silence countdown immediately, so he had 1.5s to
        # begin instead of six, and then got "didn't catch that". Measured at
        # 1.47s against 6.02s.
        #
        # Nothing is lost by this. If he really was mid-word when he reached
        # for the key, the live blocks that follow are speech too, and heard
        # goes true on the very next one.
        for block in before:
            frames.append(block)
            self._voiced(block, mid_sentence=False)

        try:
            for i in range(max_blocks):
                if stop_now is not None and stop_now():
                    asked_to_stop = True
                    break
                block = self.mic.take(timeout=BLOCK_SECONDS * 8)
                if block is None:
                    if not self.mic.running:
                        asked_to_stop = True
                        break
                    continue
                frames.append(block)
                if self._voiced(block, mid_sentence=heard and not silent):
                    voiced += 1
                    # Talking again, so whatever was transcribed in the pause
                    # is half a sentence. Throw it away.
                    stale = stale or guess is not None
                    heard, silent = True, 0
                elif heard:
                    silent += 1
                    if silent == early_blocks and guess is None:
                        guess = self._start_guess(frames)
                    if silent >= silence_blocks:
                        break
                elif i >= wait_blocks:
                    nothing_said = True
                    break
            else:
                if heard:
                    print(f"  (stopped at the {config.MAX_RECORD_SECONDS} second limit)")
        finally:
            self.mic.done()

        self.last_voice = voiced * BLOCK_SECONDS
        if nothing_said or not frames:
            self.missed = "nothing" if nothing_said else "unclear"
            return None
        # When they pressed the key to stop, transcribe what was captured even if
        # no voice was ever found in it. A quiet microphone kept "heard" false,
        # so every deliberate recording was binned as "didn't catch that".
        if not (heard or (asked_to_stop and len(frames) * BLOCK_SECONDS >= 0.4)):
            self.missed = "noise"
            return None
        if guess is not None and not stale:
            self._guess = guess
        return np.concatenate(frames).flatten()

    def _start_guess(self, frames):
        """Start transcribing while the pause is still being waited out.

        Recognition costs about 0.6s whatever the length - Whisper pads
        everything to thirty seconds - and that 0.6s used to be added on at
        the end, after the wait. Begun when the pause begins, it is finished
        before the wait is, and costs nothing at all. If he starts talking
        again it is thrown away and the whole thing is done properly.
        """
        audio = np.concatenate(frames).flatten()
        return Attempt(lambda: self._transcribe(audio))

    def _ears(self):
        """The client for hosted hearing, built once and kept.

        Built per call it cost 2.69s on a cold one against 0.20s warm, which
        is the whole saving thrown away.
        """
        if getattr(self, "_ears_client", None) is None:
            from openai import OpenAI

            from . import providers
            host = providers.hosting(config.HOSTED_WHISPER)
            if host is None:
                return None
            self._ears_client = OpenAI(api_key=host.key, base_url=host.base_url,
                                       timeout=config.HOSTED_HEARING_TIMEOUT,
                                       max_retries=0)
        return self._ears_client

    def _hosted(self, clean):
        """Words from Groq, or None - which means do it here instead.

        None rather than an exception on purpose: no network, a spent key and
        a bad day at Groq are all the same thing from here, and all of them
        are answered by the model already sitting in memory.
        """
        # Only a language he pinned himself, never the sticky guess. That
        # guess exists because working it out locally costs about as much as
        # the transcription; Groq does it for nothing, and does it better.
        # Passing the guess would force his Telugu through an English model
        # for the rest of the session because the first thing he said was
        # in English.
        extra = {"language": config.LANGUAGE} if config.LANGUAGE else {}
        try:
            # Inside the try on purpose: building the client can fail too -
            # no key, a bad url, a missing import - and that is just as good
            # a reason to use the model here as the request failing.
            client = self._ears()
            if client is None:
                return None
            answer = client.audio.transcriptions.create(
                model=config.HOSTED_WHISPER,
                file=("speech.wav", au.as_wav(clean), "audio/wav"),
                response_format="verbose_json",
                prompt=config.VOCABULARY or None, **extra)
        except Exception as e:
            print(f"  [listen] groq could not hear that ({type(e).__name__}); "
                  f"using the model here")
            return None
        segments = getattr(answer, "segments", None) or []
        if not segments:
            return (getattr(answer, "text", "") or "").strip()
        # Same filter as the local path: whisper rates how likely each segment
        # is to be silence, and taking its word for that costs nothing.
        kept = []
        for segment in segments:
            get = segment.get if isinstance(segment, dict) else (
                lambda k, d=None: getattr(segment, k, d))
            if float(get("no_speech_prob", 0.0) or 0.0) <= config.NO_SPEECH_PROB:
                kept.append(str(get("text", "") or ""))
        return " ".join(kept).strip()

    def _transcribe(self, raw) -> str:
        """Audio in, words out. Everything that is not words is dropped here."""
        # Whisper does not return nothing for silence - it invents. Only needed
        # when the voice detector is missing; otherwise this was already decided
        # block by block while recording.
        if not self.voice.ready and not au.has_speech(raw, config.SPEECH_FLOOR):
            print("  (only room noise)")
            return ""
        clean = au.normalize(raw)
        text = self._hosted(clean) if config.HEARING == "hosted" else None
        self.heard_by = "groq"
        if text is None:
            self.heard_by = "here"
            text = self._here(clean)
        if au.looks_hallucinated(text):
            self.missed = "garbled"
            return ""
        return text

    def _here(self, clean) -> str:
        """The model on this machine. The fallback, and the whole thing when
        REO_HEARING=local."""
        if not self._one_at_a_time.acquire(timeout=config.TRANSCRIBE_TIMEOUT):
            print("  [listen] the speech model is still busy with the last one")
            return ""
        try:
            if self.language is None:
                self._choose_language(clean)
            segments, _ = self.model.transcribe(
                clean, language=self.language, beam_size=config.BEAM_SIZE,
                hotwords=config.VOCABULARY or None, vad_filter=True, **au.CAREFUL)
            # Whisper rates how likely each segment is to be silence. Taking
            # its word for that costs nothing and catches the inventions that
            # get past the loudness gate.
            text = " ".join(
                s.text for s in segments
                if getattr(s, "no_speech_prob", 0.0) <= config.NO_SPEECH_PROB).strip()
        finally:
            self._one_at_a_time.release()
        return text

    def listen(self, stop_now=None) -> str:
        audio = self._record(stop_now)
        self.transcribe_seconds = 0.0
        self.heard_nothing = False
        if audio is None:
            self.last_level = 0.0
            return ""
        self.last_level = float(np.max(np.abs(audio)))
        if self.last_level < DEAD_DEVICE:
            # Not a quiet room: nothing at all. Whisper invents words out of
            # this, and no amount of listening harder will help.
            self.heard_nothing = True
            self.missed = "silent"
            print("  [listen] the microphone gave back silence - is it muted?")
            return ""

        guess, self._guess = getattr(self, "_guess", None), None
        started = time.time()
        if guess is None:
            guess = Attempt(lambda: self._transcribe(audio))
        # What he actually waited for. When the pause did the work this is a
        # few milliseconds, because the words were ready before he asked.
        text = guess.wait(config.TRANSCRIBE_TIMEOUT)
        self.transcribe_seconds = time.time() - started
        if text is None:
            print(f"  [listen] gave up on the transcription after "
                  f"{config.TRANSCRIBE_TIMEOUT:.0f}s")
            return ""

        if not text:
            # Nothing usable came back. If the language was a guess, it may have
            # been the wrong guess, so let the next utterance decide again.
            if self.missed in ("", "unclear"):
                self.missed = ("quiet" if self.last_level < FAINTLY else "unclear")
            self._forget_language()
            return ""
        self.missed = ""
        return text

    def how_i_hear_you(self) -> str:
        """What the last thing he said looked like from in here.

        Measured on the recording he has just made rather than asking him to
        say something else: it is the real thing, at his real distance, with
        whatever the room was doing at the time.
        """
        level, voice = self.last_level, self.last_voice
        if level < DEAD_DEVICE:
            return ("The microphone is giving me digital silence. Check it is "
                    "not muted, and that nothing else has taken it.")
        if not voice:
            return (f"I can hear the room, but found no voice in it. The level "
                    f"was {level:.3f}, where speech is usually around 0.06.")
        if level < FAINTLY:
            return (f"Only just. You came through at {level:.3f}, where I expect "
                    f"about 0.06. Try closer, or turn the microphone up.")
        if level < CLEARLY:
            return (f"Quietly, but I got you: {level:.3f}, with {voice:.1f} "
                    f"seconds of voice in it.")
        return (f"Clearly. {level:.3f}, with {voice:.1f} seconds of voice in it. "
                f"Nothing wrong with the microphone.")

    def _choose_language(self, audio) -> None:
        """Work out the language once, choosing only among the ones spoken here.

        Left to choose freely, Whisper read this accent as Russian and Spanish
        and wrote confident nonsense. Ranking only the configured languages
        cannot produce a language the user does not speak.
        """
        try:
            _, _, every = self.model.detect_language(audio, vad_filter=True)
        except Exception as e:
            print(f"  [listen] could not detect the language ({e}); assuming English")
            self.language, self.detected = "en", True
            return
        allowed = set(config.LANGUAGES)
        ranked = sorted(((p, code) for code, p in every if code in allowed), reverse=True)
        if not ranked:
            return      # nothing configured matched; let whisper decide this once
        confidence, best = ranked[0]
        if confidence >= LANGUAGE_CONFIDENCE:
            self.language, self.detected = best, True
            print(f"  [listen] language is {best} ({confidence:.0%}); reusing it")

    def _forget_language(self) -> None:
        if self.detected:
            self.language, self.detected = None, False
