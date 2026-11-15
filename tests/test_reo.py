"""Offline tests: no microphone, no API key, no network."""
import io
import re
import shutil
import tempfile
import threading
import time
import unittest
import urllib.parse
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from openai import APIConnectionError, APIStatusError, RateLimitError

from reo import config, fastpath, providers, tools
from reo import memory as memory_module
from reo import brain as brain_module
from reo import pointer
from reo import workspace
from reo import system as reo_system
from reo.memory import Memory
from reo.audio import has_speech, looks_hallucinated
from reo.audio import normalize as normalize_gain
from reo import __main__ as main
from reo import speak as speak_module
from reo.__main__ import EXIT_PHRASES
from reo.text import is_speech, normalize
from reo.brain import KNOWN_TOOLS, SYSTEM, Brain, OutOfAllowance
from reo import audio as au_mod
from reo import listen
from reo.audio import VoiceDetector
from reo.listen import Attempt, Listener, Microphone
from reo.speak import VOICE_SETTINGS, split_emotion



def _diagnose_clip():
    """Six seconds of him, saved by diagnose.py. Not in the repository."""
    import wave
    path = Path(__file__).resolve().parent.parent / "diagnose.wav"
    if not path.exists():
        return None
    with wave.open(str(path), "rb") as w:
        rate, channels = w.getframerate(), w.getnchannels()
        raw = w.readframes(w.getnframes())
    clip = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        clip = clip.reshape(-1, channels).mean(axis=1)
    return clip if rate == 16000 else None


class ExitPhrases(unittest.TestCase):
    """Whisper punctuates inside phrases, which used to defeat the lookup."""

    def test_exits(self):
        for said in [" Goodbye.", " Bye!", "Goodbye, Reo.", " Bye, Reo.",
                     "  BYE,  REO.  ", " Stop listening.", " Exit", " Quit!"]:
            with self.subTest(said=said):
                self.assertIn(normalize(said), EXIT_PHRASES)

    def test_keeps_going(self):
        for said in [" Thanks, goodbye.", " Say goodbye to John.", " Goodbye forever.",
                     " What time is it?", " Open notepad."]:
            with self.subTest(said=said):
                self.assertNotIn(normalize(said), EXIT_PHRASES)


class WokenBy(unittest.TestCase):
    """The prompt must name whatever actually wakes it."""

    def test_follows_the_engine(self):
        from unittest import mock
        from reo.__main__ import woken_by
        with mock.patch.object(config, "WAKE_ENGINE", "phrase"),              mock.patch.object(config, "WAKE_PHRASE", "hey_jarvis"):
            self.assertEqual(woken_by(), "hey jarvis")
        with mock.patch.object(config, "WAKE_ENGINE", "whisper"),              mock.patch.object(config, "WAKE_NAME", "reo"):
            self.assertEqual(woken_by(), "reo")


class NoiseFilter(unittest.TestCase):
    def test_rejects_noise(self):
        for said in ["", ".", " . ", "...", " ?!", "  "]:
            with self.subTest(said=said):
                self.assertFalse(is_speech(said))

    def test_accepts_speech(self):
        for said in [" Thank you.", "what time is it", "7"]:
            with self.subTest(said=said):
                self.assertTrue(is_speech(said))


class EmotionTags(unittest.TestCase):
    def test_tag_is_stripped_and_read(self):
        self.assertEqual(split_emotion("[happy] All done."), ("happy", "All done."))
        self.assertEqual(split_emotion("[excited]No space."), ("excited", "No space."))
        self.assertEqual(split_emotion("  [sad]  Padded.  "), ("sad", "Padded."))
        self.assertEqual(split_emotion("[APOLOGETIC] Shouty."), ("apologetic", "Shouty."))

    def test_unknown_tag_is_still_removed(self):
        """An unknown tag must never be spoken aloud."""
        self.assertEqual(split_emotion("[banana] Hello."), ("neutral", "Hello."))

    def test_doubled_reply_is_said_once(self):
        """gpt-oss-120b sometimes repeats its whole answer, tag and all."""
        self.assertEqual(
            split_emotion("[neutral] It's 11:03 AM.[neutral] It's 11:03 AM."),
            ("neutral", "It's 11:03 AM."))

    def test_genuine_repetition_survives(self):
        self.assertEqual(split_emotion("[neutral] Yes. No. Yes.")[1], "Yes. No. Yes.")

    def test_no_tag(self):
        self.assertEqual(split_emotion("Just words."), ("neutral", "Just words."))

    def test_prompt_and_voice_table_agree(self):
        """Every tag the brain is told to use needs a voice setting, or it goes flat."""
        promised = set(re.findall(r"\[(\w+)]", SYSTEM))
        self.assertTrue(promised, "no tags found in the system prompt")
        self.assertEqual(promised - set(VOICE_SETTINGS), set())


class WakeWord(unittest.TestCase):
    """What follows the name is the command; everything else is overheard chatter."""

    @staticmethod
    def _match(heard):
        from reo.wake import WakeListener
        return WakeListener._command_after_wake(heard)

    def test_command_said_with_the_name(self):
        self.assertEqual(self._match("Reo, what time is it?"), "what time is it")
        self.assertEqual(self._match(" reo open notepad"), "open notepad")

    def test_name_alone_returns_empty(self):
        self.assertEqual(self._match("Reo?"), "")
        self.assertEqual(self._match(" Reo."), "")

    def test_mishearings_still_wake(self):
        for heard in ["Rio, what time is it?", "Rayo what time is it"]:
            with self.subTest(heard=heard):
                self.assertEqual(self._match(heard), "what time is it")

    def test_not_addressed_is_ignored(self):
        for heard in ["what time is it?", "I was talking to Sam about the weather", ""]:
            with self.subTest(heard=heard):
                self.assertIsNone(self._match(heard))

    def test_name_mid_sentence_still_counts(self):
        self.assertEqual(self._match("okay reo open notepad"), "open notepad")


class Hallucinations(unittest.TestCase):
    """Whisper loops a phrase when there is too little real speech. These are
    verbatim from reo's own logs on a weak microphone."""

    def test_repeats_are_rejected(self):
        for text in [
            "I'm going to put it in the middle.  I'm going to put it in the middle.  I'm going to put it in the middle.",
            "That's what it is.  That's what it is.  That's what it is.",
            "Hello how are you? Hello how are you?",
        ]:
            with self.subTest(text=text[:30]):
                self.assertTrue(looks_hallucinated(text))

    def test_real_speech_survives(self):
        for text in ["Reo, what time is it?", "open notepad",
                     "I'm fine. How are you?", "", "Yes."]:
            with self.subTest(text=text):
                self.assertFalse(looks_hallucinated(text))


class Gain(unittest.TestCase):
    def test_quiet_audio_is_lifted(self):
        """0.1 is about what a laptop mic gives; 5x gain is within the cap."""
        quiet = np.full(100, 0.1, dtype=np.float32)
        self.assertAlmostEqual(float(np.max(np.abs(normalize_gain(quiet)))), 0.5, places=3)

    def test_silence_is_left_alone(self):
        silent = np.zeros(100, dtype=np.float32)
        self.assertEqual(float(np.max(np.abs(normalize_gain(silent)))), 0.0)

    def test_gain_is_capped(self):
        """Amplifying near-silence just makes louder noise."""
        nearly = np.full(100, 0.001, dtype=np.float32)
        self.assertLess(float(np.max(np.abs(normalize_gain(nearly)))), 0.5)


class Typing(unittest.TestCase):
    """Decided without touching a real window: guessing typed into the wrong
    Notepad tab once, and into an audio control panel once."""

    @staticmethod
    def _type(powershell_says, text="hello", window="notepad"):
        from unittest import mock
        from reo import keys
        with mock.patch.object(keys, "_powershell", return_value=powershell_says):
            return keys.type_text(text, window)

    def test_several_matches_types_nothing(self):
        out = self._type("MANY:Notes - Notepad|config - Notepad")
        self.assertIn("nothing was sent", out)
        self.assertIn("Notes - Notepad", out)

    def test_enter_is_appended_when_asked(self):
        """Typing without sending was the complaint: it typed, nothing submitted."""
        from unittest import mock
        from reo import keys
        with mock.patch.object(keys, "_powershell", return_value="TYPED:Chrome") as ps:
            keys.type_text("hello", "chrome", enter=True)
        self.assertTrue(ps.call_args[0][1]["REO_TEXT"].endswith("{ENTER}"))

        with mock.patch.object(keys, "_powershell", return_value="TYPED:Chrome") as ps:
            keys.type_text("hello", "chrome")
        self.assertFalse(ps.call_args[0][1]["REO_TEXT"].endswith("{ENTER}"))

    def test_keystrokes_understood(self):
        from reo.keys import keystroke
        self.assertEqual(keystroke("enter"), "{ENTER}")
        self.assertEqual(keystroke("send"), "{ENTER}")
        self.assertEqual(keystroke("ctrl+a"), "^a")
        self.assertEqual(keystroke("alt+f4"), "%{F4}")

    def test_unknown_key_presses_nothing(self):
        from reo.keys import press_key
        self.assertIn("not a key reo knows", press_key("banana", "notepad"))

    def test_one_match_is_typed(self):
        self.assertIn("Notes - Notepad", self._type("TYPED:Notes - Notepad"))

    def test_no_window_named_types_nothing(self):
        from unittest import mock
        from reo import keys
        with mock.patch.object(keys, "list_windows", return_value="Notepad"):
            self.assertIn("Say which window", keys.type_text("hello"))

    def test_sendkeys_specials_are_escaped(self):
        from reo.keys import escape
        self.assertEqual(escape("50% (a)+b"), "50{%} {(}a{)}{+}b")
        self.assertEqual(escape("plain text"), "plain text")


class Recording(unittest.TestCase):
    """A quiet mic made every deliberate recording come back
    "(didn't catch that)", because the loudness flag never flipped."""

    @staticmethod
    def _record(level, stop_after):
        import time
        from unittest import mock
        from reo.listen import Listener

        block = np.full((1600, 1), level, dtype=np.float32)
        started = time.monotonic()

        class FakeStream:
            def start(self): pass
            def stop(self): pass
            def close(self): pass
            def read(self, n):
                time.sleep(0.01)
                return block, None

        listener = Listener.__new__(Listener)
        with mock.patch("reo.listen.sd.InputStream", lambda **kw: FakeStream()):
            return listener._record(
                stop_now=lambda: time.monotonic() - started > stop_after)

    def test_quiet_speech_is_kept_when_you_press_stop(self):
        self.assertIsNotNone(self._record(level=0.002, stop_after=0.6))

    def test_nothing_recorded_is_still_discarded(self):
        self.assertIsNone(self._record(level=0.002, stop_after=0.0))


class FastPath(unittest.TestCase):
    """Answering these through the model cost about two seconds each."""

    def test_answers_what_the_machine_knows(self):
        from reo.fastpath import answer
        for asked in ["what time is it", "time", "whats the date", "battery",
                      "how much battery do i have", "how much ram am i using",
                      "disk space", "uptime", "how long has my laptop been on"]:
            with self.subTest(asked=asked):
                self.assertIsNotNone(answer(asked))

    def test_does_the_arithmetic(self):
        from reo.fastpath import answer
        self.assertEqual(answer("what is 25 times 17"), "425.")
        self.assertEqual(answer("whats 100 divided by 4"), "25.")
        self.assertEqual(answer("15 percent of 200"), "30.")
        self.assertEqual(answer("what is the square root of 144"), "12.")

    def test_leaves_the_model_its_work(self):
        """A keyword search would have hijacked every one of these."""
        from reo.fastpath import answer
        for asked in ["what time is it in Tokyo", "what time does the shop close",
                      "how is my battery health compared to last year",
                      "remind me to charge the battery",
                      "how much disk space does Chrome use",
                      "tell me about the date of the moon landing",
                      "calculate the risk of this plan", "what is love"]:
            with self.subTest(asked=asked):
                self.assertIsNone(answer(asked))

    def test_arithmetic_cannot_run_code(self):
        from reo.fastpath import answer
        for asked in ["what is __import__('os').system('calc')",
                      "calculate open('x','w')",
                      "what is 2 times len('abc')"]:
            with self.subTest(asked=asked):
                self.assertIsNone(answer(asked))


class SecondCopy(unittest.TestCase):
    """Four copies were found running at once. They fight over the sound device."""

    @staticmethod
    def _with_processes(entries):
        from unittest import mock
        import psutil
        fakes = [mock.Mock(info={"pid": pid, "name": name, "cmdline": cmd})
                 for pid, name, cmd in entries]
        return mock.patch.object(psutil, "process_iter", return_value=fakes)

    def test_spots_another_reo(self):
        import os
        from reo.__main__ import already_running
        with self._with_processes([(os.getpid() + 1, "python.exe", ["python.exe", "-m", "reo"])]):
            self.assertTrue(already_running())

    def test_ignores_itself(self):
        import os
        from reo.__main__ import already_running
        with self._with_processes([(os.getpid(), "python.exe", ["python.exe", "-m", "reo"])]):
            self.assertFalse(already_running())

    def test_ignores_unrelated_python(self):
        import os
        from reo.__main__ import already_running
        with self._with_processes([(os.getpid() + 1, "python.exe", ["python.exe", "manage.py"])]):
            self.assertFalse(already_running())

    def test_working_directory_is_not_a_running_reo(self):
        """This false positive refused to start reo when nothing was running."""
        import os
        from reo.__main__ import already_running
        with self._with_processes([(os.getpid() + 1, "python.exe",
                                    ["python.exe", "-m", "pip", "install",
                                     r"C:\Users\kollu\OneDrive\Desktop\reo"])]):
            self.assertFalse(already_running())

    def test_ignores_non_python_processes(self):
        import os
        from reo.__main__ import already_running
        with self._with_processes([(os.getpid() + 1, "powershell.exe",
                                    ["powershell.exe", "-m", "reo"])]):
            self.assertFalse(already_running())


class Playback(unittest.TestCase):
    """sd.wait() has no timeout. One hang ran to 72,304 seconds."""

    @staticmethod
    def _play(plays, until=None, grace=0.3):
        """Play a short clip with a fake sound card.

        `plays` decides whether the card actually takes the samples; when it
        does not, nothing finishes and reo has to give up on its own.
        """
        audio = np.full(1600, 500, dtype=np.int16)    # 0.1s at 16kHz

        def card(rate):
            if plays:
                threading.Thread(target=_drain, daemon=True).start()
            return SimpleNamespace(active=True)

        def _drain():
            time.sleep(0.02)
            out = np.zeros((1600, 1), dtype=np.int16)
            for _ in range(6):
                speak_module._feed(out, 400, None, None)
                time.sleep(0.01)

        with mock.patch.object(speak_module, "_speaker", card), \
             mock.patch.object(speak_module, "PLAY_GRACE", grace):
            return speak_module._play(audio, 16000, until)

    def test_returns_when_the_clip_finishes(self):
        began = time.time()
        self.assertFalse(self._play(plays=True))
        self.assertLess(time.time() - began, 2.0)

    def test_gives_up_when_the_card_never_takes_it(self):
        """sd.wait() had no timeout at all, and one hang ran to 72,304s."""
        began = time.time()
        self.assertFalse(self._play(plays=False, grace=0.2))
        self.assertLess(time.time() - began, 3.0)

    def test_a_press_stops_it_where_it_got_to(self):
        self.assertTrue(self._play(plays=False, until=lambda: True))

    def test_the_speaker_is_not_reopened_for_every_reply(self):
        """Opening one costs 287ms, measured, and it was being paid on every
        single reply - by sd.play() as much as anything."""
        made = []

        class Stream:
            def __init__(self, **kw): made.append(kw.get("samplerate"))
            active = True
            def start(self): pass
            def stop(self): pass
            def close(self): pass

        with mock.patch.object(speak_module.sd, "OutputStream", Stream):
            speak_module.close()
            speak_module._speaker(24000)
            speak_module._speaker(24000)
            speak_module._speaker(24000)
            self.assertEqual(made, [24000], "reopened the device")
            speak_module._speaker(16000)        # a different rate does rebuild
            self.assertEqual(made, [24000, 16000])
            speak_module.close()


class ArrivingAudio(unittest.TestCase):
    """The clip is spoken while it is still downloading: the first bytes land
    0.20s in and the last at 0.74s, so two thirds of that wait was transfer."""

    @staticmethod
    def _wav(samples, rate=24000):
        import struct
        body = samples.astype(np.int16).tobytes()
        return (b"RIFF" + struct.pack("<I", 36 + len(body)) + b"WAVE"
                + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
                + b"data" + struct.pack("<I", len(body)) + body)

    def test_the_header_is_read_before_it_has_all_arrived(self):
        whole = self._wav(np.full(4000, 300))
        self.assertIsNone(speak_module._wav_shape(whole[:8]))
        rate, channels, bits, start = speak_module._wav_shape(whole[:60])
        self.assertEqual((rate, channels, bits), (24000, 1, 16))
        self.assertEqual(start, 44)

    def test_rubbish_is_not_mistaken_for_a_wav(self):
        self.assertIsNone(speak_module._wav_shape(b"<html>not audio</html>"))

    def test_the_pause_at_the_front_is_skipped(self):
        clip = np.zeros(2000, dtype=np.int16)
        clip[800:] = 4000
        self.assertGreater(speak_module._first_sound(clip), 700)
        self.assertEqual(speak_module._first_sound(np.zeros(10, dtype=np.int16)), 0)

    def test_it_plays_what_arrives_in_pieces(self):
        samples = np.concatenate([np.zeros(600), np.full(9000, 2500),
                                  np.zeros(3000)]).astype(np.int16)
        whole = self._wav(samples)
        pieces = [whole[i:i + 900] for i in range(0, len(whole), 900)]
        taken = []

        def card(rate):
            def drain():
                out = np.zeros((2048, 1), dtype=np.int16)
                for _ in range(80):
                    speak_module._feed(out, 512, None, None)
                    taken.append(1)
                    time.sleep(0.004)
            threading.Thread(target=drain, daemon=True).start()
            return SimpleNamespace(active=True)

        with mock.patch.object(speak_module, "_speaker", card):
            cut = speak_module._play_arriving(iter(pieces))
        self.assertFalse(cut)
        self.assertTrue(taken)
        # the silence on the end was trimmed off rather than played out
        self.assertLess(speak_module._clip["end"], len(samples))
        self.assertGreater(speak_module._clip["end"], 9000)

    def test_a_response_with_no_wav_in_it_is_refused(self):
        """So the caller can fall back to asking for the whole file."""
        with self.assertRaises(speak_module._NoAudio):
            speak_module._play_arriving(iter([b"not a wav at all"]))


class ShellGuards(unittest.TestCase):
    """Voice mishears and web pages can steer a model, so these matter.

    The strings below are never executed: every one is asserted to be stopped
    before it reaches a shell.
    """

    @staticmethod
    def _run(command, confirmed=False):
        from unittest import mock
        from reo import shell
        with mock.patch.object(config, "ALLOW_SHELL", True):
            return shell.run_command(command, confirmed)

    def test_off_unless_turned_on(self):
        from unittest import mock
        from reo import shell
        with mock.patch.object(config, "ALLOW_SHELL", False):
            self.assertIn("switched off", shell.run_command("Get-Date"))

    def test_catastrophic_refused_even_when_confirmed(self):
        for command in ["format " + "c:", "disk" + "part",
                        "Remove-Item -Recurse -Force C:" + chr(92) + "Windows",
                        "vssadmin delete shadows /all"]:
            with self.subTest(command=command):
                self.assertTrue(self._run(command, confirmed=True).startswith("Refused"))

    def test_destructive_asks_first(self):
        for command in ["Remove-Item C:" + chr(92) + "temp" + chr(92) + "a.txt",
                        "shutdown /s /t 0",
                        "Stop-Process -Name notepad"]:
            with self.subTest(command=command):
                self.assertTrue(self._run(command).startswith("This would"))

    def test_harmless_is_not_gated(self):
        out = self._run("Get-Date -Format yyyy")
        self.assertFalse(out.startswith(("Refused", "This would")))


class ToolFailures(unittest.TestCase):
    """run_tool must hand failures back to the model, never raise."""

    def test_unknown_app_says_so(self):
        """Decided from a stubbed PowerShell: a test must not launch programs.
        The version of this test that called through really did open Spotify."""
        from unittest import mock
        from reo import apps
        with mock.patch.object(apps, "_powershell", return_value="NOTFOUND"):
            self.assertIn("Nothing installed matches", apps.open_app("nosuchapp"))

    def test_several_apps_match_opens_nothing(self):
        from unittest import mock
        from reo import apps
        with mock.patch.object(apps, "_powershell", return_value="MANY:Word|WordPad"):
            out = apps.open_app("word")
        self.assertIn("nothing was opened", out)
        self.assertIn("WordPad", out)

    def test_unknown_tool(self):
        self.assertIn("Unknown tool", tools.run_tool("fly_to_moon", {}))

    def test_bad_arguments_do_not_raise(self):
        for name, args in [("open_app", {"app": "notepad"}),
                           ("search_web", {}),
                           ("get_datetime", {"tz": "UTC"})]:
            with self.subTest(name=name):
                self.assertIn("failed", tools.run_tool(name, args))

    def test_readonly_tools_answer(self):
        self.assertTrue(tools.get_datetime())
        self.assertTrue(tools.get_battery())


class HistoryTrimming(unittest.TestCase):
    """Trimming must not orphan a tool call from its result."""

    @staticmethod
    def _conversation(turns):
        history = []
        for i in range(turns):
            history += [
                {"role": "user", "content": f"q{i}"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": f"t{i}", "type": "function",
                     "function": {"name": "get_datetime", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": f"t{i}", "content": "now"},
                {"role": "assistant", "content": f"a{i}"},
            ]
        return history

    def _trimmed(self, turns):
        brain = Brain.__new__(Brain)          # no API key needed
        brain.history = self._conversation(turns)
        brain._trim_history()
        return brain.history

    def test_starts_on_a_user_message(self):
        self.assertEqual(self._trimmed(6)[0]["role"], "user")

    def test_tool_calls_keep_their_results(self):
        history = self._trimmed(6)
        for i, message in enumerate(history):
            for call in message.get("tool_calls", []):
                later = [m.get("tool_call_id") for m in history[i + 1:]]
                self.assertIn(call["id"], later)

    def test_short_history_untouched(self):
        self.assertEqual(len(self._trimmed(1)), 4)


class ProviderPool(unittest.TestCase):
    """More than one key is the answer to a per-minute token limit."""

    ENV = {
        "LLM_API_KEY": "groq-one",
        "LLM_BASE_URL": "https://api.groq.com/openai/v1",
        "LLM_MODEL": "openai/gpt-oss-120b",
        "LLM_API_KEYS": "groq-two, groq-three",
        "LLAMA_API_KEY": "llama-one",
        "OPENROUTER_API_KEY": "",
    }

    def test_configured_key_goes_first(self):
        found = providers.build(self.ENV)
        self.assertEqual(found[0].key, "groq-one")
        self.assertEqual(found[0].model, "openai/gpt-oss-120b")

    def test_spares_share_the_service(self):
        found = providers.build(self.ENV)
        spares = [p for p in found if p.key.startswith("groq-")]
        self.assertEqual([p.key for p in spares], ["groq-one", "groq-two", "groq-three"])
        self.assertEqual({p.base_url for p in spares}, {self.ENV["LLM_BASE_URL"]})

    def test_known_providers_come_after(self):
        with mock.patch.object(providers, "local", lambda env=None: None):
            found = providers.build(self.ENV)
        self.assertEqual(found[-1].name, "llama")
        self.assertIn("api.llama.com", found[-1].base_url)

    def test_the_local_model_comes_after_even_those(self):
        """It is the last resort, not the first choice."""
        here = providers.Provider("ollama", "x", "http://localhost:11434/v1", "q:7b")
        with mock.patch.object(providers, "local", lambda env=None: here):
            found = providers.build(self.ENV)
        self.assertEqual(found[-1].name, "ollama")
        self.assertEqual(found[-2].name, "llama")

    def test_blank_keys_are_not_providers(self):
        names = [p.name for p in providers.build(self.ENV)]
        self.assertNotIn("openrouter", names)

    def test_the_same_key_is_not_listed_twice(self):
        env = dict(self.ENV, GROQ_API_KEY="groq-one")   # same key, known slot
        keys = [p.key for p in providers.build(env)]
        self.assertEqual(len(keys), len(set(keys)))

    def test_no_keys_and_nothing_local_means_no_providers(self):
        with mock.patch.object(providers, "local", lambda env=None: None):
            self.assertEqual(providers.build({"LLM_API_KEY": ""}), [])

    def test_a_local_model_is_enough_on_its_own(self):
        """No key at all, but something running here: reo still works, and
        nothing it is asked ever leaves the laptop."""
        here = providers.Provider("ollama", "x", "http://localhost:11434/v1", "q:7b")
        with mock.patch.object(providers, "local", lambda env=None: here):
            found = providers.build({"LLM_API_KEY": ""})
        self.assertEqual([p.name for p in found], ["ollama"])

    def test_a_benched_key_is_skipped(self):
        pool = providers.Pool(providers.build(self.ENV))
        self.assertEqual(pool.current()[0], 0)
        pool.bench(0)
        self.assertEqual(pool.current()[0], 1)
        self.assertEqual(pool.free(), len(pool) - 1)

    def test_a_bench_expires(self):
        pool = providers.Pool(providers.build(self.ENV))
        pool.bench(0, seconds=-1)        # already served its minute
        self.assertEqual(pool.current()[0], 0)

    def test_all_benched_still_offers_the_soonest(self):
        """The cooldown is a guess, so refusing to answer would be worse."""
        pool = providers.Pool(providers.build(self.ENV))
        for i in range(len(pool)):
            pool.bench(i, seconds=100 - i)
        self.assertEqual(pool.current()[0], len(pool) - 1)
        self.assertEqual(pool.free(), 0)

    def test_the_voice_stays_with_groq(self):
        """Orpheus is hosted by Groq alone, whatever is doing the thinking."""
        env = {"LLM_API_KEY": "llama-key", "LLM_BASE_URL": "https://api.llama.com/compat/v1",
               "GROQ_API_KEY": "groq-key"}
        host = providers.hosting("canopylabs/orpheus-v1-english", env)
        self.assertEqual(host.key, "groq-key")

    def test_no_groq_key_means_no_orpheus(self):
        env = {"LLM_API_KEY": "llama-key", "LLM_BASE_URL": "https://api.llama.com/compat/v1"}
        self.assertIsNone(providers.hosting("canopylabs/orpheus-v1-english", env))


def _refusal(status):
    """An openai error with a given status, built without a real HTTP round trip.
    A SimpleNamespace is enough: the client only reads these three attributes."""
    response = SimpleNamespace(status_code=status, request=SimpleNamespace(), headers={})
    kind = RateLimitError if status == 429 else APIStatusError
    return kind("busy", response=response, body=None)


class _Client:
    """Stands in for an OpenAI client: does whatever the test tells it to."""

    def __init__(self, outcome):
        self.calls = 0

        def create(**kwargs):
            self.calls += 1
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


class KeyFailover(unittest.TestCase):
    """A spent key should cost a retry, not an apology out loud."""

    def _brain(self, outcomes):
        pool = providers.Pool([
            providers.Provider(f"p{i}", f"k{i}", f"https://p{i}.invalid", "m")
            for i in range(len(outcomes))
        ])
        brain = Brain.__new__(Brain)
        brain.history = [{"role": "user", "content": "hello"}]
        brain.pool = pool
        clients = [_Client(o) for o in outcomes]
        brain._clients = dict(enumerate(clients))
        brain._client = lambda index, provider: clients[index]
        return brain, pool, clients

    def test_a_rate_limit_moves_to_the_next_key(self):
        answer = SimpleNamespace(choices=[])
        brain, pool, clients = self._brain([_refusal(429), answer])
        self.assertIs(brain._complete(with_tools=False), answer)
        self.assertEqual([c.calls for c in clients], [1, 1])
        self.assertTrue(pool.benched(0))

    def test_the_spent_key_is_not_asked_again(self):
        answer = SimpleNamespace(choices=[])
        brain, pool, clients = self._brain([_refusal(429), answer])
        brain._complete(with_tools=False)
        brain._complete(with_tools=False)
        self.assertEqual(clients[0].calls, 1)      # benched, so skipped
        self.assertEqual(clients[1].calls, 2)

    def test_every_key_refusing_raises(self):
        brain, _, clients = self._brain([_refusal(429), _refusal(429)])
        with self.assertRaises(RuntimeError):
            brain._complete(with_tools=False)
        self.assertEqual([c.calls for c in clients], [1, 1])

    def test_our_own_bad_request_is_not_retried_elsewhere(self):
        """A 400 fails identically on every key; spending them all hides the bug."""
        brain, _, clients = self._brain([_refusal(400), SimpleNamespace(choices=[])])
        with self.assertRaises(APIStatusError):
            brain._complete(with_tools=False)
        self.assertEqual(clients[1].calls, 0)

    def test_a_server_error_tries_the_next_key(self):
        answer = SimpleNamespace(choices=[])
        brain, _, clients = self._brain([_refusal(503), answer])
        self.assertIs(brain._complete(with_tools=False), answer)
        self.assertEqual(clients[1].calls, 1)

    def test_no_keys_at_all_says_so(self):
        brain = Brain.__new__(Brain)
        brain.pool = providers.Pool([])
        brain.history = []
        with self.assertRaises(RuntimeError):
            brain._complete(with_tools=False)

    def test_the_window_names_whoever_answered(self):
        answer = SimpleNamespace(choices=[])
        brain, _, _ = self._brain([_refusal(429), answer])
        brain._complete(with_tools=False)
        self.assertEqual(providers.active_label(), "m")


class StickyLanguage(unittest.TestCase):
    """Detecting the language costs about as much as the transcription, so it
    is worth doing once - and only among the languages actually spoken here."""

    @staticmethod
    def _listener(pinned=None, probs=None, blow_up=False):
        listener = Listener.__new__(Listener)       # no model, no microphone
        listener.language, listener.detected = pinned, False
        listener.transcribe_seconds = 0.0

        def detect_language(audio, vad_filter=False):
            if blow_up:
                raise RuntimeError("no features")
            ranked = sorted(probs.items(), key=lambda kv: -kv[1])
            return ranked[0][0], ranked[0][1], list(probs.items())

        listener.model = SimpleNamespace(detect_language=detect_language)
        return listener

    def test_a_confident_detection_is_kept(self):
        listener = self._listener(probs={"en": 0.97, "te": 0.02})
        listener._choose_language(None)
        self.assertEqual(listener.language, "en")
        self.assertTrue(listener.detected)

    def test_a_shaky_detection_is_not_kept(self):
        listener = self._listener(probs={"en": 0.41, "hi": 0.30})
        listener._choose_language(None)
        self.assertIsNone(listener.language)

    def test_only_the_configured_languages_can_win(self):
        """Whisper read this accent as Russian and wrote confident nonsense."""
        listener = self._listener(probs={"ru": 0.80, "en": 0.90, "cy": 0.70})
        with mock.patch.object(config, "LANGUAGES", ["en", "te", "hi"]):
            listener._choose_language(None)
        self.assertEqual(listener.language, "en")

    def test_a_language_outside_the_list_is_ignored_even_when_it_leads(self):
        listener = self._listener(probs={"ru": 0.99, "te": 0.88})
        with mock.patch.object(config, "LANGUAGES", ["en", "te", "hi"]):
            listener._choose_language(None)
        self.assertEqual(listener.language, "te")

    def test_nothing_configured_matching_leaves_it_undecided(self):
        listener = self._listener(probs={"ru": 0.99})
        with mock.patch.object(config, "LANGUAGES", ["en"]):
            listener._choose_language(None)
        self.assertIsNone(listener.language)

    def test_a_detection_failure_falls_back_to_english(self):
        listener = self._listener(blow_up=True)
        listener._choose_language(None)
        self.assertEqual(listener.language, "en")

    def test_a_guess_is_dropped_when_nothing_comes_back(self):
        """Silence after a guess may mean the guess was wrong, so try again."""
        listener = self._listener(probs={"en": 0.99})
        listener._choose_language(None)
        listener._forget_language()
        self.assertIsNone(listener.language)
        self.assertFalse(listener.detected)

    def test_a_pinned_language_survives_a_bad_recording(self):
        listener = self._listener(pinned="en")
        listener._forget_language()
        self.assertEqual(listener.language, "en")


class Vocabulary(unittest.TestCase):
    """Proper nouns are most of what it gets wrong, and it cannot spell a name
    it has never been told: Hyderabad came back as "Bunt" and "chydra bad"."""

    def test_the_names_it_kept_missing_are_in_the_hint(self):
        words = config.VOCABULARY.lower()
        for name in ["reo", "claude", "hyderabad", "telugu"]:
            with self.subTest(name=name):
                self.assertIn(name, words)

    def test_english_is_considered_first(self):
        self.assertEqual(config.LANGUAGES[0], "en")

    def test_a_wider_beam_than_one(self):
        self.assertGreater(config.BEAM_SIZE, 1)


class ReasoningEffort(unittest.TestCase):
    """Reo answers in one sentence; a reasoning model need not deliberate."""

    def _kwargs(self, model):
        sent = {}
        brain = Brain.__new__(Brain)
        brain.history = []

        def create(**kwargs):
            sent.update(kwargs)
            return SimpleNamespace(choices=[])

        client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)))
        provider = providers.Provider("p", "k", "https://p.invalid", model)
        brain._ask(provider, client, with_tools=True)
        return sent

    def test_gpt_oss_is_asked_to_think_less(self):
        self.assertEqual(self._kwargs("openai/gpt-oss-120b")["reasoning_effort"],
                         config.REASONING_EFFORT)

    def test_other_models_are_not_sent_an_unknown_parameter(self):
        """Llama and Gemini reject it, and a 400 is not worth the saving."""
        self.assertNotIn("reasoning_effort",
                         self._kwargs("Llama-4-Maverick-17B-128E-Instruct-FP8"))


class Contractions(unittest.TestCase):
    """People say "what's", and the fast path was letting every one through."""

    def test_an_apostrophe_does_not_split_a_word(self):
        self.assertEqual(normalize("what's the battery"), "whats the battery")

    def test_a_curly_apostrophe_too(self):
        """Whisper prefers the typographic one."""
        self.assertEqual(normalize("what’s the time"), "whats the time")

    def test_contractions_reach_the_fast_path(self):
        for said in ["what's the battery", "what’s the time", "what's the date",
                     "what's my uptime", "what's the square root of 81"]:
            with self.subTest(said=said):
                self.assertIsNotNone(fastpath.answer(said))

    def test_sentences_still_break_on_real_punctuation(self):
        self.assertEqual(normalize(" Goodbye, Reo. "), "goodbye reo")


class Echoes(unittest.TestCase):
    """gpt-oss sometimes repeats the answer it has just given."""

    def test_a_whole_reply_said_twice(self):
        self.assertEqual(split_emotion("[neutral] Lima. Lima.")[1], "Lima.")

    def test_a_bare_answer_tacked_on_the_end(self):
        emotion, spoken = split_emotion(
            "[neutral] The capital of Peru is Lima.[neutral] Lima.")
        self.assertEqual(spoken, "The capital of Peru is Lima.")
        self.assertEqual(emotion, "neutral")

    def test_a_real_second_sentence_is_kept(self):
        self.assertEqual(
            split_emotion("[happy] I like Rome. Rome is warm.")[1],
            "I like Rome. Rome is warm.")

    def test_a_long_tail_is_not_treated_as_an_echo(self):
        said = "It is raining. Raining hard enough to flood the road."
        self.assertEqual(split_emotion("[sad] " + said)[1], said)


class Remembering(unittest.TestCase):
    """Reo used to forget everything the moment it closed."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = Path(self.dir) / "memory.json"

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _memory(self):
        return Memory(self.path)

    def test_a_fact_survives_a_restart(self):
        self._memory().remember("His name is Nandan.")
        self.assertIn("Nandan", self._memory().recall())

    def test_the_same_fact_is_not_stored_twice(self):
        m = self._memory()
        m.remember("His name is Nandan.")
        m.remember("his name is nandan")
        self.assertEqual(len(m.facts), 1)

    def test_forgetting_matches_on_content(self):
        m = self._memory()
        m.remember("He lives in Hyderabad.")
        m.remember("His name is Nandan.")
        m.forget("hyderabad")
        self.assertEqual([f["text"] for f in m.facts], ["His name is Nandan."])

    def test_forgetting_something_unknown_says_so(self):
        self.assertIn("Nothing remembered", self._memory().forget("quantum mechanics"))

    def test_an_empty_memory_adds_nothing_to_the_prompt(self):
        self.assertEqual(self._memory().for_prompt(), "")

    def test_the_prompt_carries_the_facts(self):
        m = self._memory()
        m.remember("He prefers short answers.")
        self.assertIn("He prefers short answers.", m.for_prompt())

    def test_only_so_many_facts_ride_along(self):
        """Every one is resent on every request, against an 8000 token minute."""
        m = self._memory()
        for i in range(memory_module.CARRY + 15):
            m.remember(f"Fact number {i}.")
        self.assertEqual(m.for_prompt().count(";"), memory_module.CARRY - 1)

    def test_the_file_is_capped(self):
        m = self._memory()
        for i in range(memory_module.KEEP + 20):
            m.remember(f"Fact number {i}.")
        self.assertEqual(len(self._memory().facts), memory_module.KEEP)

    def test_a_corrupt_file_is_not_fatal(self):
        self.path.write_text("{ this is not json", encoding="utf-8")
        self.assertEqual(self._memory().facts, [])

    def test_a_blank_fact_is_refused(self):
        m = self._memory()
        m.remember("   ")
        self.assertEqual(m.facts, [])


class _Page:
    def __init__(self, status, body):
        self.status_code, self.text = status, body


class WikipediaLookups(unittest.TestCase):
    """An ambiguous word has to reach the article, not the list of articles."""

    def _stub(self, pages, hits):
        import json as _json

        def client(timeout=15):
            def get(url):
                if "list=search" in url:
                    return _Page(200, _json.dumps(
                        {"query": {"search": [{"title": h} for h in hits]}}))
                title = urllib.parse.unquote(url.rsplit("/", 1)[-1]).replace("_", " ")
                page = pages.get(title)
                return _Page(200, _json.dumps(page)) if page else _Page(404, "")
            return SimpleNamespace(get=get)
        return client

    def _look_up(self, topic, pages, hits):
        original = tools._client
        tools._client = self._stub(pages, hits)
        try:
            return tools.look_up(topic)
        finally:
            tools._client = original

    def test_a_plain_article_comes_straight_back(self):
        pages = {"Hyderabad": {"title": "Hyderabad", "extract": "A city in Telangana."}}
        self.assertEqual(self._look_up("Hyderabad", pages, []),
                         "Hyderabad: A city in Telangana.")

    def test_a_disambiguation_page_is_stepped_over(self):
        pages = {
            "Mercury": {"title": "Mercury", "type": "disambiguation",
                        "extract": "Mercury may refer to:"},
            "Mercury (planet)": {"title": "Mercury (planet)",
                                 "extract": "The first planet from the Sun."},
        }
        answer = self._look_up("Mercury", pages, ["Mercury", "Mercury (planet)"])
        self.assertEqual(answer, "Mercury (planet): The first planet from the Sun.")

    def test_a_missing_title_is_searched_for(self):
        pages = {"Transformer (deep learning)":
                 {"title": "Transformer (deep learning)", "extract": "An architecture."}}
        answer = self._look_up("transformer neural net", pages,
                               ["Transformer (deep learning)"])
        self.assertIn("An architecture.", answer)

    def test_nothing_found_says_so_rather_than_crashing(self):
        self.assertIn("no single article", self._look_up("zzz nonsense", {}, []))

    def test_only_disambiguation_pages_still_says_so(self):
        pages = {"Mercury": {"title": "Mercury", "type": "disambiguation",
                             "extract": "Mercury may refer to:"}}
        self.assertIn("no single article", self._look_up("Mercury", pages, ["Mercury"]))


class ToolTable(unittest.TestCase):
    """A tool the model is offered but cannot call is a silent dead end."""

    def test_every_offered_tool_can_be_run(self):
        for spec in tools.TOOLS:
            with self.subTest(tool=spec["name"]):
                self.assertIn(spec["name"], tools.FUNCTIONS)

    def test_every_runnable_tool_is_offered(self):
        offered = {spec["name"] for spec in tools.TOOLS}
        for name in tools.FUNCTIONS:
            with self.subTest(tool=name):
                self.assertIn(name, offered)

    def test_each_tool_is_described(self):
        for spec in tools.TOOLS:
            with self.subTest(tool=spec["name"]):
                self.assertTrue(spec.get("description", "").strip())


class InventedTools(unittest.TestCase):
    """One invented tool name used to break every question that followed.

    Groq checks the tool_calls in the history against the tools sent with the
    new request, so a recorded call to a tool reo does not have turns into a
    400 forever after - and the retry without tools is rejected too, because
    the call is still in the messages.
    """

    @staticmethod
    def _call(name, id="c1"):
        return SimpleNamespace(id=id, function=SimpleNamespace(name=name, arguments="{}"))

    def _brain(self, turns):
        """turns: a list of (tool_calls, content) the model will answer with."""
        brain = Brain.__new__(Brain)
        brain.history = []
        brain.pool = providers.Pool([providers.Provider("p", "k", "https://p.invalid", "m")])
        brain._clients = {}
        replies = iter(turns)

        def complete(with_tools):
            calls, content = next(replies)
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(tool_calls=calls, content=content),
                finish_reason="stop")])

        brain._complete = complete
        return brain

    def test_an_invented_call_is_not_recorded(self):
        brain = self._brain([([self._call("browser")], "Let me look."),
                             (None, "[neutral] Here you go.")])
        brain.think("close it for me")
        for message in brain.history:
            for call in message.get("tool_calls") or []:
                self.assertIn(call["function"]["name"], KNOWN_TOOLS)

    def test_a_real_call_alongside_an_invented_one_still_runs(self):
        brain = self._brain([
            ([self._call("get_datetime", "a"), self._call("browser", "b")], ""),
            (None, "[neutral] Done."),
        ])
        brain.think("what time is it")
        recorded = [c["function"]["name"] for m in brain.history
                    for c in (m.get("tool_calls") or [])]
        self.assertEqual(recorded, ["get_datetime"])

    def test_words_alongside_an_invented_call_become_the_answer(self):
        brain = self._brain([([self._call("browser")], "[neutral] I cannot do that.")])
        self.assertEqual(brain.think("close it"), "[neutral] I cannot do that.")

    def test_a_silent_invented_call_asks_the_model_again(self):
        brain = self._brain([([self._call("browser")], ""),
                             (None, "[neutral] Closed it.")])
        self.assertEqual(brain.think("close it"), "[neutral] Closed it.")

    def test_every_tool_call_keeps_its_result(self):
        brain = self._brain([
            ([self._call("get_datetime", "a"), self._call("browser", "b")], ""),
            (None, "[neutral] Done."),
        ])
        brain.think("what time is it")
        for i, message in enumerate(brain.history):
            for call in message.get("tool_calls") or []:
                later = [m.get("tool_call_id") for m in brain.history[i + 1:]]
                self.assertIn(call["id"], later)

    def test_a_poisoned_history_can_be_cleaned_up(self):
        """Belt and braces for a history that got one in anyway."""
        brain = Brain.__new__(Brain)
        brain.history = [
            {"role": "user", "content": "close it"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "x", "type": "function",
                 "function": {"name": "browser", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "x", "content": "Unknown tool browser."},
            {"role": "assistant", "content": "done"},
        ]
        self.assertTrue(brain._forget_invented_calls())
        self.assertEqual([m["role"] for m in brain.history], ["user", "assistant"])
        self.assertFalse(brain._forget_invented_calls())   # nothing left to drop

    def test_a_clean_history_is_left_alone(self):
        brain = Brain.__new__(Brain)
        brain.history = [
            {"role": "user", "content": "what time is it"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "x", "type": "function",
                 "function": {"name": "get_datetime", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "x", "content": "now"},
        ]
        self.assertFalse(brain._forget_invented_calls())
        self.assertEqual(len(brain.history), 3)


class MicrophoneFailures(unittest.TestCase):
    """A microphone that goes away mid-sentence used to end the session.

    sounddevice raised PortAudioError("Stream is stopped") out of
    stream.read(), it reached the top of the loop, and reo exited in the
    middle of a conversation.
    """

    def _listener(self):
        listener = Listener.__new__(Listener)
        listener.language, listener.detected = "en", False
        listener.transcribe_seconds = 0.0
        return listener

    def test_a_stream_that_dies_keeps_what_it_heard(self):
        import sounddevice as sd
        blocks = [np.full((160, 1), 0.3, dtype=np.float32)] * 8

        class Stream:
            def __init__(self): self.left = list(blocks)
            def start(self): pass
            def stop(self): pass
            def close(self): pass
            def read(self, n):
                if not self.left:
                    raise sd.PortAudioError("Stream is stopped", -9983)
                return self.left.pop(0), False

        listener = self._listener()
        with mock.patch.object(sd, "InputStream", lambda **kw: Stream()):
            audio = listener._record()
        self.assertIsNotNone(audio)          # not thrown away
        self.assertEqual(len(audio), 160 * 8)

    def test_a_dying_close_does_not_escape(self):
        """Handling the failed read was not enough: stopping a stream the
        device already dropped raises again, and that killed the process."""
        import sounddevice as sd
        size = listen.BLOCK_SAMPLES

        class Stream:
            def __init__(self): self.left = 8
            def start(self): pass
            def stop(self): raise sd.PortAudioError("Stream is stopped", -9983)
            def close(self): raise sd.PortAudioError("Stream is stopped", -9983)
            def read(self, n):
                if self.left <= 0:
                    raise sd.PortAudioError("Stream is stopped", -9983)
                self.left -= 1
                return np.full((size, 1), 0.3, dtype=np.float32), False

        listener = self._listener()
        with mock.patch.object(sd, "InputStream", lambda **kw: Stream()):
            audio = listener._record()      # must not raise
        self.assertEqual(len(audio), size * 8)

    def test_letting_go_of_the_device_really_closes_it(self):
        import sounddevice as sd
        closed = []

        class Stream:
            def start(self): pass
            def stop(self): closed.append("stop")
            def close(self): closed.append("close")
            def read(self, n):
                time.sleep(0.005)
                return np.zeros((listen.BLOCK_SAMPLES, 1), dtype=np.float32), False

        listener = self._listener()
        listener._mic = Microphone(keep_open=False)
        with mock.patch.object(sd, "InputStream", lambda **kw: Stream()):
            with mock.patch.object(config, "WAIT_FOR_SPEECH_SECONDS", 0.2):
                self.assertIsNone(listener._record())
        self.assertIn("close", closed)

    def test_a_microphone_that_will_not_open_returns_nothing(self):
        import sounddevice as sd

        def refuse(**kwargs):
            raise sd.PortAudioError("Invalid device", -9996)

        listener = self._listener()
        with mock.patch.object(sd, "InputStream", refuse):
            self.assertIsNone(listener._record())   # and does not raise


class YouTube(unittest.TestCase):
    """"Click the first video" is not a clicking problem: the search page
    carries every result's id, so the video has an address."""

    PAGE = ('{"videoId":"aaaaaaaaaaa","title":{"runs":[{"text":"First one"}]},'
            '"videoId":"bbbbbbbbbbb","title":{"runs":[{"text":"Second one"}]},'
            '"videoId":"aaaaaaaaaaa"}')

    def _play(self, page, status=200, **kwargs):
        opened = []
        client = lambda timeout=15: SimpleNamespace(
            get=lambda url: SimpleNamespace(status_code=status, text=page))
        with mock.patch.object(tools, "_client", client),              mock.patch.object(tools, "webbrowser",
                               SimpleNamespace(open=opened.append)):
            said = tools.play_youtube("mrbeast gaming", **kwargs)
        return said, opened

    def test_the_first_result_is_opened(self):
        said, opened = self._play(self.PAGE)
        self.assertEqual(opened, ["https://www.youtube.com/watch?v=aaaaaaaaaaa"])
        self.assertIn("First one", said)

    def test_a_later_result_can_be_asked_for(self):
        said, opened = self._play(self.PAGE, position=2)
        self.assertEqual(opened, ["https://www.youtube.com/watch?v=bbbbbbbbbbb"])
        self.assertIn("Second one", said)

    def test_a_repeated_id_is_not_counted_twice(self):
        """The page mentions the same video several times over."""
        said, opened = self._play(self.PAGE, position=3)
        self.assertIn("Only found 2", said)
        self.assertEqual(opened, [])

    def test_nothing_found_opens_nothing(self):
        said, opened = self._play("{}")
        self.assertIn("No videos found", said)
        self.assertEqual(opened, [])

    def test_a_refusal_from_youtube_is_reported(self):
        said, opened = self._play("", status=429)
        self.assertIn("429", said)
        self.assertEqual(opened, [])

    def test_an_escaped_title_is_readable(self):
        page = '{"videoId":"ccccccccccc","title":{"runs":[{"text":"100 \u0024 Challenge"}]}}'
        said, _ = self._play(page)
        self.assertIn("100 $ Challenge", said)


class Pointer(unittest.TestCase):
    """reo has no eyes, so a coordinate it cannot check is one it must refuse."""

    def test_an_off_screen_click_is_refused_not_clamped(self):
        """Clamping turns a misheard number into a plausible click, and the
        corners of a screen hold Close and Start."""
        with mock.patch.object(pointer, "screen", lambda: (2560, 1600)):
            self.assertIn("off the screen", pointer.move(9999, 10))
            self.assertIn("off the screen", pointer.move(10, -5))

    def test_a_coordinate_on_screen_is_accepted(self):
        moved = []
        fake = SimpleNamespace(position=(0, 0))
        with mock.patch.object(pointer, "screen", lambda: (2560, 1600)),              mock.patch.object(pointer, "_controller", lambda: fake):
            said = pointer.move(100, 200)
        self.assertEqual(fake.position, (100, 200))
        self.assertIn("(100, 200)", said)

    def test_turning_the_mouse_off_refuses_everything(self):
        with mock.patch.object(config, "ALLOW_MOUSE", False):
            for said in (pointer.move(1, 1), pointer.click(1, 1),
                         pointer.scroll(-3), pointer.where()):
                self.assertIn("REO_ALLOW_MOUSE", said)

    def test_an_unknown_button_is_refused(self):
        with mock.patch.object(pointer, "screen", lambda: (2560, 1600)),              mock.patch.object(pointer, "_controller",
                               lambda: SimpleNamespace(position=(0, 0))):
            self.assertIn("No such mouse button", pointer.click(button="elbow"))


class Inventions(unittest.TestCase):
    """Whisper does not return nothing for silence. It makes something up.

    Given a few seconds of room noise it produced "I've been on 30 YouTube
    videos again and again and again and again", and reo answered it as
    though it had been said.
    """

    HEARD = [
        "I've been on 30 YouTube videos again and again and again and again.",
        "Thank you. Thank you. Thank you.",
        "you you you you",
        "Subtitles by the Amara.org community Subtitles by the Amara.org community",
    ]
    REAL = [
        "Open the first video",
        "Play MrBeast Gaming and then open the second video",
        "I said it again and again",          # twice over is emphasis, not a loop
        "No no, open it",
        "No no no, stop",                    # a person insisting, not a loop
        "What's the weather in Hyderabad",
        "Play it again",
    ]

    def test_the_inventions_are_caught(self):
        for said in self.HEARD:
            with self.subTest(said=said):
                self.assertTrue(looks_hallucinated(said))

    def test_real_speech_is_left_alone(self):
        for said in self.REAL:
            with self.subTest(said=said):
                self.assertFalse(looks_hallucinated(said))

    def test_silence_never_reaches_whisper(self):
        self.assertFalse(has_speech(np.zeros(16000, dtype=np.float32), 0.012))

    def test_a_quiet_room_never_reaches_whisper(self):
        room = np.full(16000, 0.0012, dtype=np.float32)   # his measured floor
        self.assertFalse(has_speech(room, 0.012))

    def test_ordinary_speech_gets_through(self):
        speech = np.full(16000, 0.07, dtype=np.float32)   # his measured peak
        self.assertTrue(has_speech(speech, 0.012))

    def test_an_empty_recording_is_not_speech(self):
        self.assertFalse(has_speech(np.array([], dtype=np.float32), 0.012))
        self.assertFalse(has_speech(None, 0.012))

    def test_a_single_loud_moment_counts(self):
        """One word in an otherwise quiet six seconds is still a word."""
        clip = np.full(16000, 0.001, dtype=np.float32)
        clip[8000:8400] = 0.06
        self.assertTrue(has_speech(clip, 0.012))


class SilentSegments(unittest.TestCase):
    """Whisper rates how likely each segment is to be silence; reo now listens.

    Pinned to the local path: this is about what reo does with the segments,
    and without pinning it these quietly started posting their audio to Groq.
    """

    def setUp(self):
        local = mock.patch.object(config, "HEARING", "local")
        local.start()
        self.addCleanup(local.stop)

    def _listener(self, segments):
        listener = Listener.__new__(Listener)
        listener.language, listener.detected = "en", True
        listener.transcribe_seconds = 0.0
        listener._record = lambda stop_now=None: np.full(16000, 0.07, dtype=np.float32)
        listener.model = SimpleNamespace(
            transcribe=lambda *a, **kw: (iter(segments), SimpleNamespace()))
        return listener

    @staticmethod
    def _segment(text, no_speech=0.0):
        return SimpleNamespace(text=text, no_speech_prob=no_speech)

    def test_a_segment_whisper_calls_silence_is_dropped(self):
        listener = self._listener([self._segment(" Open it", 0.1),
                                   self._segment(" Thanks for watching!", 0.95)])
        self.assertEqual(listener.listen(), "Open it")

    def test_all_silent_segments_leave_nothing(self):
        listener = self._listener([self._segment(" Thanks for watching!", 0.95)])
        self.assertEqual(listener.listen(), "")

    def test_confident_speech_is_kept(self):
        listener = self._listener([self._segment(" Play the first video", 0.02)])
        self.assertEqual(listener.listen(), "Play the first video")


class Endpointing(unittest.TestCase):
    """How long reo waits before deciding a sentence is over. Too short cuts
    it in half and transcribes the first part alone - which is where "Abad"
    came from, the tail of Hyderabad arriving by itself. Too long is three
    seconds of nothing at the end of every single command."""

    def test_a_thinking_pause_survives(self):
        """Ordinary mid-sentence pauses run to about a second."""
        self.assertGreaterEqual(config.HOTKEY_SILENCE_SECONDS, 1.2)

    def test_but_not_so_long_that_it_feels_broken(self):
        self.assertLessEqual(config.HOTKEY_SILENCE_SECONDS, 2.0)

    def test_without_a_key_silence_still_ends_it(self):
        """Nothing else is going to, so it has to stay short."""
        self.assertLess(config.SILENCE_SECONDS, 1.5)

    def test_the_guess_starts_well_inside_the_wait(self):
        """Or there is no pause left to do the transcribing in."""
        self.assertLess(config.EARLY_GUESS_SECONDS, config.SILENCE_SECONDS)
        self.assertLess(config.EARLY_GUESS_SECONDS, config.HOTKEY_SILENCE_SECONDS)

    def _blocks_before_stopping(self, stop_now):
        """How many blocks of pure silence _record tolerates after speech."""
        listener, heard = _scripted(
            [True] * 3 + [False] * 200, transcripts=["something"])
        listener._record(stop_now=stop_now)
        return len(heard)

    def test_a_pause_survives_longer_when_a_key_will_end_it(self):
        with_key = self._blocks_before_stopping(stop_now=lambda: False)
        without = self._blocks_before_stopping(stop_now=None)
        self.assertGreater(with_key, without)


class ParallelTools(unittest.TestCase):
    """Several searches in one reply took a second or two each, one after
    another. Looking things up can happen at once; pressing keys cannot."""

    def _run(self, names):
        order, started = [], []
        brain = Brain.__new__(Brain)
        brain.history = []
        brain.pool = providers.Pool([providers.Provider("p", "k", "https://p.invalid", "m")])
        brain._clients = {}

        calls = [SimpleNamespace(id=f"c{i}", function=SimpleNamespace(
            name=n, arguments="{}")) for i, n in enumerate(names)]
        replies = iter([(calls, ""), (None, "[neutral] Done.")])

        def complete(with_tools):
            c, content = next(replies)
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(tool_calls=c, content=content),
                finish_reason="stop")])

        brain._complete = complete

        import reo.brain as brainmod
        import threading
        lock = threading.Lock()
        overlap = {"seen": False}

        def fake(name, args):
            with lock:
                started.append(name)
                if len(started) - len(order) > 1:
                    overlap["seen"] = True
            time.sleep(0.12)
            with lock:
                order.append(name)
            return "ok " + name

        with mock.patch.object(brainmod, "run_tool", fake):
            t0 = time.time()
            brain.think("go")
            elapsed = time.time() - t0
        return order, elapsed, overlap["seen"], brain

    def test_lookups_run_at_the_same_time(self):
        order, elapsed, overlapped, _ = self._run(["search_web", "look_up", "read_page"])
        self.assertTrue(overlapped)
        self.assertLess(elapsed, 0.33)          # not 3 x 0.12 end to end

    def test_anything_that_acts_runs_in_order(self):
        """press_key then type_text has to happen in that order or it is wrong."""
        order, elapsed, overlapped, _ = self._run(["press_key", "type_text"])
        self.assertFalse(overlapped)
        self.assertEqual(order, ["press_key", "type_text"])

    def test_one_action_among_lookups_makes_the_whole_batch_wait(self):
        order, _, overlapped, _ = self._run(["search_web", "press_key", "look_up"])
        self.assertFalse(overlapped)
        self.assertEqual(order, ["search_web", "press_key", "look_up"])

    def test_results_stay_paired_with_their_calls(self):
        _, _, _, brain = self._run(["search_web", "look_up", "read_page"])
        pairs = {m["tool_call_id"]: m["content"]
                 for m in brain.history if m.get("role") == "tool"}
        self.assertEqual(pairs, {"c0": "ok search_web", "c1": "ok look_up",
                                 "c2": "ok read_page"})


class SystemControls(unittest.TestCase):
    """Volume and media are not characters, so SendKeys cannot reach them."""

    def _fake(self):
        taps = []
        return taps, SimpleNamespace(
            keybd_event=lambda code, a, flags, b: taps.append((code, flags)),
            GetForegroundWindow=lambda: 1234,
            GetWindowTextLengthW=lambda h: 7,
            GetWindowTextW=lambda h, buf, n: buf.__setitem__(slice(0, 7), "Notepad"),
            ShowWindow=lambda h, cmd: taps.append(("show", cmd)),
            PostMessageW=lambda h, msg, a, b: taps.append(("post", msg)),
            LockWorkStation=lambda: taps.append(("lock", 0)),
        )

    def test_volume_repeats_because_windows_moves_two_notches(self):
        taps, fake = self._fake()
        with mock.patch.object(reo_system, "_user32", lambda: fake):
            said = reo_system.control("volume_up")
        self.assertEqual(len([t for t in taps if t[0] == 0xAF and t[1] == 0]), 5)
        self.assertIn("up", said)

    def test_an_amount_overrides_the_default(self):
        taps, fake = self._fake()
        with mock.patch.object(reo_system, "_user32", lambda: fake):
            reo_system.control("volume_down", 2)
        self.assertEqual(len([t for t in taps if t[0] == 0xAE and t[1] == 0]), 2)

    def test_media_keys_fire_once(self):
        taps, fake = self._fake()
        with mock.patch.object(reo_system, "_user32", lambda: fake):
            reo_system.control("play_pause")
        self.assertEqual(len([t for t in taps if t[0] == 0xB3 and t[1] == 0]), 1)

    def test_spoken_wording_is_accepted(self):
        taps, fake = self._fake()
        with mock.patch.object(reo_system, "_user32", lambda: fake):
            self.assertIn("up", reo_system.control("Volume Up"))
            self.assertIn("up", reo_system.control("volume-up"))

    def test_closing_asks_the_program_rather_than_killing_it(self):
        """WM_CLOSE lets it prompt about unsaved work."""
        taps, fake = self._fake()
        with mock.patch.object(reo_system, "_user32", lambda: fake):
            said = reo_system.control("close_window")
        self.assertIn(("post", 0x0010), taps)
        self.assertIn("Notepad", said)

    def test_an_unknown_control_lists_the_real_ones(self):
        said = reo_system.control("make me a sandwich")
        self.assertIn("No such control", said)
        self.assertIn("volume_up", said)

    def test_it_is_reachable_as_a_tool(self):
        self.assertIn("control_system", tools.FUNCTIONS)


def _fake_stream(blocks=None, pace=0.005):
    """A microphone that hands out blocks forever, as fast as it is asked."""
    size = listen.BLOCK_SAMPLES
    made = []

    class Stream:
        def start(self): pass
        def stop(self): pass
        def close(self): pass
        def read(self, n):
            time.sleep(pace)
            if blocks:
                block = blocks.pop(0)
            else:
                block = np.full((size, 1), 0.01 * (len(made) % 7 + 1), dtype=np.float32)
            made.append(block)
            return block, False

    return Stream, made


def _scripted(voiced, transcripts, stop_now=None, pace=None):
    """A Listener whose ears are a script, so the timing can be tested without
    a microphone or a model. Returns the listener and the list it fills in."""
    import sounddevice as sd
    heard = []
    Stream, _ = _fake_stream(pace=listen.BLOCK_SECONDS if pace is None else pace)
    patcher = mock.patch.object(sd, "InputStream", lambda **kw: Stream())
    patcher.start()

    listener = Listener.__new__(Listener)
    listener.language, listener.detected = "en", True
    listener.transcribe_seconds = 0.0
    listener._guess = None
    listener._mic = Microphone(keep_open=False)
    listener._voice = SimpleNamespace(reset=lambda: None, ready=True)
    script = list(voiced)

    def ears(block, mid_sentence):
        heard.append(mid_sentence)
        return script.pop(0) if script else False

    listener._voiced = ears
    said = list(transcripts)

    def transcribe(raw):
        time.sleep(0.05)
        listener.transcribed.append(len(raw))
        return said.pop(0) if said else ""

    listener.transcribed = []
    listener._transcribe = transcribe
    listener._stopper = patcher
    return listener, heard


class HeldMicrophone(unittest.TestCase):
    """Opening the device costs half a second - 491ms measured on this laptop -
    and that half second used to sit between the key press and the first word."""

    def _mic(self, keep_open=True):
        import sounddevice as sd
        Stream, made = _fake_stream()
        patcher = mock.patch.object(sd, "InputStream", lambda **kw: Stream())
        patcher.start()
        self.addCleanup(patcher.stop)
        mic = Microphone(keep_open=keep_open)
        self.addCleanup(mic.close)
        return mic

    def test_what_came_before_the_key_press_comes_with_it(self):
        mic = self._mic()
        self.assertTrue(mic.open())
        time.sleep(0.15)                        # the room, before he presses
        before = mic.start()
        mic.done()
        self.assertTrue(before, "nothing was kept from before the key press")
        self.assertGreater(float(np.max(np.abs(np.concatenate(before)))), 0.0)

    def test_it_keeps_only_a_moment_of_it(self):
        """It is a ring, not a recording: the room is not being collected."""
        mic = self._mic()
        mic.open()
        time.sleep(0.6)
        kept = len(mic.start()) * listen.BLOCK_SECONDS
        mic.done()
        self.assertLessEqual(kept, config.PRE_ROLL_SECONDS + listen.BLOCK_SECONDS)

    def test_the_ring_is_emptied_when_a_recording_ends(self):
        mic = self._mic()
        mic.open()
        time.sleep(0.15)
        mic.start()
        mic.done()
        self.assertEqual(list(mic._recent), [])

    def test_a_held_microphone_is_opened_once(self):
        mic = self._mic(keep_open=True)
        mic.start()
        mic.done()
        first = mic._stream
        mic.start()
        mic.done()
        self.assertIs(mic._stream, first, "reopened the device between commands")

    def test_letting_go_closes_it(self):
        mic = self._mic(keep_open=False)
        mic.start()
        mic.done()
        self.assertIsNone(mic._stream)

    def test_no_microphone_is_not_a_crash(self):
        import sounddevice as sd

        def refuse(**kwargs):
            raise sd.PortAudioError("Invalid device", -9996)

        with mock.patch.object(sd, "InputStream", refuse):
            mic = Microphone()
            self.addCleanup(mic.close)
            self.assertFalse(mic.open())
            self.assertIsNone(mic.start())

    def test_the_wake_word_is_not_made_to_share_the_device(self):
        """In wake mode openWakeWord holds a stream of its own on the same
        microphone, so only the hotkey keeps one open."""
        listener = Listener.__new__(Listener)
        with mock.patch.object(config, "MODE", "wake"):
            self.assertFalse(listener.mic.keep_open)


class TranscribingInThePause(unittest.TestCase):
    """Recognition costs about 0.6s whatever the length, and that 0.6s used to
    be added on after the wait rather than spent during it."""

    def tearDown(self):
        for listener in getattr(self, "_made", []):
            listener._stopper.stop()

    def _listener(self, voiced, transcripts, **kw):
        listener, _ = _scripted(voiced, transcripts, **kw)
        self._made = getattr(self, "_made", []) + [listener]
        return listener

    def test_the_words_are_ready_when_the_silence_is_up(self):
        silence = int(config.SILENCE_SECONDS / listen.BLOCK_SECONDS) + 2
        listener = self._listener([True] * 5 + [False] * silence,
                                  ["open chrome"])
        self.assertEqual(listener.listen(), "open chrome")
        self.assertLess(listener.transcribe_seconds, 0.03,
                        "he waited for the transcription instead of it waiting for him")

    def test_talking_again_throws_the_half_sentence_away(self):
        silence = int(config.SILENCE_SECONDS / listen.BLOCK_SECONDS)
        early = int(config.EARLY_GUESS_SECONDS / listen.BLOCK_SECONDS)
        listener = self._listener(
            [True] * 5 + [False] * (early + 2) + [True] * 3 + [False] * (silence + 2),
            ["half of it", "the whole of it"])
        self.assertEqual(listener.listen(), "the whole of it")
        self.assertEqual(len(listener.transcribed), 2, "it did not try twice")
        self.assertGreater(listener.transcribed[1], listener.transcribed[0],
                           "the second attempt did not cover more of the sentence")

    def test_a_recording_with_no_voice_in_it_is_never_transcribed(self):
        listener = self._listener([False] * 400, ["should not happen"])
        with mock.patch.object(config, "WAIT_FOR_SPEECH_SECONDS", 0.3):
            self.assertEqual(listener.listen(), "")
        self.assertEqual(listener.transcribed, [])


class HearingAVoice(unittest.TestCase):
    """Loudness cannot tell a person from a door. This is the model whisper
    itself uses to find speech, run live on each block as it arrives."""

    @classmethod
    def setUpClass(cls):
        cls.detector = VoiceDetector()
        if not cls.detector.ready:
            raise unittest.SkipTest("no onnxruntime, so no voice detection")

    def _marks(self, audio):
        self.detector.reset()
        size = listen.BLOCK_SAMPLES
        return [self.detector.chance(audio[i:i + size]) > au_mod.SPEECH_STARTS
                for i in range(0, len(audio) - size, size)]

    def test_his_own_voice_is_found(self):
        audio = _diagnose_clip()
        if audio is None:
            self.skipTest("no recording of him to test against")
        marks = self._marks(audio)
        self.assertTrue(any(marks), "did not hear him at all")
        # He speaks for about 2.6s of the six, with a long pause in the middle.
        heard = sum(marks) * listen.BLOCK_SECONDS
        self.assertGreater(heard, 1.5)
        self.assertLess(heard, 4.0)

    def test_the_quiet_part_of_his_own_room_is_not_a_voice(self):
        audio = _diagnose_clip()
        if audio is None:
            self.skipTest("no recording of him to test against")
        # The gap in the middle of that recording: his actual room, not silence.
        self.assertFalse(any(self._marks(audio[16000 * 2 + 8000: 16000 * 4])))

    def test_silence_is_not_a_voice(self):
        self.assertFalse(any(self._marks(np.zeros(16000, dtype=np.float32))))

    def test_a_loud_bang_is_not_a_voice(self):
        """A door, a dropped book, a hand on the desk. Louder than he is."""
        bang = np.zeros(16000, dtype=np.float32)
        bang[4000:4300] = np.linspace(1.0, 0.0, 300) ** 2
        self.assertFalse(any(self._marks(bang)))

    def test_a_hiss_is_not_a_voice(self):
        rng = np.random.default_rng(7)
        hiss = rng.normal(0, 0.05, 16000).astype(np.float32)
        self.assertFalse(any(self._marks(hiss)))

    def test_losing_the_model_falls_back_to_loudness(self):
        detector = VoiceDetector()
        detector._session = None
        self.assertEqual(detector.chance(np.zeros(1024, dtype=np.float32)), -1.0)
        listener = Listener.__new__(Listener)
        listener._voice = detector
        self.assertTrue(listener._voiced(
            np.full(1024, 0.3, dtype=np.float32), mid_sentence=False))
        self.assertFalse(listener._voiced(
            np.zeros(1024, dtype=np.float32), mid_sentence=False))


class NeverWaitForEver(unittest.TestCase):
    """One 2.5s clip of him took 853 seconds to transcribe and never did it
    again. Whatever that was, reo should give up rather than freeze."""

    def test_an_attempt_that_finishes_hands_back_its_words(self):
        attempt = Attempt(lambda: "open chrome")
        self.assertEqual(attempt.wait(2.0), "open chrome")
        self.assertTrue(attempt.finished)

    def test_an_attempt_that_will_not_finish_is_given_up_on(self):
        stuck = threading.Event()
        self.addCleanup(stuck.set)
        attempt = Attempt(lambda: stuck.wait() or "too late")
        self.assertIsNone(attempt.wait(0.1))
        self.assertFalse(attempt.finished)

    def test_trouble_in_the_thread_is_not_swallowed(self):
        def explode():
            raise RuntimeError("the model fell over")

        attempt = Attempt(explode)
        with self.assertRaises(RuntimeError):
            attempt.wait(2.0)

    def test_listening_gives_up_instead_of_freezing(self):
        stuck = threading.Event()
        self.addCleanup(stuck.set)
        listener = Listener.__new__(Listener)
        listener.language, listener.detected = "en", True
        listener.transcribe_seconds = 0.0
        listener._guess = None
        listener._record = lambda stop_now=None: np.zeros(16000, dtype=np.float32)
        listener._transcribe = lambda raw: stuck.wait() or "too late"
        with mock.patch.object(config, "TRANSCRIBE_TIMEOUT", 0.15):
            began = time.time()
            self.assertEqual(listener.listen(), "")
        self.assertLess(time.time() - began, 2.0)

    def test_a_stuck_attempt_does_not_hold_the_model_for_ever(self):
        """The next command gets told the model is busy, not left hanging."""
        self.enterContext(mock.patch.object(config, "HEARING", "local"))
        listener = Listener.__new__(Listener)
        listener.language, listener.detected = "en", True
        listener._model_lock = threading.Lock()
        listener._voice = SimpleNamespace(ready=True, reset=lambda: None)
        listener._model_lock.acquire()
        self.addCleanup(listener._model_lock.release)
        with mock.patch.object(config, "TRANSCRIBE_TIMEOUT", 0.1):
            self.assertEqual(
                listener._transcribe(np.full(16000, 0.07, dtype=np.float32)), "")


class ADeadDevice(unittest.TestCase):
    """A muted microphone and a mumbled sentence look identical from inside
    reo, and they need completely different things done about them."""

    def _listener(self, audio):
        listener = Listener.__new__(Listener)
        listener.language, listener.detected = "en", True
        listener.transcribe_seconds = 0.0
        listener._guess = None
        listener._record = lambda stop_now=None: audio
        listener._transcribe = lambda raw: "should never be asked"
        return listener

    def test_digital_silence_is_never_transcribed(self):
        listener = self._listener(np.zeros(16000, dtype=np.float32))
        self.assertEqual(listener.listen(), "")
        self.assertTrue(listener.heard_nothing)

    def test_a_muted_realtek_array_counts_as_nothing(self):
        """Measured on his own laptop: 0.00003 peak with nothing coming in."""
        listener = self._listener(np.full(16000, 0.00003, dtype=np.float32))
        self.assertEqual(listener.listen(), "")
        self.assertTrue(listener.heard_nothing)

    def test_a_quiet_room_is_not_a_dead_device(self):
        """0.0012 is his measured room. That is a mumble problem, not a
        plugged-in problem, and it still goes to whisper."""
        listener = self._listener(np.full(16000, 0.0012, dtype=np.float32))
        listener._transcribe = lambda raw: "turn it down"
        self.assertEqual(listener.listen(), "turn it down")
        self.assertFalse(listener.heard_nothing)

    def test_he_is_told_which_of_the_two_it_was(self):
        import reo.__main__ as main
        said = []
        listener = self._listener(np.zeros(16000, dtype=np.float32))
        listener.listen()
        ui = SimpleNamespace(closed=False, line=lambda kind, text: said.append(text),
                             set_state=lambda state: None)
        toggle = SimpleNamespace(listening=True,
                                 wait_until_listening=lambda: None)

        def stop_after_one(*a, **kw):
            ui.closed = True
            return ""

        listener.listen = stop_after_one
        listener.heard_nothing = True
        with mock.patch.object(main, "speak", lambda *a, **kw: None):
            main._loop(listener, None, None, toggle, "ctrl+alt+r", ui)
        self.assertTrue(any("muted" in line for line in said),
                        f"he was not told the microphone was silent: {said}")


class ActingWithoutTheModel(unittest.TestCase):
    """"Volume up" was a round trip to the model, a tool call and a second
    round trip, to press a key this machine can press itself."""

    def _routed(self, said):
        """Which handler takes this, without letting it actually run."""
        asked = normalize(said)
        for pattern, respond in fastpath.COMPILED:
            if pattern.fullmatch(asked):
                return getattr(respond, "__name__", "lambda")
        return None

    def _acted(self, said):
        """What reo did, with the laptop's controls faked out."""
        done = []
        with mock.patch.object(reo_system, "control",
                               lambda action, amount=0: done.append((action, amount))
                               or f"did {action}"):
            answered = fastpath.answer(said)
        return done, answered

    def test_the_volume_needs_no_model(self):
        for said, expected in [("turn it up", ("volume_up", 0)),
                               ("volume up", ("volume_up", 0)),
                               ("turn the volume down", ("volume_down", 0)),
                               ("turn it up by 3", ("volume_up", 3)),
                               ("louder", ("volume_up", 0)),
                               ("quieter", ("volume_down", 0)),
                               ("make it softer", ("volume_down", 0)),
                               ("mute", ("mute", 0)),
                               ("unmute it", ("mute", 0))]:
            with self.subTest(said=said):
                done, answered = self._acted(said)
                self.assertEqual(done, [expected])
                self.assertIsNotNone(answered)

    def test_media_keys_need_no_model(self):
        for said, action in [("play", "play_pause"), ("pause it", "play_pause"),
                             ("resume", "play_pause"), ("next track", "next_track"),
                             ("skip", "next_track"), ("previous song", "previous_track"),
                             ("last track", "previous_track")]:
            with self.subTest(said=said):
                done, _ = self._acted(said)
                self.assertEqual(done, [(action, 0)])

    def test_windows_and_the_lock_screen_need_no_model(self):
        for said, action in [("minimise this window", "minimise"),
                             ("minimize it", "minimise"),
                             ("maximise", "maximise"),
                             ("restore it", "restore"),
                             ("show desktop", "show_desktop"),
                             ("close this window", "close_window"),
                             ("lock the laptop", "lock"),
                             ("lock the screen", "lock"),
                             ("lock it", "lock")]:
            with self.subTest(said=said):
                done, _ = self._acted(said)
                self.assertEqual(done, [(action, 0)])

    def test_opening_a_program_needs_no_model(self):
        opened = []
        import reo.apps as apps
        with mock.patch.object(apps, "open_app",
                               lambda name: opened.append(name) or f"Opened {name}."):
            self.assertEqual(fastpath.answer("open chrome"), "Opened chrome.")
            self.assertEqual(fastpath.answer("launch notepad"), "Opened notepad.")
        self.assertEqual(opened, ["chrome", "notepad"])

    def test_a_program_it_cannot_find_goes_to_the_model(self):
        """Reading "Nothing installed matches" out loud is the worst of both:
        the model can look at the list and work out what he meant."""
        import reo.apps as apps
        with mock.patch.object(apps, "open_app",
                               lambda name: "Nothing installed matches 'spotifi'."):
            self.assertIsNone(fastpath.answer("open spotifi"))

    def test_a_website_is_opened_rather_than_launched(self):
        """normalize() has already eaten the dot, so "youtube.com" arrives as
        "youtube com" and has to be put back together."""
        went = []
        import reo.tools as tools_module
        with mock.patch.object(tools_module, "open_website",
                               lambda url: went.append(url) or f"Opened {url}."):
            fastpath.answer("open youtube.com")
            fastpath.answer("go to github.com")
        self.assertEqual(went, ["https://youtube.com", "https://github.com"]
                         if went and went[0].startswith("http")
                         else ["youtube.com", "github.com"])

    def test_the_things_that_must_still_reach_the_model(self):
        """A keyword search would hijack all of these."""
        for said in ["open the first result", "open it", "open my email",
                     "launch the second one", "play the first video",
                     "close the browser and open chrome", "how do i turn it up",
                     "turn up the heating in my room", "lock",
                     "up", "down", "go back", "what is the volume",
                     "whats on youtube", "should i lock my laptop"]:
            with self.subTest(said=said):
                self.assertIsNone(fastpath.answer(said),
                                  f"{said!r} was taken by the fast path")

    def test_a_question_is_never_answered_by_doing_it(self):
        """The questions are tried before the actions on purpose: asking about
        the volume must not change it."""
        done, _ = self._acted("whats my battery")
        self.assertEqual(done, [])

    def test_nothing_here_deletes_types_or_spends(self):
        """The whole list, read back. Anything irreversible is the model's
        decision, where it has a system prompt and a confirmation to lean on."""
        import reo.system as system_module
        allowed = set(system_module.VK) | set(system_module.SHOW) | {
            "show_desktop", "close_window", "lock"}
        seen = set()
        for said in ["turn it up", "turn it down", "mute", "play", "skip",
                     "previous song", "minimise", "maximise", "restore",
                     "show desktop", "close it", "lock it"]:
            done, _ = self._acted(said)
            seen.update(action for action, _ in done)
        self.assertTrue(seen)
        self.assertTrue(seen <= allowed, f"not on the reversible list: {seen - allowed}")


class CuttingReoOff(unittest.TestCase):
    """A reply to a misheard question used to have to be sat through to the
    end. Measured: 11.28s to say a long sentence, 1.17s when stopped."""

    def test_the_key_being_pressed_again_is_what_counts(self):
        """Not whether it is listening: in hotkey mode that stays on between
        commands, so it is true the whole time reo is talking."""
        from reo.hotkey import Toggle
        toggle = Toggle.__new__(Toggle)
        toggle._on = threading.Event()
        toggle._woke = threading.Event()
        toggle._presses = 0
        toggle._on_change = None

        toggle._flip()                      # he pressed to start
        self.assertTrue(toggle.listening)
        watch = main._interrupt_watch(toggle)
        self.assertFalse(watch(), "cut itself off before he touched anything")
        toggle._flip()                      # and again, mid-sentence
        self.assertTrue(watch())

    def test_cutting_in_hands_the_floor_straight_back(self):
        """Interrupting means he wants to say something, not that he wants to
        be asked again, so reo starts listening rather than waiting."""
        from reo.hotkey import Toggle
        toggle = Toggle.__new__(Toggle)
        toggle._on = threading.Event()
        toggle._woke = threading.Event()
        toggle._presses = 0
        toggle._on_change = None
        self.assertFalse(toggle.listening)

        with mock.patch.object(main, "speak", lambda *a, **kw: True):
            main._said("a long wrong answer", None, toggle, lambda: True)
        self.assertTrue(toggle.listening)

    def test_finishing_normally_changes_nothing(self):
        from reo.hotkey import Toggle
        toggle = Toggle.__new__(Toggle)
        toggle._on = threading.Event()
        toggle._woke = threading.Event()
        toggle._presses = 0
        toggle._on_change = None

        with mock.patch.object(main, "speak", lambda *a, **kw: False):
            main._said("the right answer", None, toggle, lambda: False)
        self.assertFalse(toggle.listening)

    def test_no_hotkey_means_nothing_to_watch(self):
        """Wake mode has no key to press, so there is nothing to interrupt."""
        self.assertIsNone(main._interrupt_watch(None))

    def test_the_offline_voice_is_asked_whether_to_stop(self):
        """Without passing it down, the watch is never consulted at all."""
        asked = []
        with mock.patch.object(config, "TTS_BACKEND", "pyttsx3"),              mock.patch.object(speak_module, "_speak_pyttsx3",
                               lambda text, emotion, until=None: bool(
                                   asked.append(until)) or False):
            speak_module.speak("[neutral] Done.", None, until=lambda: False)
        self.assertEqual(len(asked), 1)
        self.assertIsNotNone(asked[0], "the voice was given no way to be stopped")

    def test_the_groq_voice_is_asked_too(self):
        """Which is the one that matters here: his .env asks for Orpheus."""
        asked = []
        with mock.patch.object(config, "TTS_BACKEND", "orpheus"),              mock.patch.object(speak_module, "_orpheus_failed", False),              mock.patch.object(speak_module, "_speak_orpheus",
                               lambda text, until=None: bool(
                                   asked.append(until)) or False):
            speak_module.speak("[neutral] Done.", None, until=lambda: False)
        self.assertEqual(len(asked), 1)
        self.assertIsNotNone(asked[0], "the voice was given no way to be stopped")

    def test_playback_stops_where_it_got_to(self):
        """The Groq voice is the one his .env asks for, and its playback loop
        is where a press has to be noticed."""
        with mock.patch.object(speak_module, "_speaker",
                               lambda rate: SimpleNamespace(active=True)):
            cut = speak_module._play(np.full(16000, 400, dtype=np.int16), 16000,
                                     until=lambda: True)
        self.assertTrue(cut)
        self.assertTrue(speak_module._clip["final"],
                        "the clip was left running after he cut in")

class ReosDesk(unittest.TestCase):
    """Asked to write a program, reo could do everything except the writing.
    The names come from speech, so where they are allowed to land matters."""

    def setUp(self):
        self.home = Path(tempfile.mkdtemp(prefix="reo-desk-"))
        self.addCleanup(shutil.rmtree, self.home, True)
        patch = mock.patch.object(config, "WORKSPACE", str(self.home))
        patch.start()
        self.addCleanup(patch.stop)

    def test_it_writes_what_it_was_given(self):
        said = workspace.write("hello.py", "print('hi')\n")
        self.assertIn("Wrote hello.py", said)
        self.assertEqual((self.home / "hello.py").read_text(encoding="utf-8"),
                         "print('hi')\n")

    def test_a_subfolder_is_made_on_the_way(self):
        workspace.write("notes/ideas.md", "one")
        self.assertEqual((self.home / "notes" / "ideas.md").read_text(encoding="utf-8"),
                         "one")

    def test_replacing_keeps_the_one_before(self):
        """A misheard file name should not cost him an hour of work."""
        workspace.write("script.py", "first")
        said = workspace.write("script.py", "second")
        self.assertIn("Replaced", said)
        self.assertEqual((self.home / "script.py").read_text(encoding="utf-8"), "second")
        self.assertEqual((self.home / "script.py.last").read_text(encoding="utf-8"), "first")

    def test_it_reads_back_what_it_wrote(self):
        workspace.write("a.txt", "the contents")
        self.assertEqual(workspace.read("a.txt"), "the contents")

    def test_reading_something_that_is_not_there(self):
        self.assertIn("no missing.txt", workspace.read("missing.txt"))

    def test_listing_an_empty_desk_says_where_it_is(self):
        self.assertIn(str(self.home), workspace.listing())

    def test_listing_shows_what_is_there(self):
        workspace.write("one.py", "x")
        workspace.write("two/three.md", "y")
        shown = workspace.listing()
        self.assertIn("one.py", shown)
        self.assertIn("three.md", shown)

    def test_nothing_lands_outside_the_folder(self):
        """The whole point. These come from speech, which mishears."""
        for name in ["../escaped.py", "..\\escaped.py", "/etc/passwd",
                     "C:/Windows/System32/drivers/etc/hosts",
                     "notes/../../escaped.py", "....//escaped.py",
                     "~/escaped.py", "", "   ", "a\x00b"]:
            with self.subTest(name=name):
                said = workspace.write(name, "should never be written")
                self.assertNotIn("Wrote", said, f"{name!r} was allowed through")
                self.assertNotIn("Replaced", said)
        # and nothing at all appeared anywhere
        self.assertEqual(list(self.home.rglob("*")), [])

    def test_a_huge_file_is_refused(self):
        said = workspace.write("big.txt", "x" * 500_000)
        self.assertIn("limit", said)
        self.assertFalse((self.home / "big.txt").exists())

    def test_opening_something_that_is_not_there_does_not_launch_anything(self):
        opened = []
        with mock.patch.object(workspace.os, "startfile", lambda path: opened.append(path)):
            self.assertIn("no ghost.py", workspace.show("ghost.py"))
            self.assertEqual(opened, [])

    def test_opening_a_file_it_wrote(self):
        workspace.write("real.py", "x = 1")
        opened = []
        with mock.patch.object(workspace.os, "startfile", lambda path: opened.append(path)):
            self.assertIn("Opened real.py", workspace.show("real.py"))
        self.assertEqual(len(opened), 1)

    def test_the_tools_are_wired_up(self):
        for name in ("write_file", "read_file", "list_files", "open_file"):
            self.assertIn(name, tools.FUNCTIONS)
            self.assertIn(name, {spec["name"] for spec in tools.TOOLS})

    def test_reading_and_listing_count_as_lookups(self):
        """So several of them can run at once; writing cannot."""
        self.assertIn("read_file", tools.READ_ONLY)
        self.assertIn("list_files", tools.READ_ONLY)
        self.assertNotIn("write_file", tools.READ_ONLY)
        self.assertNotIn("open_file", tools.READ_ONLY)

    def test_there_is_room_to_write_a_program(self):
        """500 tokens stopped in the middle of one."""
        self.assertGreaterEqual(config.MAX_REPLY_TOKENS, 2000)


class AllowanceRunsOut(unittest.TestCase):
    """The free tier is 8000 tokens a minute and a program is two thousand of
    them. This is the one failure he will actually meet, and it is not a fault."""

    def _brain(self, keys, fails_with):
        brain = Brain.__new__(Brain)
        brain.history = []
        brain.pool = providers.Pool([
            providers.Provider(f"p{i}", "k", "https://p.invalid", "m")
            for i in range(keys)])
        brain._clients = {}
        brain._ask = fails_with
        return brain

    @staticmethod
    def _rate_limited(*a, **kw):
        raise RateLimitError("429", response=SimpleNamespace(
            status_code=429, request=None, headers={}), body=None)

    def test_one_key_out_of_allowance_says_so(self):
        brain = self._brain(1, self._rate_limited)
        with self.assertRaises(OutOfAllowance):
            brain._complete(with_tools=True)

    def test_every_key_out_of_allowance_says_so(self):
        brain = self._brain(3, self._rate_limited)
        with self.assertRaises(OutOfAllowance):
            brain._complete(with_tools=True)

    def test_a_real_fault_is_still_a_real_fault(self):
        def broken(*a, **kw):
            raise APIConnectionError(request=None)

        brain = self._brain(1, broken)
        with self.assertRaises(RuntimeError) as caught:
            brain._complete(with_tools=True)
        self.assertNotIsInstance(caught.exception, OutOfAllowance)

    def test_he_is_told_it_is_the_limit_rather_than_a_breakage(self):
        said = []
        brain = SimpleNamespace(
            history=[], think=mock.Mock(side_effect=OutOfAllowance("groq")))
        with mock.patch.object(main, "speak",
                               lambda text, ui=None, until=None: said.append(text)):
            main.handle(brain, "write me a game", None, 0.0, None, None)
        self.assertTrue(said)
        self.assertIn("allowance", said[0].lower())
        self.assertNotIn("went wrong", said[0].lower())

    def test_the_client_does_not_sit_on_a_429(self):
        """It used to wait out whatever retry-after the 429 carried: measured
        at 33 seconds of silence before reo said anything at all."""
        made = {}
        brain = Brain.__new__(Brain)
        brain._clients = {}
        with mock.patch.object(brain_module, "OpenAI",
                               lambda **kw: made.update(kw) or SimpleNamespace()):
            brain._client(0, providers.Provider("p", "k", "https://p.invalid", "m"))
        self.assertEqual(made.get("max_retries"), 0)


class TheModelOnThisMachine(unittest.TestCase):
    """Ollama has no key - it is this laptop - so it is found by asking.

    The capability is what matters, not the name: reo is nothing but tool
    calls, and his llama3 refuses them outright with "llama3:latest does not
    support tools".
    """

    def setUp(self):
        providers._local_seen = (0.0, False)
        self.addCleanup(setattr, providers, "_local_seen", (0.0, False))

    def _ollama(self, tags, capabilities):
        """Stand in for the local ollama's two endpoints."""
        asked = []

        def answer(path, body=None, timeout=1.5):
            asked.append(path)
            if path == "/api/tags":
                return tags
            if path == "/api/show":
                return {"capabilities": capabilities.get(body["model"], [])}
            return None

        return answer, asked

    def test_nothing_running_is_not_an_error(self):
        with mock.patch.object(providers, "_ollama", lambda *a, **kw: None):
            self.assertIsNone(providers.local_model({}))
            self.assertIsNone(providers.local({}))

    def test_a_model_without_tools_is_no_use(self):
        answer, _ = self._ollama({"models": [{"name": "llama3:latest"}]},
                                 {"llama3:latest": ["completion"]})
        with mock.patch.object(providers, "_ollama", answer):
            self.assertIsNone(providers.local_model({}))

    def test_a_model_with_tools_is_taken(self):
        answer, _ = self._ollama({"models": [{"name": "qwen2.5:7b"}]},
                                 {"qwen2.5:7b": ["completion", "tools"]})
        with mock.patch.object(providers, "_ollama", answer):
            self.assertEqual(providers.local_model({}), "qwen2.5:7b")

    def test_the_one_that_can_is_picked_out_of_several(self):
        answer, _ = self._ollama(
            {"models": [{"name": "llama3:latest"}, {"name": "qwen2.5:7b"}]},
            {"llama3:latest": ["completion"], "qwen2.5:7b": ["tools"]})
        with mock.patch.object(providers, "_ollama", answer):
            self.assertEqual(providers.local_model({}), "qwen2.5:7b")

    def test_a_named_model_wins(self):
        answer, _ = self._ollama(
            {"models": [{"name": "qwen2.5:7b"}, {"name": "llama3.1:8b"}]},
            {"qwen2.5:7b": ["tools"], "llama3.1:8b": ["tools"]})
        with mock.patch.object(providers, "_ollama", answer):
            self.assertEqual(
                providers.local_model({"REO_OLLAMA_MODEL": "llama3.1:8b"}),
                "llama3.1:8b")

    def test_it_is_not_asked_again_for_every_reply(self):
        """hosting() calls build() on every single spoken reply."""
        answer, asked = self._ollama({"models": [{"name": "q:7b"}]},
                                     {"q:7b": ["tools"]})
        with mock.patch.object(providers, "_ollama", answer):
            providers.local({})
            first = len(asked)
            providers.local({})
            providers.local({})
            self.assertEqual(len(asked), first, "asked ollama again")

    def _with_local(self, env):
        answer, _ = self._ollama({"models": [{"name": "qwen2.5:7b"}]},
                                 {"qwen2.5:7b": ["tools"]})
        base = {"LLM_API_KEY": "k", "LLM_BASE_URL": "https://api.groq.com/openai/v1",
                "LLM_MODEL": "openai/gpt-oss-120b"}
        with mock.patch.object(providers, "_ollama", answer):
            return providers.build({**base, **env})

    def test_the_hosted_model_answers_first_by_default(self):
        """It is better with twenty six tools. Local is what answers when the
        allowance has run out, which is the thing that was going wrong."""
        found = self._with_local({})
        self.assertEqual([p.name for p in found], ["groq", "ollama"])

    def test_preferring_local_puts_it_first(self):
        found = self._with_local({"REO_PREFER_LOCAL": "1"})
        self.assertEqual([p.name for p in found][0], "ollama")

    def test_the_local_one_needs_no_key_in_env(self):
        found = self._with_local({})
        here = next(p for p in found if p.name == "ollama")
        self.assertTrue(here.key, "the client needs some string for a key")
        self.assertIn("localhost", here.base_url)
        self.assertTrue(here.base_url.endswith("/v1"))

    def test_the_voice_is_never_asked_of_it(self):
        """Ollama does text, not speech, and Orpheus is Groq's alone."""
        answer, _ = self._ollama({"models": [{"name": "qwen2.5:7b"}]},
                                 {"qwen2.5:7b": ["tools"]})
        env = {"LLM_API_KEY": "k", "LLM_BASE_URL": "https://api.groq.com/openai/v1",
               "LLM_MODEL": "m", "REO_PREFER_LOCAL": "1"}
        with mock.patch.object(providers, "_ollama", answer):
            voice = providers.hosting("canopylabs/orpheus-v1-english", env)
        self.assertIsNotNone(voice)
        self.assertNotEqual(voice.name, "ollama")
        self.assertIn("groq", voice.base_url)

    def test_a_rate_limited_groq_falls_through_to_it(self):
        """The whole point: an answer instead of an apology."""
        spoke = []
        brain = Brain.__new__(Brain)
        brain.history = []
        brain._clients = {}
        brain.pool = providers.Pool([
            providers.Provider("groq", "k", "https://api.groq.com/openai/v1", "m"),
            providers.Provider("ollama", "x", "http://localhost:11434/v1", "qwen2.5:7b"),
        ])

        def ask(provider, client, with_tools):
            spoke.append(provider.name)
            if provider.name == "groq":
                raise RateLimitError("429", response=SimpleNamespace(
                    status_code=429, request=None, headers={}), body=None)
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(tool_calls=None, content="[neutral] Done."),
                finish_reason="stop")])

        brain._ask = ask
        brain._client = lambda index, provider: None
        answer = brain._complete(with_tools=True)
        self.assertEqual(spoke, ["groq", "ollama"])
        self.assertEqual(answer.choices[0].message.content, "[neutral] Done.")

    def test_reasoning_effort_is_not_sent_to_it(self):
        """That setting is a gpt-oss one and other models reject it."""
        sent = {}
        brain = Brain.__new__(Brain)
        brain.history = []
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kw: sent.update(kw) or SimpleNamespace(
                usage=None, choices=[]))))
        here = providers.Provider("ollama", "x", "http://localhost:11434/v1",
                                  "qwen2.5:7b")
        with mock.patch.object(providers, "remember", lambda p: None):
            brain._ask(here, client, with_tools=False)
        self.assertNotIn("reasoning_effort", sent)


class TheMomentBeforeTheKey(unittest.TestCase):
    """Keeping the 0.8s before the key press is what stops the first word
    being clipped. Letting it decide he had STARTED talking was a bug: he got
    1.5s to begin instead of six, and then "didn't catch that"."""

    def _waited(self, voice_before_the_press):
        """Seconds of live audio reo sat through before giving up."""
        import sounddevice as sd
        size = listen.BLOCK_SAMPLES
        pre = [np.zeros((size, 1), dtype=np.float32) for _ in range(12)]
        seen = {"blocks": 0}

        class Stream:
            def start(self): pass
            def stop(self): pass
            def close(self): pass
            def read(self, n):
                time.sleep(0.001)
                return np.zeros((size, 1), dtype=np.float32), False

        listener = Listener.__new__(Listener)
        listener.language, listener.detected = "en", True
        listener.transcribe_seconds = 0.0
        listener._guess = None
        listener.missed = ""
        listener._mic = Microphone(keep_open=False)
        listener._voice = SimpleNamespace(ready=True, reset=lambda: None)
        listener._transcribe = lambda raw: ""

        def ears(block, mid_sentence):
            seen["blocks"] += 1
            if seen["blocks"] <= len(pre):
                return voice_before_the_press
            return False                     # he has not started yet

        listener._voiced = ears
        begin = Microphone.start

        def with_preroll(self):
            got = begin(self)
            return pre if got is not None else got

        with mock.patch.object(sd, "InputStream", lambda **kw: Stream()), \
             mock.patch.object(Microphone, "start", with_preroll), \
             mock.patch.object(config, "WAIT_FOR_SPEECH_SECONDS", 1.2):
            listener._record(stop_now=lambda: False)
        return (seen["blocks"] - len(pre)) * listen.BLOCK_SECONDS, listener

    def test_a_voice_before_the_press_does_not_start_the_clock(self):
        quiet, _ = self._waited(voice_before_the_press=False)
        noisy, _ = self._waited(voice_before_the_press=True)
        self.assertAlmostEqual(noisy, quiet, delta=0.3,
                               msg="the pre-roll cut his thinking time short")

    def test_he_still_gets_the_whole_window(self):
        waited, _ = self._waited(voice_before_the_press=True)
        self.assertGreater(waited, 1.0)

    def test_and_it_is_still_kept_so_the_first_word_survives(self):
        """The pre-roll audio has to end up in the recording even though it
        no longer votes on whether he was speaking."""
        import sounddevice as sd
        size = listen.BLOCK_SAMPLES
        pre = [np.full((size, 1), 0.2, dtype=np.float32) for _ in range(4)]
        calls = {"n": 0}

        class Stream:
            def start(self): pass
            def stop(self): pass
            def close(self): pass
            def read(self, n):
                time.sleep(0.001)
                return np.full((size, 1), 0.3, dtype=np.float32), False

        listener = Listener.__new__(Listener)
        listener.language, listener.detected = "en", True
        listener.transcribe_seconds = 0.0
        listener._guess = None
        listener.missed = ""
        listener._mic = Microphone(keep_open=False)
        listener._voice = SimpleNamespace(ready=True, reset=lambda: None)
        listener._transcribe = lambda raw: ""

        def ears(block, mid_sentence):
            calls["n"] += 1
            return calls["n"] <= 10          # speaking, then stops

        listener._voiced = ears
        begin = Microphone.start

        def with_preroll(self):
            got = begin(self)
            return pre if got is not None else got

        with mock.patch.object(sd, "InputStream", lambda **kw: Stream()), \
             mock.patch.object(Microphone, "start", with_preroll):
            audio = listener._record(stop_now=lambda: False)
        self.assertIsNotNone(audio)
        self.assertGreaterEqual(len(audio), size * len(pre),
                                "the pre-roll was dropped from the recording")


class WhyItWasMissed(unittest.TestCase):
    """A muted microphone, a silent room, a mumble and whisper inventing
    words all printed "didn't catch that", and they need different fixes."""

    def _listener(self, audio, transcribes=""):
        listener = Listener.__new__(Listener)
        listener.language, listener.detected = "en", True
        listener.transcribe_seconds = 0.0
        listener._guess = None
        listener.missed = ""
        listener.last_voice = 1.0
        listener._record = lambda stop_now=None: audio
        listener._transcribe = lambda raw: transcribes
        return listener

    def test_a_dead_device_says_so(self):
        listener = self._listener(np.zeros(16000, dtype=np.float32))
        listener.listen()
        self.assertEqual(listener.missed, "silent")

    def test_a_mumble_is_called_quiet(self):
        faint = np.full(16000, 0.005, dtype=np.float32)
        listener = self._listener(faint, transcribes="")
        listener.listen()
        self.assertEqual(listener.missed, "quiet")

    def test_whisper_inventing_words_is_called_garbled(self):
        self.enterContext(mock.patch.object(config, "HEARING", "local"))
        listener = self._listener(np.full(16000, 0.07, dtype=np.float32))
        listener._transcribe = Listener._transcribe.__get__(listener)
        listener._voice = SimpleNamespace(ready=True)
        listener._model_lock = threading.Lock()
        listener.model = SimpleNamespace(transcribe=lambda *a, **kw: (
            iter([SimpleNamespace(text=" again and again and again and again",
                                  no_speech_prob=0.0)]), None))
        listener.listen()
        self.assertEqual(listener.missed, "garbled")

    def test_a_good_command_leaves_no_complaint(self):
        listener = self._listener(np.full(16000, 0.07, dtype=np.float32),
                                  transcribes="open chrome")
        self.assertEqual(listener.listen(), "open chrome")
        self.assertEqual(listener.missed, "")

    def test_every_reason_has_something_to_say(self):
        for reason in ("silent", "nothing", "noise", "quiet", "garbled", "unclear"):
            self.assertIn(reason, listen.MISSED)
            self.assertTrue(listen.MISSED[reason].strip())


class AskingHowHeSounds(unittest.TestCase):
    """"Can you hear me" should be answered by measurement, not by the model
    guessing - and measured on what he just said, not on a fresh recording."""

    def _report(self, level, voice):
        listener = Listener.__new__(Listener)
        listener.last_level, listener.last_voice = level, voice
        return listener.how_i_hear_you()

    def test_a_muted_microphone_is_named_as_such(self):
        said = self._report(0.0, 0.0)
        self.assertIn("silence", said.lower())
        self.assertIn("muted", said.lower())

    def test_a_room_with_no_voice_in_it(self):
        said = self._report(0.004, 0.0)
        self.assertIn("no voice", said.lower())

    def test_too_quiet_tells_him_what_to_do(self):
        said = self._report(0.008, 1.0)
        self.assertTrue("closer" in said.lower() or "turn the microphone up" in said.lower())

    def test_a_normal_voice_is_reassuring(self):
        said = self._report(0.07, 2.0)
        self.assertIn("clearly", said.lower())
        self.assertIn("nothing wrong", said.lower())

    def test_the_numbers_are_in_there(self):
        """So he can tell "a bit quiet" from "nearly nothing"."""
        self.assertIn("0.020", self._report(0.02, 1.4))

    def test_it_never_reaches_the_model(self):
        said = []
        listener = Listener.__new__(Listener)
        listener.last_level, listener.last_voice = 0.07, 2.0
        listener.transcribe_seconds = 0.0
        brain = SimpleNamespace(history=[], think=mock.Mock(
            side_effect=AssertionError("the model was asked")))
        with mock.patch.object(main, "speak",
                               lambda text, ui=None, until=None: said.append(text)):
            main.handle(brain, "can you hear me", None, 0.0, listener, None)
        brain.think.assert_not_called()
        self.assertTrue(said and "clearly" in said[0].lower())

    def test_the_usual_ways_of_asking_are_covered(self):
        for said in ["can you hear me", "Can you hear me?", "mic check",
                     "microphone check", "how do i sound", "test the microphone"]:
            with self.subTest(said=said):
                self.assertIn(normalize(said), main.HEARING_CHECK)


class SearchingSomewhereElse(unittest.TestCase):
    """Measured on a live run: duckduckgo, bing and brave were all answering
    "No results found" while google answered fine. ddgs picks for itself when
    asked for nothing in particular, so about a third of searches came back
    empty and reo said it could not find anything."""

    def setUp(self):
        tools._search_cache.clear()
        self.addCleanup(tools._search_cache.clear)

    def _searching(self, works):
        """works: the backends that have anything to say. None = the default."""
        asked = []

        def attempt(query, backend=None):
            asked.append(backend)
            if backend in works:
                return [{"title": f"from {backend}", "body": "the answer"}]
            return []

        return attempt, asked

    def test_the_default_is_tried_first_and_usually_enough(self):
        attempt, asked = self._searching({None})
        with mock.patch.object(tools, "_results", attempt):
            out = tools.search_web("how tall is everest")
        self.assertIn("the answer", out)
        self.assertEqual(asked, [None], "asked elsewhere when it did not need to")

    def test_it_walks_on_when_the_default_has_nothing(self):
        attempt, asked = self._searching({"google"})
        with mock.patch.object(tools, "_results", attempt):
            out = tools.search_web("population of hyderabad")
        self.assertIn("from google", out)
        self.assertEqual(asked[0], None)
        self.assertIn("google", asked)

    def test_it_keeps_walking_past_a_dead_one(self):
        attempt, asked = self._searching({"bing"})
        with mock.patch.object(tools, "_results", attempt):
            out = tools.search_web("price of gold")
        self.assertIn("from bing", out)
        self.assertGreater(len(asked), 2, "gave up before reaching bing")

    def test_when_nothing_anywhere_has_it_he_is_told_where_it_looked(self):
        attempt, asked = self._searching(set())
        with mock.patch.object(tools, "_results", attempt):
            out = tools.search_web("zzz nonsense query")
        self.assertIn("No results", out)
        self.assertIn("google", out)
        self.assertEqual(len(asked), 1 + len(tools.SEARCH_BACKENDS))

    def test_a_backend_that_throws_is_just_one_that_did_not_answer(self):
        """So one library error does not end the whole search."""
        class Broken:
            def text(self, *args, **kwargs):
                raise RuntimeError("the library fell over")

        with mock.patch.object(tools, "_searcher", Broken()):
            self.assertEqual(tools._results("x"), [])
            self.assertEqual(tools._results("x", "google"), [])

    def test_the_answer_is_remembered(self):
        attempt, asked = self._searching({None})
        with mock.patch.object(tools, "_results", attempt):
            tools.search_web("same question")
            tools.search_web("Same Question")
        self.assertEqual(len(asked), 1, "paid for the same search twice")

    def test_an_empty_answer_is_not_remembered(self):
        """Or a backend being down for one minute poisons that query for the
        rest of the session."""
        attempt, _ = self._searching(set())
        with mock.patch.object(tools, "_results", attempt):
            tools.search_web("something")
        self.assertNotIn("something", tools._search_cache)

    def test_there_is_a_timeout_on_it(self):
        """Nothing above run_tool has one, so a search that never came back
        would hang the whole assistant mid-reply."""
        self.assertTrue(getattr(tools, "SEARCH_TIMEOUT", 0) > 0)


class HearingThroughGroq(unittest.TestCase):
    """whisper-large-v3-turbo is on his key and is the biggest whisper there
    is: 0.27s from his last word to the words being ready, against 3.37s for
    the small model running here. The local one stays loaded as the fallback."""

    def _listener(self, hosted_says=None, blows_up=False, local_says="local heard it"):
        listener = Listener.__new__(Listener)
        listener.language, listener.detected = "en", True
        listener.missed = ""
        listener._voice = SimpleNamespace(ready=True)
        listener._here = lambda clean: local_says
        asked = {}

        def ears():
            if blows_up:
                raise RuntimeError("no network")
            return SimpleNamespace(audio=SimpleNamespace(
                transcriptions=SimpleNamespace(create=make)))

        def make(**kwargs):
            asked.update(kwargs)
            if isinstance(hosted_says, Exception):
                raise hosted_says
            return hosted_says

        listener._ears = ears
        return listener, asked

    @staticmethod
    def _answer(text, segments=None):
        return SimpleNamespace(text=text, segments=segments)

    def test_it_hears_through_groq_when_asked_to(self):
        listener, asked = self._listener(self._answer(" Open chrome"))
        with mock.patch.object(config, "HEARING", "hosted"):
            self.assertEqual(listener._transcribe(
                np.full(16000, 0.07, dtype=np.float32)), "Open chrome")
        self.assertEqual(asked.get("model"), config.HOSTED_WHISPER)

    def test_local_only_never_sends_anything(self):
        """REO_HEARING=local has to mean nothing leaves the laptop."""
        listener, asked = self._listener(self._answer(" should not be used"))
        with mock.patch.object(config, "HEARING", "local"):
            self.assertEqual(listener._transcribe(
                np.full(16000, 0.07, dtype=np.float32)), "local heard it")
        self.assertEqual(asked, {}, "it called out to Groq anyway")

    def test_no_network_falls_back_to_the_model_here(self):
        listener, _ = self._listener(blows_up=True)
        with mock.patch.object(config, "HEARING", "hosted"):
            self.assertEqual(listener._transcribe(
                np.full(16000, 0.07, dtype=np.float32)), "local heard it")

    def test_groq_refusing_falls_back_too(self):
        listener, _ = self._listener(RuntimeError("429"))
        with mock.patch.object(config, "HEARING", "hosted"):
            self.assertEqual(listener._transcribe(
                np.full(16000, 0.07, dtype=np.float32)), "local heard it")

    def test_segments_groq_calls_silence_are_dropped(self):
        """The same filter as the local path, on the same field."""
        listener, _ = self._listener(self._answer("", segments=[
            {"text": " Open it", "no_speech_prob": 0.1},
            {"text": " Thanks for watching!", "no_speech_prob": 0.95}]))
        with mock.patch.object(config, "HEARING", "hosted"):
            self.assertEqual(listener._transcribe(
                np.full(16000, 0.07, dtype=np.float32)), "Open it")

    def test_segments_as_objects_work_too(self):
        """The SDK hands back models, not dicts, depending on the version."""
        listener, _ = self._listener(self._answer("", segments=[
            SimpleNamespace(text=" Open it", no_speech_prob=0.1)]))
        with mock.patch.object(config, "HEARING", "hosted"):
            self.assertEqual(listener._transcribe(
                np.full(16000, 0.07, dtype=np.float32)), "Open it")

    def test_whisper_inventing_words_is_still_caught(self):
        listener, _ = self._listener(
            self._answer(" again and again and again and again"))
        with mock.patch.object(config, "HEARING", "hosted"):
            self.assertEqual(listener._transcribe(
                np.full(16000, 0.07, dtype=np.float32)), "")
        self.assertEqual(listener.missed, "garbled")

    def test_the_sticky_guess_is_not_forced_on_it(self):
        """That guess exists because working it out here is expensive. Groq
        does it for nothing and better, and forcing it would push his Telugu
        through an English model all session."""
        listener, asked = self._listener(self._answer(" hello"))
        listener.language, listener.detected = "en", True      # a guess
        with mock.patch.object(config, "HEARING", "hosted"), \
             mock.patch.object(config, "LANGUAGE", None):
            listener._transcribe(np.full(16000, 0.07, dtype=np.float32))
        self.assertNotIn("language", asked)

    def test_a_language_he_pinned_himself_is_sent(self):
        listener, asked = self._listener(self._answer(" namaste"))
        with mock.patch.object(config, "HEARING", "hosted"), \
             mock.patch.object(config, "LANGUAGE", "hi"):
            listener._transcribe(np.full(16000, 0.07, dtype=np.float32))
        self.assertEqual(asked.get("language"), "hi")

    def test_his_vocabulary_goes_with_it(self):
        """Proper nouns are most of what whisper gets wrong."""
        listener, asked = self._listener(self._answer(" hello"))
        with mock.patch.object(config, "HEARING", "hosted"):
            listener._transcribe(np.full(16000, 0.07, dtype=np.float32))
        self.assertIn("Hyderabad", asked.get("prompt", ""))

    def test_a_recording_is_sent_as_a_wav(self):
        listener, asked = self._listener(self._answer(" hello"))
        with mock.patch.object(config, "HEARING", "hosted"):
            listener._transcribe(np.full(16000, 0.07, dtype=np.float32))
        name, payload, kind = asked["file"]
        self.assertTrue(name.endswith(".wav"))
        self.assertEqual(kind, "audio/wav")
        self.assertTrue(payload.startswith(b"RIFF"))

    def test_the_wav_really_is_the_recording(self):
        import wave as wavelib
        tone = (np.sin(np.linspace(0, 400, 16000)) * 0.5).astype(np.float32)
        with wavelib.open(io.BytesIO(au_mod.as_wav(tone)), "rb") as f:
            self.assertEqual(f.getframerate(), 16000)
            self.assertEqual(f.getnchannels(), 1)
            back = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
        self.assertEqual(len(back), len(tone))
        self.assertLess(float(np.abs(back / 32767 - tone).max()), 0.001)

    def test_the_hosted_ear_is_built_once(self):
        """Per call it cost 2.69s cold against 0.20s warm."""
        made = []
        listener = Listener.__new__(Listener)
        with mock.patch("openai.OpenAI", lambda **kw: made.append(kw) or object()), \
             mock.patch.object(providers, "hosting",
                               lambda model, env=None: providers.Provider(
                                   "groq", "k", "https://api.groq.com/openai/v1", "m")):
            listener._ears()
            listener._ears()
            listener._ears()
        self.assertEqual(len(made), 1, "rebuilt the client every time")

    def test_whisper_is_asked_of_groq_and_not_the_laptop(self):
        """ollama does not host whisper, and must never be handed the audio."""
        here = providers.Provider("ollama", "x", "http://localhost:11434/v1", "q")
        env = {"LLM_API_KEY": "k", "LLM_BASE_URL": "https://api.groq.com/openai/v1",
               "LLM_MODEL": "m", "REO_PREFER_LOCAL": "1"}
        with mock.patch.object(providers, "local", lambda env=None: here):
            host = providers.hosting("whisper-large-v3-turbo", env)
        self.assertIsNotNone(host)
        self.assertNotEqual(host.name, "ollama")


class EnoughFetching(unittest.TestCase):
    """Every tool has its own timeout but nothing added them up, and the
    model can ask for round after round. One reply in his log took 18.5
    seconds, which from where he is sitting is reo having died."""

    def _brain(self, rounds, each_takes):
        """A brain that asks for one tool per round, each taking `each_takes`."""
        ran = []
        brain = Brain.__new__(Brain)
        brain.history = []
        brain._clients = {}
        brain.pool = providers.Pool([
            providers.Provider("p", "k", "https://p.invalid", "m")])

        asked = {"left": rounds}

        def complete(with_tools):
            if asked["left"] > 0:
                asked["left"] -= 1
                call = SimpleNamespace(id=f"c{asked['left']}", function=SimpleNamespace(
                    name="search_web", arguments='{"query": "x"}'))
                return SimpleNamespace(choices=[SimpleNamespace(
                    message=SimpleNamespace(tool_calls=[call], content=""),
                    finish_reason="stop")])
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(tool_calls=None, content="[neutral] Done."),
                finish_reason="stop")])

        brain._complete = complete

        def slow(name, args):
            ran.append(name)
            time.sleep(each_takes)
            return "some result"

        return brain, ran, slow

    def test_a_quick_reply_is_not_interfered_with(self):
        brain, ran, slow = self._brain(rounds=2, each_takes=0.01)
        import reo.brain as brainmod
        with mock.patch.object(brainmod, "run_tool", slow):
            self.assertEqual(brain.think("go"), "[neutral] Done.")
        self.assertEqual(len(ran), 2, "it stopped fetching when it had time left")

    def test_it_stops_fetching_once_the_budget_is_gone(self):
        brain, ran, slow = self._brain(rounds=6, each_takes=0.05)
        import reo.brain as brainmod
        with mock.patch.object(brainmod, "run_tool", slow), \
             mock.patch.object(config, "TOOL_BUDGET", 0.12):
            began = time.time()
            self.assertEqual(brain.think("go"), "[neutral] Done.")
            took = time.time() - began
        self.assertLess(len(ran), 6, "it kept fetching past the budget")
        self.assertLess(took, 1.0)

    def test_the_calls_still_get_an_answer_each(self):
        """Every tool_calls entry must have a result beside it or the next
        request is a 400: that is what broke reo for a whole evening.

        Trimming is held off, or there is nothing left to look at and this
        passes by comparing two empty lists.
        """
        brain, ran, slow = self._brain(rounds=5, each_takes=0.05)
        import reo.brain as brainmod
        with mock.patch.object(brainmod, "run_tool", slow), \
             mock.patch.object(Brain, "_trim_history", lambda self: None), \
             mock.patch.object(config, "TOOL_BUDGET", 0.08):
            brain.think("go")
        wanted = [c["id"] for m in brain.history
                  for c in (m.get("tool_calls") or [])]
        answered = [m["tool_call_id"] for m in brain.history
                    if m.get("role") == "tool"]
        self.assertEqual(sorted(wanted), sorted(answered),
                         "a tool call was left without a result")

    def test_it_is_told_to_answer_rather_than_told_nothing(self):
        brain, ran, slow = self._brain(rounds=4, each_takes=0.05)
        import reo.brain as brainmod
        with mock.patch.object(brainmod, "run_tool", slow), \
             mock.patch.object(Brain, "_trim_history", lambda self: None), \
             mock.patch.object(config, "TOOL_BUDGET", 0.08):
            brain.think("go")
        said = [m["content"] for m in brain.history if m.get("role") == "tool"]
        self.assertTrue(any("Answer now" in s for s in said),
                        f"the model was given no reason to stop: {said}")

    def test_the_budget_is_a_sane_length(self):
        """Long enough for a real search and a page read, short enough that
        it is not reo having died."""
        self.assertGreaterEqual(config.TOOL_BUDGET, 10)
        self.assertLessEqual(config.TOOL_BUDGET, 30)


class WhichEarHeardIt(unittest.TestCase):
    """The log is how a complaint about lag gets diagnosed later."""

    def _heard(self, where):
        listener = Listener.__new__(Listener)
        listener.transcribe_seconds = 0.4
        listener.heard_by = where
        return main._heard_detail(3.2, listener)

    def test_it_says_when_groq_heard_him(self):
        self.assertIn("groq", self._heard("groq"))

    def test_it_says_when_the_laptop_did(self):
        self.assertIn("here", self._heard("here"))

    def test_the_old_shape_still_reads(self):
        self.assertIn("you 2.8s", self._heard("groq"))

    def test_nothing_transcribed_says_nothing_extra(self):
        listener = Listener.__new__(Listener)
        listener.transcribe_seconds = 0.0
        self.assertEqual(main._heard_detail(1.0, listener), "heard 1.0s")

    def test_the_fallback_is_recorded_as_here(self):
        listener = Listener.__new__(Listener)
        listener.language, listener.detected = "en", True
        listener.missed = ""
        listener._voice = SimpleNamespace(ready=True)
        listener._hosted = lambda clean: None          # groq could not
        listener._here = lambda clean: "the laptop heard it"
        with mock.patch.object(config, "HEARING", "hosted"):
            listener._transcribe(np.full(16000, 0.07, dtype=np.float32))
        self.assertEqual(listener.heard_by, "here")

    def test_groq_answering_is_recorded_as_groq(self):
        listener = Listener.__new__(Listener)
        listener.language, listener.detected = "en", True
        listener.missed = ""
        listener._voice = SimpleNamespace(ready=True)
        listener._hosted = lambda clean: "groq heard it"
        listener._here = lambda clean: "should not be used"
        with mock.patch.object(config, "HEARING", "hosted"):
            listener._transcribe(np.full(16000, 0.07, dtype=np.float32))
        self.assertEqual(listener.heard_by, "groq")


if __name__ == "__main__":
    unittest.main()
