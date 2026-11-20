"""Offline tests: no microphone, no API key, no network."""
import datetime
import contextlib
import sys
import io
import json
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

from reo import apps, config, fastpath, providers, tools
from reo import memory as memory_module
from reo import brain as brain_module
from reo import stats as stats_module
from reo import chats as chatsmod
from reo.chats import Chats
from reo import pointer
from reo import webui
from reo import shell as shell_module
from reo import undo
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
from reo.reply import tighten
from reo import about



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


# The suite drives handle(), and handle() records every exchange and lets the
# model remember things. Both of those went to his own files: "write me a
# game" turned up in his chats.jsonl twice, and earlier runs left facts in
# his memory.json. Everything is pointed at a temp folder for the whole run,
# here rather than per test, so a test added later cannot forget to.
_elsewhere = []


def setUpModule():
    home = Path(tempfile.mkdtemp(prefix="reo-tests-"))
    scratch_memory = Memory(home / "memory.json")
    scratch_chats = Chats(home / "chats.jsonl")
    for where, name, thing in [
            (memory_module, "memory", scratch_memory),
            (brain_module, "memory", scratch_memory),
            (chatsmod, "chats", scratch_chats),
            (main, "chats", scratch_chats),
            (tools, "chats", scratch_chats)]:
        patch = mock.patch.object(where, name, thing)
        patch.start()
        _elsewhere.append(patch)
    _elsewhere.append(home)


def tearDownModule():
    while _elsewhere:
        going = _elsewhere.pop()
        if isinstance(going, Path):
            shutil.rmtree(going, ignore_errors=True)
        else:
            going.stop()



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

    def test_a_wrong_keyword_is_reported_rather_than_raised(self):
        """The model inventing a parameter is its mistake, and run_tool hands
        it back so it can try again."""
        for name, args in [("open_app", {"app": "notepad"}),
                           ("get_datetime", {"tz": "UTC"})]:
            with self.subTest(name=name):
                self.assertIn("failed", tools.run_tool(name, args))

    def test_a_missing_argument_says_what_is_missing(self):
        """This used to say "Tool search_web failed: ... object has no
        attribute 'strip'", and that reached him as the answer."""
        said = tools.run_tool("search_web", {})
        self.assertNotIn("failed", said)
        self.assertIn("what to search for", said)

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


class PuttingItBack(unittest.TestCase):
    """Reo is driven by speech and speech mishears. "Turn it down" comes back
    as "turn it up" and the fix should be two words, not a trip to the
    settings."""

    def setUp(self):
        undo.forget()
        self.addCleanup(undo.forget)

    def _user32(self, handles=(1111, 2222)):
        fake = mock.MagicMock()
        fake.GetForegroundWindow.side_effect = list(handles)
        fake.GetWindowTextLengthW.return_value = 7
        fake.GetWindowTextW.side_effect = (
            lambda h, buf, n: buf.__setitem__(slice(0, 7), "Notepad"))
        return fake

    def test_nothing_done_means_nothing_to_undo(self):
        self.assertIn("nothing", undo.undo().lower())

    def test_the_volume_goes_back(self):
        taps = []
        with mock.patch.object(reo_system, "_tap",
                               lambda code, times=1: taps.append((code, times))):
            reo_system.control("volume_up", 3)
            self.assertEqual(undo.waiting(), "volume up")
            undo.undo()
        self.assertEqual(taps, [(0xAF, 3), (0xAE, 3)],
                         "it did not put back the same number of notches")

    def test_mute_is_its_own_reverse(self):
        taps = []
        with mock.patch.object(reo_system, "_tap",
                               lambda code, times=1: taps.append(code)):
            reo_system.control("mute")
            undo.undo()
        self.assertEqual(taps, [0xAD, 0xAD])

    def test_undoing_twice_does_not_do_it_again(self):
        """Putting the volume back calls control() like anything else, so
        without care it recorded itself and "undo" twice turned it up again."""
        taps = []
        with mock.patch.object(reo_system, "_tap",
                               lambda code, times=1: taps.append(code)):
            reo_system.control("volume_up", 1)
            undo.undo()
            second = undo.undo()
        self.assertEqual(len(taps), 2, "the undo was itself undone")
        self.assertIn("nothing", second.lower())

    def test_the_window_it_minimised_is_the_one_it_puts_back(self):
        """By the time he changes his mind something else is in front, and
        restoring "the window in front" would be the wrong window."""
        shown = []
        fake = self._user32()
        fake.ShowWindow.side_effect = lambda h, cmd: shown.append((h, cmd))
        with mock.patch.object(reo_system, "_user32", lambda: fake):
            reo_system.control("minimise")
            undo.undo()
        self.assertEqual([h for h, _ in shown], [1111, 1111])
        self.assertEqual(shown[-1][1], reo_system.SHOW["restore"])

    def test_locking_cannot_be_undone_and_clears_the_record(self):
        """Or "lock the screen" then "undo that" would quietly put the volume
        back to where it was five minutes ago."""
        fake = self._user32()
        with mock.patch.object(reo_system, "_tap", lambda code, times=1: None), \
             mock.patch.object(reo_system, "_user32", lambda: fake):
            reo_system.control("volume_up", 2)
            self.assertTrue(undo.waiting())
            reo_system.control("lock")
        self.assertEqual(undo.waiting(), "")
        self.assertIn("nothing", undo.undo().lower())

    def test_closing_a_window_cannot_be_undone_either(self):
        fake = self._user32()
        with mock.patch.object(reo_system, "_tap", lambda code, times=1: None), \
             mock.patch.object(reo_system, "_user32", lambda: fake):
            reo_system.control("volume_up", 2)
            reo_system.control("close_window")
        self.assertEqual(undo.waiting(), "")

    def test_a_file_written_over_comes_back(self):
        home = Path(tempfile.mkdtemp(prefix="reo-undo-"))
        self.addCleanup(shutil.rmtree, home, True)
        with mock.patch.object(config, "WORKSPACE", str(home)):
            workspace.write("notes.txt", "the first version")
            workspace.write("notes.txt", "the one he did not want")
            self.assertEqual((home / "notes.txt").read_text(encoding="utf-8"),
                             "the one he did not want")
            undo.undo()
        self.assertEqual((home / "notes.txt").read_text(encoding="utf-8"),
                         "the first version")

    def test_a_brand_new_file_is_removed_again(self):
        home = Path(tempfile.mkdtemp(prefix="reo-undo-"))
        self.addCleanup(shutil.rmtree, home, True)
        with mock.patch.object(config, "WORKSPACE", str(home)):
            workspace.write("fresh.txt", "only version")
            undo.undo()
        self.assertFalse((home / "fresh.txt").exists())

    def test_a_write_that_failed_is_not_offered_as_undoable(self):
        home = Path(tempfile.mkdtemp(prefix="reo-undo-"))
        self.addCleanup(shutil.rmtree, home, True)
        with mock.patch.object(config, "WORKSPACE", str(home)):
            workspace.write("../escaped.txt", "should not be written")
        self.assertEqual(undo.waiting(), "")

    def test_an_undo_that_throws_says_so_rather_than_crashing(self):
        def explode():
            raise RuntimeError("the window is gone")

        undo.remember("minimising something", explode)
        said = undo.undo()
        self.assertIn("Could not undo", said)
        self.assertEqual(undo.waiting(), "", "it stayed on the hook")

    def test_saying_it_never_reaches_the_model(self):
        for said in ["undo", "undo that", "undo it", "put that back",
                     "take it back", "can you undo that", "reverse that",
                     "undo the last thing"]:
            with self.subTest(said=said):
                self.assertIsNotNone(fastpath.answer(said))

    def test_talking_about_undo_still_reaches_the_model(self):
        for said in ["undo my homework", "how do i undo this in word",
                     "whats the undo shortcut", "undo button"]:
            with self.subTest(said=said):
                self.assertIsNone(fastpath.answer(said))


class KnowingWhatItIs(unittest.TestCase):
    """From his own log: asked who made it, reo said OpenAI three times and
    kept saying it after he corrected it. Asked how to end the program, it
    told him to say "stop", which is not one of the words that works.

    Guessing about yourself goes wrong in a way that guessing about the
    weather does not, and none of it needs a model.
    """

    def test_he_made_it(self):
        said = fastpath.answer("who created you")
        self.assertIsNotNone(said)
        self.assertIn("you did", said.lower())

    def test_it_names_no_company(self):
        """The one thing the system prompt asks of it, which it ignored."""
        for asked in ["who created you", "who made you", "who built you",
                      "who are you", "what model are you using",
                      "what are you built on", "what can you do"]:
            with self.subTest(asked=asked):
                said = (fastpath.answer(asked) or "").lower()
                for company in ("openai", "anthropic", "google", "meta", "groq's gpt"):
                    self.assertNotIn(company, said)

    def test_the_words_it_gives_for_stopping_actually_work(self):
        """It told him to say "stop". That does nothing."""
        said = fastpath.answer("how do i stop you")
        self.assertIsNotNone(said)
        offered = {w.strip(" .") for w in
                   said.lower().split("say ")[1].split(".")[0].replace(" or ", ", ").split(",")}
        self.assertTrue(offered)
        for word in offered:
            self.assertIn(word, main.EXIT_PHRASES,
                          f"it offers {word!r}, which does nothing")

    def test_it_knows_how_it_is_summoned(self):
        said = fastpath.answer("who are you")
        with mock.patch.object(config, "MODE", "hotkey"):
            self.assertIn("ctrl", fastpath.answer("who are you").lower())
        self.assertIn("reo", said.lower())

    def test_it_knows_which_model_is_answering(self):
        with mock.patch.object(providers, "active_label", lambda: "qwen2.5:7b"):
            said = fastpath.answer("what model are you using")
        self.assertIn("qwen2.5:7b", said)

    def test_it_knows_where_it_writes(self):
        with mock.patch.object(config, "WORKSPACE", r"C:\somewhere\else"):
            self.assertIn(r"C:\somewhere\else",
                          fastpath.answer("where do you put the files"))

    def test_it_says_undo_exists(self):
        """He never knew, because it was never mentioned anywhere he would
        hear it."""
        self.assertIn("undo", fastpath.answer("what can you do").lower())

    def test_none_of_it_reaches_the_model(self):
        for asked in ["who created you", "who are you", "what can you do",
                      "how do i stop you", "what model are you using",
                      "where do you put the files", "whats your name",
                      "tell me about yourself", "how do i turn you off"]:
            with self.subTest(asked=asked):
                self.assertIsNotNone(fastpath.answer(asked),
                                     f"{asked!r} still costs a round trip")

    def test_questions_about_other_things_are_left_alone(self):
        for asked in ["who created python", "who made the iphone",
                      "what can you do with a raspberry pi",
                      "how do i stop the rain", "what model is this car",
                      "who built the taj mahal", "where do you put the bins"]:
            with self.subTest(asked=asked):
                self.assertIsNone(fastpath.answer(asked),
                                  f"{asked!r} was answered as if it were about reo")

    def test_asking_about_itself_is_tried_before_anything_else(self):
        """"What can you do" must not be read as a command to do something."""
        self.assertIn("ABOUT", open(
            Path(__file__).resolve().parent.parent / "reo" / "fastpath.py",
            encoding="utf-8").read().split("COMPILED = ")[1][:80])


class OneNameNotThree(unittest.TestCase):
    """His memory ended up holding "User's name is Nandan", "User's name is
    Byron" and "User's name is Nanan Sai" at once - two written by test runs
    of mine, one a mishearing - because each was a different enough string to
    count as new. Reo had three names for him and picked whichever it liked.
    """

    def _memory(self):
        home = Path(tempfile.mkdtemp(prefix="reo-mem-"))
        self.addCleanup(shutil.rmtree, home, True)
        return Memory(home / "m.json")

    def test_a_fact_that_opens_the_same_way_replaces_the_old_one(self):
        m = self._memory()
        m.remember("User's name is Nandan")
        said = m.remember("User's name is Byron")
        self.assertIn("Changed that", said)
        self.assertEqual(len(m.facts), 1)
        self.assertIn("Byron", m.recall())
        self.assertNotIn("Nandan", m.recall())

    def test_a_different_fact_is_kept_beside_it(self):
        m = self._memory()
        m.remember("User's name is Nandan")
        m.remember("He lives in Hyderabad")
        self.assertEqual(len(m.facts), 2)

    def test_saying_the_same_thing_twice_changes_nothing(self):
        m = self._memory()
        m.remember("User's name is Nandan")
        said = m.remember("User's name is Nandan")
        self.assertIn("Already knew", said)
        self.assertEqual(len(m.facts), 1)

    def test_a_two_word_fact_is_not_treated_as_an_update(self):
        """Too short an opening to mean anything."""
        m = self._memory()
        m.remember("likes tea")
        m.remember("likes coffee")
        self.assertEqual(len(m.facts), 2)

    def test_the_replacement_is_written_to_disk(self):
        m = self._memory()
        m.remember("User's name is Nandan")
        m.remember("User's name is Nandan Sai")
        again = Memory(m.path)
        self.assertEqual(len(again.facts), 1)
        self.assertIn("Nandan Sai", again.recall())


class RememberingTheConversation(unittest.TestCase):
    """Closing reo threw the conversation away. Ask it something on Tuesday
    and follow it up on Wednesday and it had no idea what "it" was."""

    def _chats(self):
        home = Path(tempfile.mkdtemp(prefix="reo-chats-"))
        self.addCleanup(shutil.rmtree, home, True)
        return Chats(home / "c.jsonl")

    def test_an_exchange_survives_being_closed(self):
        store = self._chats()
        store.add("my sisters name is anu", "Got it.")
        again = Chats(store.path)
        self.assertEqual(again.how_many(), 1)

    def test_the_last_few_come_back_as_messages(self):
        store = self._chats()
        store.add("open chrome", "Chrome is open.")
        store.add("what is the capital of australia", "Canberra.")
        carried = store.carry_over()
        self.assertEqual([m["role"] for m in carried],
                         ["user", "assistant", "user", "assistant"])
        self.assertEqual(carried[0]["content"], "open chrome")
        self.assertEqual(carried[-1]["content"], "Canberra.")

    def test_only_a_handful_is_carried(self):
        """Every carried message is resent on every request, against an 8000
        token minute. It is a few turns, not a diary."""
        store = self._chats()
        for i in range(30):
            store.add(f"question {i}", f"answer {i}")
        self.assertEqual(len(store.carry_over()), chatsmod.CARRY * 2)

    def test_something_from_another_day_says_when_it_was(self):
        """Without it, "open it again" refers to something reo thinks
        happened a minute ago."""
        store = self._chats()
        store.path.write_text(json.dumps(
            {"when": "2026-11-09T10:00:00", "you": "open chrome",
             "reo": "Done."}) + "\n", encoding="utf-8")
        said = store.carry_over()[0]["content"]
        self.assertIn("2026-11-09", said)
        self.assertIn("open chrome", said)

    def test_today_is_not_laboured(self):
        store = self._chats()
        store.add("open chrome", "Done.")
        self.assertEqual(store.carry_over()[0]["content"], "open chrome")

    def test_it_can_look_further_back_than_it_carries(self):
        store = self._chats()
        store.add("my sisters name is anu", "Got it.")
        for i in range(10):
            store.add(f"filler {i}", f"answer {i}")
        found = store.search("sister")
        self.assertIn("anu", found.lower())

    def test_searching_needs_every_word_to_match(self):
        store = self._chats()
        store.add("my sisters name is anu", "Got it.")
        store.add("open chrome", "Done.")
        self.assertIn("Nothing", store.search("sister chrome"))

    def test_nothing_found_says_so_plainly(self):
        store = self._chats()
        store.add("open chrome", "Done.")
        self.assertIn("Nothing", store.search("quantum physics"))

    def test_a_half_written_line_costs_one_exchange_not_the_file(self):
        """Reo is killed by closing its window, which can cut a line in half."""
        store = self._chats()
        store.add("first", "one")
        store.add("second", "two")
        with open(store.path, "a", encoding="utf-8") as f:
            f.write('{"when": "2026-11-15T10:00:00", "you": "third", "re')
        self.assertEqual(store.how_many(), 2)
        self.assertIn("first", store.search("first"))

    def test_an_empty_thing_said_is_not_recorded(self):
        store = self._chats()
        store.add("", "Done.")
        store.add("   ", "Done.")
        self.assertEqual(store.how_many(), 0)

    def test_no_file_yet_is_not_an_error(self):
        home = Path(tempfile.mkdtemp(prefix="reo-chats-"))
        self.addCleanup(shutil.rmtree, home, True)
        store = Chats(home / "never-written.jsonl")
        self.assertEqual(store.carry_over(), [])
        self.assertEqual(store.how_many(), 0)
        self.assertIn("Nothing", store.search("anything"))

    def test_saving_cannot_take_the_assistant_down(self):
        store = self._chats()
        with mock.patch("builtins.open", side_effect=OSError("disk full")):
            store.add("something", "Done.")      # must not raise

    def test_looking_back_is_a_tool_the_model_has(self):
        self.assertIn("past_chats", tools.FUNCTIONS)
        self.assertIn("past_chats", {spec["name"] for spec in tools.TOOLS})
        self.assertIn("past_chats", tools.READ_ONLY)

    def test_the_conversation_is_not_committed_to_the_repository(self):
        """It is his conversation."""
        ignored = (Path(__file__).resolve().parent.parent / ".gitignore"
                   ).read_text(encoding="utf-8")
        self.assertIn("chats.jsonl", ignored)


class TighteningTheReply(unittest.TestCase):
    """Reo's own prompt says "no offers of further help" and reo made them
    anyway, in 13% of the replies in his log. Each one is a sentence he has
    to sit through: measured at 2.5s by synthesising both versions."""

    def test_the_offer_at_the_end_goes(self):
        self.assertEqual(
            tighten("This phrase is from the song by The Foundations. "
                    "Would you like more information about the song?"),
            "This phrase is from the song by The Foundations.")

    def test_a_sign_off_goes_too(self):
        self.assertEqual(
            tighten("Google is now open. Let me know if you need anything else."),
            "Google is now open.")

    def test_several_of_them_go(self):
        self.assertEqual(
            tighten("Done. Anything else? Let me know."), "Done.")

    def test_a_real_question_stays(self):
        """The one thing this must not do. Each of these is reo doing its
        job, and losing it leaves him answering a question nobody asked."""
        for real in ["Which file did you mean?",
                     "Do you want Celsius or Fahrenheit?",
                     "There is already a game.py. Should I overwrite it?",
                     "I found three. Which one?",
                     "Your battery is at twelve percent. Do you need the charger?"]:
            self.assertEqual(tighten(real), real, real)

    def test_an_offer_that_is_the_whole_reply_stays(self):
        """Saying something useless beats saying nothing at all."""
        only = "Would you like me to look that up?"
        self.assertEqual(tighten(only), only)

    def test_the_warmth_is_left_alone(self):
        """Cutting the opening pleasantry too turned "Sure thing, digital
        son" into "Digital son", and all of the time is at the end anyway."""
        kept = "Sure thing, digital son - happy to be your sidekick."
        self.assertEqual(tighten(kept), kept)

    def test_running_it_twice_changes_nothing(self):
        once = tighten("Hello! How can I help you today?")
        self.assertEqual(tighten(once), once)

    def test_nothing_in_nothing_out(self):
        for empty in ["", "   ", None]:
            self.assertEqual(tighten(empty), empty)

    def test_what_is_spoken_is_what_is_trimmed(self):
        """In split_emotion, so the console line, the window and the kept
        conversation all show what was actually said out loud."""
        self.assertEqual(split_emotion("[happy] Hello! How can I help you today?"),
                         ("happy", "Hello!"))

    def test_the_brain_does_not_keep_the_offer_either(self):
        reply = _answer_once(brain_module, "Done. Anything else you'd like?")
        self.assertEqual(reply, "Done.")


class WhatHeToldItToSay(unittest.TestCase):
    """In his own words: "from now on if anyone asks who are you you should
    tell you are Nandan's baby". Reo could not obey him - about.py answers
    the identity questions without the model or the memory being consulted -
    so he asked "whose baby are you" eight times and got eight answers."""

    def _memory(self):
        home = Path(tempfile.mkdtemp(prefix="reo-mem-"))
        self.addCleanup(shutil.rmtree, home, True)
        store = Memory(home / "m.json")
        patch = mock.patch.object(memory_module, "memory", store)
        patch.start()
        self.addCleanup(patch.stop)
        return store

    def test_it_says_what_he_told_it_to(self):
        store = self._memory()
        store.remember("Reo says: I am Nandan's baby")
        self.assertIn("I am Nandan's baby", fastpath.answer("who are you"))

    def test_the_odd_question_only_he_can_have_answered(self):
        store = self._memory()
        store.remember("Reo says: I am Nandan's baby")
        for asked in ["whose baby are you", "whose son are you", "are you my baby"]:
            self.assertEqual(fastpath.answer(asked), "I am Nandan's baby.", asked)

    def test_without_an_instruction_that_question_goes_to_the_model(self):
        self._memory()
        self.assertIsNone(fastpath.answer("whose baby are you"))

    def test_the_plain_answer_is_still_there_underneath(self):
        store = self._memory()
        store.remember("Reo says: I am Nandan's baby")
        said = fastpath.answer("who are you")
        self.assertIn("voice assistant", said)

    def test_a_new_instruction_replaces_the_last_one(self):
        """He should not have to unpick a pile of them by voice."""
        store = self._memory()
        store.remember("Reo says: I am Nandan's baby")
        store.remember("Reo says: I am Nandan's assistant")
        self.assertEqual(store.about_reo(), ["I am Nandan's assistant"])

    def test_facts_about_him_are_not_facts_about_reo(self):
        store = self._memory()
        store.remember("User's name is Nandan")
        self.assertEqual(store.about_reo(), [])

    def test_who_built_it_is_not_up_for_negotiation(self):
        """It said OpenAI three times running and kept saying it after being
        corrected. That one is not his to redefine either."""
        store = self._memory()
        store.remember("Reo says: I was built by OpenAI")
        self.assertIn("You did", fastpath.answer("who created you"))

    def test_it_has_a_tool_of_its_own_for_this(self):
        """Asking the model to store it in the right shape was tried first
        and does not work. Told to use "Reo says: " it wrote "If anyone asks
        who are you, say you are Nandan's baby" one run and "When asked who
        I am, I should say I am Nandan's baby." the next. A tool is picked by
        its description, which the model is reliable about."""
        self.assertIn("answer_about_yourself", tools.FUNCTIONS)
        self.assertIn("answer_about_yourself", {s["name"] for s in tools.TOOLS})

    def test_the_tool_stores_it_where_about_can_read_it(self):
        store = self._memory()
        tools.FUNCTIONS["answer_about_yourself"]("I am Nandan's baby")
        self.assertEqual(store.about_reo(), ["I am Nandan's baby"])
        self.assertEqual(fastpath.answer("whose baby are you"), "I am Nandan's baby.")

    def test_an_empty_answer_is_not_stored(self):
        store = self._memory()
        tools.FUNCTIONS["answer_about_yourself"]("   ")
        self.assertEqual(store.facts, [])

    def test_a_remember_call_is_shaped_too_whichever_person_it_used(self):
        """The free half of this. The model phrased the same instruction in
        the second person one run and the first person the next."""
        for said in ["If anyone asks who are you, say you are Nandan's baby",
                     "If anyone asks who you are, tell them you are Nandan's baby",
                     "When asked who I am, I should say I am Nandan's baby.",
                     "When someone asks what you are, respond that you are his baby"]:
            self.assertTrue(memory_module._as_standing(said).startswith("Reo says: I am"),
                            said)

    def test_an_ordinary_fact_is_not_shaped(self):
        for plain in ["User's name is Nandan",
                      "User lives in Hyderabad",
                      "User wants you to say hello to Veda",
                      "User said you are his baby",
                      "User asked who you are"]:
            self.assertEqual(memory_module._as_standing(plain), plain, plain)


class KeepingAWrittenFileOutOfTheHistory(unittest.TestCase):
    """The one that actually bit him. Asked to write a game, reo puts the
    whole program in write_file's arguments, and that argument then rides
    along on every request for the next four turns against a cap of eight
    thousand tokens a minute. Twice in a row all he got was that the
    allowance was spent, and the second time only because of the first."""

    def test_a_long_argument_is_not_kept(self):
        code = "import pygame\n" + "x = 1\n" * 600
        sent = json.dumps({"name": "game.py", "text": code})
        kept = brain_module._slim(sent)
        self.assertLess(len(kept), len(sent) // 10)
        self.assertIn("game.py", kept)
        self.assertIn("characters", kept)

    def test_a_short_argument_is_untouched(self):
        sent = json.dumps({"query": "weather in hyderabad"})
        self.assertEqual(brain_module._slim(sent), sent)

    def test_it_is_still_json_afterwards(self):
        sent = json.dumps({"name": "a.py", "text": "y" * 5000, "open": True})
        back = json.loads(brain_module._slim(sent))
        self.assertEqual(back["name"], "a.py")
        self.assertIs(back["open"], True)

    def test_an_unparseable_argument_is_left_for_the_model_to_see(self):
        self.assertEqual(brain_module._slim("{not json"), "{not json")
        self.assertEqual(brain_module._slim(""), "")

    def test_the_tool_still_gets_the_whole_thing(self):
        """Slimming the history must not slim what is written to disk."""
        code = "print('hi')\n" * 200
        ran = {}
        brain = _brain_that_calls(brain_module, "write_file",
                                 json.dumps({"name": "a.py", "text": code}))
        with mock.patch.object(brain_module, "run_tool",
                               lambda name, args: ran.update(args) or "Wrote a.py"):
            brain.think("write me something")
        self.assertEqual(ran["text"], code)
        kept = next(m for m in brain.history if m.get("tool_calls"))
        self.assertIn("characters", kept["tool_calls"][0]["function"]["arguments"])


class WhenTheAllowanceRunsOut(unittest.TestCase):
    """He asked reo to write him a game twice, word for word, because the
    first attempt spent the allowance and the question went with it."""

    def test_how_long_comes_out_of_what_the_server_said(self):
        self.assertEqual(brain_module._how_long(
            "Limit 8000, Used 7500. Please try again in 5.1s. Visit..."),
            "another 5 seconds")
        self.assertEqual(brain_module._how_long("Please try again in 7m32.86s."),
                         "another 8 minutes")
        self.assertEqual(brain_module._how_long("Please try again in 2h3m1s."),
                         "another 2 hours")

    def test_no_number_means_no_claim_about_the_wait(self):
        self.assertEqual(brain_module._how_long("Rate limit reached."), "")
        self.assertEqual(brain_module._how_long(""), "")

    def test_he_is_told_how_long_rather_than_just_a_moment(self):
        said = []
        brain = SimpleNamespace(history=[], unanswered=None, think=mock.Mock(
            side_effect=OutOfAllowance("groq", "another 40 seconds")))
        with mock.patch.object(main, "speak",
                               lambda text, ui=None, until=None: said.append(text)):
            main.handle(brain, "write me a game", None, 0.0, None, None)
        self.assertIn("another 40 seconds", said[0])

    def test_the_question_is_kept_so_try_again_works(self):
        asked, said = [], []
        brain = SimpleNamespace(history=[], unanswered=None)
        brain.think = mock.Mock(side_effect=OutOfAllowance("groq"))
        with mock.patch.object(main, "speak",
                               lambda text, ui=None, until=None: said.append(text)):
            main.handle(brain, "write me a game", None, 0.0, None, None)
            self.assertEqual(brain.unanswered, "write me a game")
            brain.think = mock.Mock(side_effect=lambda text: asked.append(text) or "Done.")
            main.handle(brain, "try again", None, 0.0, None, None)
        self.assertEqual(asked, ["write me a game"])
        self.assertIsNone(brain.unanswered)

    def test_try_again_with_nothing_waiting_goes_to_the_model(self):
        """He may well mean "try opening it again", which the conversation
        answers and this must not swallow."""
        asked = []
        brain = SimpleNamespace(history=[], unanswered=None)
        brain.think = mock.Mock(side_effect=lambda text: asked.append(text) or "Done.")
        with mock.patch.object(main, "speak", lambda text, ui=None, until=None: None):
            main.handle(brain, "try again", None, 0.0, None, None)
        self.assertEqual(asked, ["try again"])

    def test_repeat_that_is_not_hijacked(self):
        """"say that again" means say it again, not retry the failed one."""
        for near in ["say that again", "again", "do that again", "play it again"]:
            self.assertNotIn(near, main.AGAIN)


def _answer_once(brain_module, content):
    """A Brain whose one API call returns this reply and no tool calls."""
    brain = _stub_brain(brain_module)
    brain._complete = lambda with_tools: SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=content, tool_calls=None),
        finish_reason="stop")])
    return brain.think("anything")


def _brain_that_calls(brain_module, name, arguments):
    """A Brain that asks for one tool and then says Done."""
    brain = _stub_brain(brain_module)
    turns = iter([
        SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="", tool_calls=[SimpleNamespace(
                id="c1", function=SimpleNamespace(name=name, arguments=arguments))]),
            finish_reason="tool_calls")]),
        SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="Done.", tool_calls=None),
            finish_reason="stop")]),
    ])
    brain._complete = lambda with_tools: next(turns)
    return brain


def _stub_brain(brain_module):
    brain = Brain.__new__(Brain)
    brain.pool, brain._clients, brain.history = None, {}, []
    brain.unanswered = None
    return brain


class TheTestsStayOutOfHisFiles(unittest.TestCase):
    """"write me a game" turned up in his own chats.jsonl twice, put there
    by this suite driving handle(). Two earlier runs left facts in his
    memory.json. The redirect is in setUpModule; this checks it is on."""

    def test_the_conversation_being_written_is_not_his(self):
        for where in (main.chats, chatsmod.chats, tools.chats):
            self.assertNotEqual(Path(where.path).resolve(),
                                Path(chatsmod.PATH).resolve())

    def test_the_memory_being_written_is_not_his(self):
        for where in (memory_module.memory, brain_module.memory):
            self.assertNotEqual(Path(where.path).resolve(),
                                Path(memory_module.PATH).resolve())

    def test_driving_handle_leaves_his_file_exactly_as_it_was(self):
        his = Path(chatsmod.PATH)
        before = his.read_bytes() if his.exists() else None
        recorded = main.chats.how_many()
        brain = SimpleNamespace(history=[], unanswered=None,
                                think=mock.Mock(return_value="Done."))
        with mock.patch.object(main, "speak", lambda text, ui=None, until=None: None):
            main.handle(brain, "write me a game", None, 0.0, None, None)
        self.assertEqual(main.chats.how_many(), recorded + 1)   # it did record
        self.assertEqual(his.read_bytes() if his.exists() else None, before)


class HowLongHeMayTalkFor(unittest.TestCase):
    """Hearing a long sentence costs almost nothing hosted and falls off a
    cliff here. Measured on his own recording, grown by repeating it: 20s
    took 0.45s hosted and 1.68s here, 28s took 0.48s and 4.71s, and 29s here
    took 38.19s - longer than he took to say it. So the limit has to know
    which ear is listening, and twenty seconds for everyone was the local
    model's limit charged to him whatever was going to hear him."""

    def _listener(self):
        ear = Listener.__new__(Listener)
        ear.heard_by = ""
        return ear

    def test_groq_can_hear_him_out(self):
        ear = self._listener()
        with mock.patch.object(config, "HEARING", "hosted"):
            self.assertEqual(ear._longest(), config.MAX_RECORD_SECONDS)

    def test_the_model_here_cannot(self):
        ear = self._listener()
        with mock.patch.object(config, "HEARING", "local"):
            self.assertEqual(ear._longest(), config.MAX_LOCAL_RECORD_SECONDS)

    def test_it_follows_whoever_actually_answered_last(self):
        """With groq unreachable every recording falls back here, and a
        forty-five second clip would then simply time out. HEARING is the
        intent; heard_by is the fact."""
        ear = self._listener()
        with mock.patch.object(config, "HEARING", "hosted"):
            ear.heard_by = "here"
            self.assertEqual(ear._longest(), config.MAX_LOCAL_RECORD_SECONDS)
            ear.heard_by = "groq"
            self.assertEqual(ear._longest(), config.MAX_RECORD_SECONDS)

    def test_each_limit_sits_where_the_measurements_put_it(self):
        """The local one under the cliff: 28s measured at 4.71s here and 29s
        at 38.19s, so anything from 29 up is slower than the person speaking.
        The hosted one above the twenty seconds he was being cut off at,
        which cost 0.45s when 45s costs 0.82s."""
        self.assertLess(config.MAX_LOCAL_RECORD_SECONDS, 29)
        self.assertGreater(config.MAX_RECORD_SECONDS, 20)

    def test_he_is_told_when_it_stopped_before_he_did(self):
        """Only the console knew, and he never sees the console. Being cut
        off mid-sentence and then answered as though that was all he said is
        the part worth knowing about."""
        lines = []
        ui = SimpleNamespace(line=lambda kind, text: lines.append((kind, text)),
                             set_state=lambda state: None)
        listener = SimpleNamespace(cut_short=45.0, transcribe_seconds=0.0,
                                   heard_by="groq")
        brain = SimpleNamespace(history=[], unanswered=None,
                                think=mock.Mock(return_value="Done."))
        with mock.patch.object(main, "speak", lambda text, ui=None, until=None: None):
            main.handle(brain, "the next step is to start a memory", ui, 0.0,
                        listener, None)
        notes = [text for kind, text in lines if kind == "note"]
        self.assertTrue(any("45 seconds" in n for n in notes), notes)

    def test_an_ordinary_command_gets_no_such_note(self):
        lines = []
        ui = SimpleNamespace(line=lambda kind, text: lines.append((kind, text)),
                             set_state=lambda state: None)
        listener = SimpleNamespace(cut_short=0.0, transcribe_seconds=0.0,
                                   heard_by="groq")
        brain = SimpleNamespace(history=[], unanswered=None,
                                think=mock.Mock(return_value="Done."))
        with mock.patch.object(main, "speak", lambda text, ui=None, until=None: None):
            main.handle(brain, "what time is it in tokyo", ui, 0.0, listener, None)
        self.assertFalse([text for kind, text in lines if kind == "note"])


class LookingBackByDay(unittest.TestCase):
    """"What did I tell you about my sister" and "what did we talk about a
    week ago" are not the same question, and only the first has a word to
    look for. Asked the second, reo called past_chats with nothing, got
    "Say what to look for." back, and handed the question to him."""

    def _chats(self, rows):
        home = Path(tempfile.mkdtemp(prefix="reo-chats-"))
        self.addCleanup(shutil.rmtree, home, True)
        store = Chats(home / "c.jsonl")
        with open(store.path, "w", encoding="utf-8") as f:
            for when, said, reply in rows:
                f.write(json.dumps({"when": when.isoformat(timespec="seconds"),
                                    "you": said, "reo": reply}) + "\n")
        return store

    def _both_days(self):
        week = datetime.datetime.now() - datetime.timedelta(days=7)
        today = datetime.datetime.now()
        return self._chats([
            (week, "lets plan the trip to goa in december", "Sounds good."),
            (week, "book the flights for the goa trip", "I cannot book flights."),
            (today, "what is the capital of peru", "Lima."),
        ]), week.date().isoformat()

    def test_asking_with_nothing_gives_the_most_recent(self):
        """"What have we been talking about" is a real question and the
        answer to it is not "say what to look for"."""
        store, _ = self._both_days()
        found = store.search()
        self.assertIn("peru", found.lower())
        self.assertNotIn("Say what to look for", found)

    def test_a_day_gives_that_day(self):
        store, week = self._both_days()
        found = store.search(day=week)
        self.assertIn("goa", found.lower())
        self.assertNotIn("peru", found.lower())

    def test_today_and_yesterday_are_days(self):
        """Someone saying "yesterday" to a voice assistant means the day."""
        store, _ = self._both_days()
        self.assertIn("peru", store.search(day="today").lower())
        self.assertIn("Nothing was said", store.search(day="yesterday"))

    def test_a_day_with_nothing_on_it_says_so_and_says_to_say_so(self):
        """Told only "nothing was said", reo searched again with no day, got
        the most recent, and answered "yesterday we covered Peru's capital"
        about a conversation from today."""
        store, week = self._both_days()
        found = store.search(day="yesterday")
        self.assertIn("Nothing was said", found)
        # The days that do have something on them, so an unscoped second
        # search tells reo nothing it was not just given. Being told "do not
        # answer with another day's conversation" did not stop it doing that.
        self.assertIn(week, found)
        self.assertIn(datetime.date.today().isoformat(), found)

    def test_a_day_it_cannot_read_asks_for_one_it_can(self):
        store, _ = self._both_days()
        self.assertIn("2026-11-10", store.search(day="a week ago"))

    def test_a_word_is_looked_for_outside_the_day_before_giving_up(self):
        """Reo volunteers the day. Asked what he had said about Goa it
        searched for Goa on today, found nothing, and said there was no note
        about Goa - when Goa was the whole of a conversation the week before.
        """
        store, week = self._both_days()
        found = store.search("goa", "today")
        self.assertIn("goa", found.lower())
        self.assertIn(week, found)
        self.assertIn("another day", found)

    def test_a_word_that_is_nowhere_at_all_still_says_so(self):
        store, _ = self._both_days()
        self.assertIn("Nothing", store.search("zebra", "today"))
        self.assertIn("Nothing", store.search("zebra"))

    def test_a_word_inside_the_day_is_not_widened(self):
        store, week = self._both_days()
        found = store.search("flights", week)
        self.assertIn("flights", found.lower())
        self.assertNotIn("another day", found)

    def test_what_comes_back_says_what_it_is(self):
        """Without it reo mixed the search up with the conversation in front
        of it: asked what was said a week ago it searched correctly, got the
        right exchanges, and answered with what had been said today - which
        was also in its history, undated, because it happened today."""
        store, week = self._both_days()
        found = store.search(day=week)
        self.assertIn("saved conversations", found.lower())
        self.assertIn(week, found)
        # and when it spans days it names them, rather than "earlier"
        spanning = store.search()
        self.assertIn(f"from {week} to {datetime.date.today().isoformat()}",
                      spanning)

    def test_nothing_said_at_all_is_not_an_empty_search_result(self):
        store = self._chats([])
        for asked in (store.search(), store.search("goa"), store.search(day="today")):
            self.assertIn("Nothing has been said", asked)

    def test_the_model_can_pass_either_or_neither(self):
        spec = next(s for s in tools.TOOLS if s["name"] == "past_chats")
        self.assertEqual(set(spec["input_schema"]["properties"]), {"about", "day"})
        self.assertFalse(spec["input_schema"].get("required"))
        self.assertEqual(tools.FUNCTIONS["past_chats"](),
                         chatsmod.chats.search())


class StoringWhatWasActuallySaid(unittest.TestCase):
    """The kept conversation was holding "[neutral] She lives in Bangalore.",
    and that tag then came back into the next session as something reo had
    said - and went out again in every request after it."""

    def test_the_emotion_tag_is_not_part_of_what_was_said(self):
        self.assertEqual(speak_module.said_aloud("[neutral] She lives in Bangalore."),
                         "She lives in Bangalore.")

    def test_nor_is_the_offer_of_further_help(self):
        self.assertEqual(speak_module.said_aloud("[happy] Hello! How can I help you today?"),
                         "Hello!")

    def test_nor_the_answer_said_twice(self):
        self.assertEqual(speak_module.said_aloud("We talked about Goa.[neutral] "
                                                 "We talked about Goa."),
                         "We talked about Goa.")

    def test_what_gets_written_down_is_what_he_heard(self):
        brain = SimpleNamespace(history=[], unanswered=None, think=mock.Mock(
            return_value="[neutral] She lives in Bangalore. Anything else?"))
        before = main.chats.how_many()
        with mock.patch.object(main, "speak", lambda text, ui=None, until=None: None):
            main.handle(brain, "where does she live", None, 0.0, None, None)
        self.assertEqual(main.chats.how_many(), before + 1)
        kept = main.chats.search("bangalore")
        self.assertIn("She lives in Bangalore.", kept)
        self.assertNotIn("[neutral]", kept)
        self.assertNotIn("Anything else", kept)

    def test_a_fast_path_answer_is_written_down_the_same_way(self):
        brain = SimpleNamespace(history=[], unanswered=None,
                                think=mock.Mock(side_effect=AssertionError(
                                    "the model should not have been asked")))
        before = main.chats.how_many()
        with mock.patch.object(main, "speak", lambda text, ui=None, until=None: None), \
             mock.patch.object(main.fastpath, "answer",
                               lambda text: "[happy] It is half past four."):
            main.handle(brain, "what time is it", None, 0.0, None, None)
        self.assertEqual(main.chats.how_many(), before + 1)
        self.assertNotIn("[happy]", main.chats.search("half past"))


class TheWayHeActuallyTalks(unittest.TestCase):
    """Run over the 164 things he has really said to reo, the fast path
    answered two of them. Every other plain instruction - "open the brave
    browser", "open Google Chrome browser", "close it for me" - went to the
    model, which is a round trip and a different answer each time for
    something the machine can simply do.

    What it was missing was not routes. It was that he puts an article in
    front, a noun on the end, and a "for me" after that."""

    HIS_APPS = ["Brave", "Google Chrome", "Chrome Remote Desktop", "Notepad",
                "ChatGPT", "Claude", "Instagram", "LinkedIn"]

    def setUp(self):
        self.opened, self.visited, self.did = [], [], []

        def open_app(name):
            exact = [a for a in self.HIS_APPS if a.lower() == name.lower()]
            hits = exact or [a for a in self.HIS_APPS if name.lower() in a.lower()]
            if not hits:
                return f"Nothing installed matches {name!r}."
            if len(hits) > 1 and not exact:
                return "Several apps match that, so nothing was opened."
            self.opened.append(hits[0])
            return f"Opened {hits[0]}."

        for where, name, thing in [
                (apps, "open_app", open_app),
                (tools, "open_website",
                 lambda url: self.visited.append(url) or f"Opened {url}."),
                (reo_system, "control",
                 lambda action, amount=0: self.did.append(action) or "Done.")]:
            patch = mock.patch.object(where, name, thing)
            patch.start()
            self.addCleanup(patch.stop)

    def test_the_manners_come_off_before_matching(self):
        self.assertEqual(fastpath.plainly("can you open notepad for me"),
                         "open notepad")
        self.assertEqual(fastpath.plainly("hey reo turn it up please"),
                         "turn it up")
        self.assertEqual(fastpath.plainly("close it for me right now"),
                         "close it")

    def test_the_article_and_the_noun_come_off_a_program(self):
        """His Start menu says "Brave" and "Google Chrome", and open_app
        matches it with *name*, so "brave browser" matched nothing at all."""
        for said in ["open the brave browser", "open the Brave browser right now",
                     "open brave browser app", "can you open the brave browser"]:
            self.opened.clear()
            self.assertIsNotNone(fastpath.answer(said), said)
            self.assertEqual(self.opened, ["Brave"], said)

    def test_google_chrome_browser_is_google_chrome(self):
        self.assertIsNotNone(fastpath.answer("Open Google Chrome browser."))
        self.assertEqual(self.opened, ["Google Chrome"])

    def test_a_thing_that_is_not_a_program_still_goes_to_the_model(self):
        """The article comes off before that check, not instead of it."""
        for said in ["open the first result", "open the next one", "open it"]:
            self.assertIsNone(fastpath.answer(said), said)
        self.assertEqual(self.opened, [])

    def test_two_programs_matching_is_not_guessed_at(self):
        """"chrome" is both Google Chrome and Chrome Remote Desktop. Picking
        one for him is worse than letting the model ask."""
        self.assertIsNone(fastpath.answer("open chrome"))
        self.assertEqual(self.opened, [])

    def test_a_route_that_declines_lets_the_next_one_try(self):
        """It used to end the search, so "open youtube" - no such program -
        never reached the route that knows youtube is a website."""
        self.assertIsNotNone(fastpath.answer("open youtube"))
        self.assertEqual(self.visited, ["youtube.com"])
        self.assertEqual(self.opened, [])

    def test_the_sites_he_says_by_name(self):
        for said, where in [("open gmail for me", "mail.google.com"),
                            ("open whatsapp", "web.whatsapp.com"),
                            ("open the youtube", "youtube.com")]:
            self.visited.clear()
            self.assertIsNotNone(fastpath.answer(said), said)
            self.assertEqual(self.visited, [where], said)

    def test_a_program_he_has_beats_the_website(self):
        """Instagram, LinkedIn and ChatGPT are installed on his laptop, so
        "open instagram" opens the one he has rather than a browser tab."""
        for said, app in [("open instagram", "Instagram"),
                          ("open chatgpt", "ChatGPT")]:
            self.opened.clear()
            self.visited.clear()
            fastpath.answer(said)
            self.assertEqual(self.opened, [app], said)
            self.assertEqual(self.visited, [], said)

    def test_the_laptop_controls_take_the_manners_too(self):
        fastpath.answer("close it for me")
        fastpath.answer("turn it up please")
        self.assertEqual(self.did, ["close_window", "volume_up"])

    def test_what_the_model_is_given_is_what_he_said(self):
        """The trim decides whether a route matches. Trimming what goes to
        the model would hand it fragments."""
        asked = []
        brain = SimpleNamespace(history=[], unanswered=None, think=mock.Mock(
            side_effect=lambda text: asked.append(text) or "Done."))
        with mock.patch.object(main, "speak", lambda text, ui=None, until=None: None):
            main.handle(brain, "Can you tell me a joke, please?", None, 0.0,
                        None, None)
        self.assertEqual(asked, ["Can you tell me a joke, please?"])


class WhenNothingWasSaidThatDay(unittest.TestCase):
    """Asked "what did we talk about yesterday" with the real model, when
    nothing had been said yesterday, it searched correctly, was told nothing
    was said, searched again with no day at all, and reported the week before
    as yesterday. Being told "nothing was said", then "do not answer with
    another day's conversation", then which days did have something on them -
    none of it stopped the second search. So that one answer is not its to
    give any more."""

    def _chats(self, when=None):
        home = Path(tempfile.mkdtemp(prefix="reo-chats-"))
        self.addCleanup(shutil.rmtree, home, True)
        store = Chats(home / "c.jsonl")
        if when is not None:
            with open(store.path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"when": when.isoformat(timespec="seconds"),
                                    "you": "what is the capital of peru",
                                    "reo": "Lima."}) + "\n")
        patch = mock.patch.object(chatsmod, "chats", store)
        patch.start()
        self.addCleanup(patch.stop)
        return store

    def test_nothing_yesterday_is_answered_without_the_model(self):
        self._chats(datetime.datetime.now())        # something today, not yesterday
        for said in ["what did we talk about yesterday",
                     "what did we discuss yesterday",
                     "what did you and i talk about yesterday"]:
            self.assertEqual(fastpath.answer(said), "Nothing was said yesterday.",
                             said)

    def test_a_day_with_something_on_it_goes_to_the_model(self):
        """Summarising a conversation is a thing only the model can do."""
        self._chats(datetime.datetime.now())
        self.assertIsNone(fastpath.answer("what did we talk about today"))

    def test_days_ago_counts_back(self):
        self._chats(datetime.datetime.now())
        self.assertEqual(fastpath.answer("what did we talk about 3 days ago"),
                         "Nothing was said 3 days ago.")

    def test_it_says_it_the_way_he_asked_it(self):
        self._chats(datetime.datetime.now() - datetime.timedelta(days=5))
        self.assertEqual(fastpath.answer("what did we talk about today"),
                         "Nothing was said today.")

    def test_a_vaguer_day_is_left_to_the_model(self):
        """"Last week" is a range, not a day, and guessing which day he
        meant would be worse than searching."""
        self._chats(datetime.datetime.now())
        for said in ["what did we talk about last week",
                     "what did we talk about a while ago",
                     "what did we talk about on monday"]:
            self.assertIsNone(fastpath.answer(said), said)

    def test_nothing_ever_said_is_still_answered(self):
        self._chats(None)
        self.assertEqual(fastpath.answer("what did we talk about yesterday"),
                         "Nothing was said yesterday.")


class NotDyingSilently(unittest.TestCase):
    """Every recent session in his log ends on "[Press Ctrl+Alt+R to start
    listening]" with nothing after it - no goodbye, no error, no traceback.
    Four things made that possible, and this is each of them."""

    def test_the_log_gets_stderr_too(self):
        """A traceback goes to stderr. Only stdout was teed, so a crash left
        the log looking like a clean startup and nothing else."""
        home = Path(tempfile.mkdtemp(prefix="reo-log-"))
        self.addCleanup(shutil.rmtree, home, True)
        was_out, was_err = sys.stdout, sys.stderr
        self.addCleanup(lambda: (setattr(sys, "stdout", was_out),
                                 setattr(sys, "stderr", was_err)))
        with mock.patch.object(main, "__file__", str(home / "reo" / "__main__.py")):
            (home / "reo").mkdir()
            main._tee_to_logfile()
            print("this went to stdout")
            print("this went to stderr", file=sys.stderr)
        sys.stdout, sys.stderr = was_out, was_err
        written = (home / "reo.log").read_text(encoding="utf-8")
        self.assertIn("this went to stdout", written)
        self.assertIn("this went to stderr", written)

    def test_a_crash_in_a_thread_is_written_down(self):
        """Without this a crash in the barge-in watcher or the microphone
        pump printed to a stderr nobody was reading and the thread died on
        the spot - which looks like reo going deaf, not like a bug."""
        was = sys.excepthook, threading.excepthook
        self.addCleanup(lambda: (setattr(sys, "excepthook", was[0]),
                                 setattr(threading, "excepthook", was[1])))
        said = io.StringIO()
        main._log_crashes()
        with contextlib.redirect_stdout(said), contextlib.redirect_stderr(said):
            thread = threading.Thread(target=lambda: 1 / 0, name="barge-in")
            thread.start()
            thread.join()
        written = said.getvalue()
        self.assertIn("[crash]", written)
        self.assertIn("barge-in", written)
        self.assertIn("ZeroDivisionError", written)

    def test_one_bad_reply_does_not_end_the_session(self):
        """handle() was called outside the loop's try/except, so anything it
        raised ended the session and took the traceback with it."""
        brain = SimpleNamespace(history=[], unanswered=None)
        listener = SimpleNamespace(
            listen=mock.Mock(side_effect=["say something", "goodbye"]),
            missed="")
        toggle = SimpleNamespace(listening=True, presses=0,
                                 wait_until_listening=lambda: None,
                                 listen_now=lambda: None)
        blew_up = iter([RuntimeError("the voice fell over"), None])

        def handle(*a, **kw):
            boom = next(blew_up)
            if boom:
                raise boom
            return False                     # second time: goodbye

        said = io.StringIO()
        with mock.patch.object(main, "handle", handle), \
             mock.patch.object(main, "speak", lambda *a, **kw: None), \
             contextlib.redirect_stdout(said):
            main._loop(listener, brain, None, toggle, "Ctrl+Alt+R", None)
        self.assertEqual(listener.listen.call_count, 2, "it gave up after the first")
        self.assertIn("fell over", said.getvalue())

    def test_losing_its_voice_is_not_fatal(self):
        """pyttsx3 talks to SAPI over COM, which fails on Windows for its own
        reasons, and this was unguarded all the way up to the main loop."""
        said = io.StringIO()
        # His .env sets REO_TTS=orpheus, so without pinning the backend this
        # went out to the network and said "Hello." out loud on his laptop.
        with mock.patch.object(config, "TTS_BACKEND", "pyttsx3"), \
             mock.patch.object(speak_module, "_speak_pyttsx3",
                               side_effect=OSError("COM said no")), \
             contextlib.redirect_stdout(said):
            cut = speak_module.speak("[neutral] Hello.")
        self.assertFalse(cut)
        self.assertIn("could not speak", said.getvalue())

    def test_the_log_says_why_it_stopped(self):
        """It used to simply stop, which told him nothing about whether reo
        had been closed or had fallen over."""
        brain = SimpleNamespace(history=[], unanswered=None)
        listener = SimpleNamespace(listen=mock.Mock(return_value="goodbye"),
                                   missed="")
        toggle = SimpleNamespace(listening=True, presses=0,
                                 wait_until_listening=lambda: None,
                                 listen_now=lambda: None)
        said = io.StringIO()
        with mock.patch.object(main, "handle", lambda *a, **kw: False), \
             mock.patch.object(main, "speak", lambda *a, **kw: None), \
             contextlib.redirect_stdout(said):
            main._loop(listener, brain, None, toggle, "Ctrl+Alt+R", None)
        self.assertIn("reo stopped: you said goodbye", said.getvalue())

    def test_a_closed_window_says_so_as_well(self):
        brain = SimpleNamespace(history=[], unanswered=None)
        said = io.StringIO()
        ui = SimpleNamespace(closed=True, set_state=lambda s: None,
                             line=lambda k, t: None)
        with mock.patch.object(main, "speak", lambda *a, **kw: None), \
             contextlib.redirect_stdout(said):
            main._loop(SimpleNamespace(), brain, None, None, "", ui)
        self.assertIn("reo stopped: the window was closed", said.getvalue())


class AskingWhereTheTimeGoes(unittest.TestCase):
    """Asked "why are you slow", reo went and asked a model - which knows
    nothing whatever about reo's timings, while reo had been counting them
    all along for the window's panel."""

    def _session(self):
        fresh = stats_module.Session()
        patch = mock.patch.object(stats_module, "session", fresh)
        patch.start()
        self.addCleanup(patch.stop)
        return fresh

    def test_before_anything_is_asked_it_says_so(self):
        self._session()
        self.assertIn("not asked me anything",
                      stats_module.session.how_long_i_take())

    def test_it_names_the_part_that_is_worst(self):
        session = self._session()
        session.add_reply(6.0, 1.0, 1.0)
        self.assertIn("listening", session.how_long_i_take())
        other = self._session()
        other.add_reply(1.0, 1.0, 6.0)
        self.assertIn("speaking", other.how_long_i_take())

    def test_the_spoken_answer_stays_short(self):
        """Spoken in full the breakdown ran to about twenty-five seconds,
        which is a strange thing to sit through from something being asked
        why it is slow. The detail is printed; the answer is said."""
        session = self._session()
        for _ in range(5):
            session.add_reply(4.5, 1.5, 3.2)
        said = io.StringIO()
        with contextlib.redirect_stdout(said):
            spoken = session.how_long_i_take()
        self.assertLess(len(spoken), 160, spoken)
        self.assertLessEqual(len(re.findall(r"[.!?]", spoken)), 2, spoken)
        # and the numbers are still written down where he can read them
        self.assertIn("heard", said.getvalue())
        self.assertIn("spoke", said.getvalue())

    def test_the_question_never_reaches_the_model(self):
        session = self._session()
        session.add_reply(4.5, 1.5, 3.2)
        for asked in ["why are you slow", "how fast are you",
                      "whats taking so long", "how long do you take",
                      "where does the time go", "how slow are you"]:
            with contextlib.redirect_stdout(io.StringIO()):
                answer = fastpath.answer(asked)
            self.assertIsNotNone(answer, asked)
            self.assertIn("seconds", answer, asked)

    def test_a_different_question_about_time_still_goes_where_it_should(self):
        self._session()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertIn("It is", fastpath.answer("what time is it"))
            self.assertIsNone(fastpath.answer("how long is a piece of string"))


class AnyoneCanInstallIt(unittest.TestCase):
    """He asked for one file anyone could install reo from. These are the
    things that would quietly break it for a stranger."""

    ROOT = Path(__file__).resolve().parent.parent

    def test_there_is_an_installer(self):
        self.assertTrue((self.ROOT / "install.ps1").exists())

    def test_the_example_env_exists_because_two_scripts_copy_it(self):
        """setup.bat copies it and set-key.ps1 falls back to it, and both do
        so silently - `copy .env.example .env >nul` - so losing this file
        would break a stranger's install without saying anything. It is a
        dotfile, which is exactly how it gets lost."""
        example = self.ROOT / ".env.example"
        self.assertTrue(example.exists())
        self.assertIn("LLM_API_KEY", example.read_text(encoding="utf-8"))

    def test_the_example_env_holds_no_real_key(self):
        text = (self.ROOT / ".env.example").read_text(encoding="utf-8")
        self.assertNotIn("gsk_", text)
        self.assertIn("your-groq-key-here", text)

    def test_nothing_asks_for_the_key_where_it_can_be_seen(self):
        """setup.bat used `set /p`, which echoes what is typed, leaves it in
        the window's history and puts it in an environment variable. His key
        is his to type, and only into a secure prompt."""
        for script in ("setup.bat", "install.ps1"):
            text = (self.ROOT / script).read_text(encoding="utf-8")
            # Comments out: setup.bat explains in a `rem` why it does not use
            # `set /p`, and that explanation is not an echoed key.
            code = "\n".join(
                line for line in text.lower().splitlines()
                if not line.strip().startswith(("rem ", "#", "::")))
            self.assertNotIn("set /p", code, script)
            self.assertIn("set-key.ps1", text, script)

    def test_the_installer_can_run_with_nobody_there(self):
        """Which is also how it gets tested."""
        text = (self.ROOT / "install.ps1").read_text(encoding="utf-8")
        self.assertIn("$Unattended", text)
        self.assertIn("function Ask", text)
        # every question goes through Ask, or -Unattended is half-honoured
        self.assertNotRegex(text, r"(?m)^\s*(?!.*function).*\(Read-Host\)\s*-")

    def test_the_installer_does_not_hide_a_failed_install(self):
        """--quiet on the requirements step would hide the one red line that
        matters."""
        text = (self.ROOT / "install.ps1").read_text(encoding="utf-8")
        requirements = [l for l in text.splitlines() if "-r requirements.txt" in l]
        self.assertTrue(requirements)
        for line in requirements:
            self.assertNotIn("--quiet", line)

    def test_the_pinned_python_matches_what_the_scripts_look_for(self):
        """requirements.txt was resolved against 3.14 while setup.bat looked
        for 3.12 and 3.11 only."""
        for script in ("setup.bat", "install.ps1"):
            text = (self.ROOT / script).read_text(encoding="utf-8")
            self.assertIn("3.14", text, script)


class TheWindow(unittest.TestCase):
    """The window is the whole visible product and had no tests at all.

    poll() is the page's only way in and it asks four times a second. If
    anything in there raises, the call fails on the JavaScript side and the
    window stops updating for good - the page cannot tell a broken bridge
    from a slow one. stats.rows() has already frozen there once, on a lock
    that was not reentrant, and every panel went empty."""

    def _bridge(self):
        from reo.webui import Bridge
        return Bridge()

    def test_what_reo_said_comes_through(self):
        bridge = self._bridge()
        bridge._pending.put(["state", "listening"])
        bridge._pending.put(["line", "reo", "Lima."])
        answer = bridge.poll()
        self.assertEqual(answer["events"],
                         [["state", "listening"], ["line", "reo", "Lima."]])

    def test_the_events_are_drained_not_repeated(self):
        bridge = self._bridge()
        bridge._pending.put(["line", "you", "hello"])
        self.assertEqual(len(bridge.poll()["events"]), 1)
        self.assertEqual(bridge.poll()["events"], [])

    def test_a_broken_panel_does_not_break_the_window(self):
        bridge = self._bridge()
        said = io.StringIO()
        with mock.patch.object(webui, "stats",
                               SimpleNamespace(rows=mock.Mock(
                                   side_effect=RuntimeError("deadlocked")))), \
             contextlib.redirect_stdout(said):
            bridge._pending.put(["line", "reo", "still here"])
            answer = bridge.poll()
        # the panel is empty for this tick, and that is all
        self.assertEqual(answer["session"], [])
        self.assertEqual(answer["events"], [["line", "reo", "still here"]])
        self.assertIn("could not read session", said.getvalue())

    def test_a_broken_device_panel_is_the_same(self):
        bridge = self._bridge()
        with mock.patch.object(webui, "_device",
                               side_effect=OSError("no psutil")), \
             contextlib.redirect_stdout(io.StringIO()):
            answer = bridge.poll()
        self.assertEqual(answer["device"], [])
        self.assertIsInstance(answer["session"], list)

    def test_the_panels_are_what_the_page_can_draw(self):
        """Pairs of strings, and a number only where there is a bar."""
        bridge = self._bridge()
        for row in bridge.poll()["session"]:
            self.assertEqual(len(row), 2, row)
            self.assertTrue(all(isinstance(cell, str) for cell in row), row)
        for row in bridge.poll()["device"]:
            self.assertIn(len(row), (2, 3), row)
            self.assertIsInstance(row[0], str)

    def test_the_page_it_loads_is_actually_there(self):
        self.assertTrue(webui.PAGE.exists(), webui.PAGE)
        page = webui.PAGE.read_text(encoding="utf-8", errors="replace")
        # the page pulls; python never pushes. If this is gone, nothing in
        # the window ever updates.
        self.assertIn("poll", page)

    def test_closing_the_window_is_noticed(self):
        with mock.patch.object(webui.webview, "create_window",
                               lambda *a, **kw: SimpleNamespace(
                                   events=SimpleNamespace(closed=_Signal()))):
            display = webui.Display(subtitle="press the key")
        self.assertFalse(display.closed)
        display._on_closed()
        self.assertTrue(display.closed)

    def test_the_voice_thread_only_ever_queues(self):
        """set_state and line are called while reo is answering. Neither may
        touch the window: WebView2 only allows calls from its own thread."""
        with mock.patch.object(webui.webview, "create_window",
                               lambda *a, **kw: SimpleNamespace(
                                   events=SimpleNamespace(closed=_Signal()))):
            display = webui.Display()
        display.set_state("thinking")
        display.line("reo", "Lima.")
        self.assertEqual(display.bridge.poll()["events"],
                         [["state", "thinking"], ["line", "reo", "Lima."]])


class _Signal:
    """Stands in for pywebview's event slot, which is used with +=."""

    def __iadd__(self, handler):
        return self


class WhatTheModelActuallySends(unittest.TestCase):
    """Every one of the 28 tools was given the arguments a misheard command
    turns into - nothing, an empty string, a null, a number where a word
    belongs. Eight of them raised. run_tool catches it, so reo does not die,
    but the exception becomes the answer: what he heard was "Tool search_web
    failed: 'NoneType' object has no attribute 'strip'"."""

    NONSENSE = [{}, {"query": None}, {"query": ""}, {"name": None},
                {"text": None}, {"text": 123}, {"url": None}, {"url": ""},
                {"key": None}, {"topic": None}, {"command": None},
                {"about": None}, {"fact": None}, {"answer": None},
                {"action": None}, {"app": None}]

    def test_no_tool_hands_back_an_exception(self):
        import webbrowser
        from reo import apps as apps_module

        home = Path(tempfile.mkdtemp(prefix="reo-fuzz-"))
        self.addCleanup(shutil.rmtree, home, True)
        with mock.patch.object(config, "WORKSPACE", str(home)), \
             mock.patch.object(webbrowser, "open", lambda *a, **kw: True), \
             mock.patch.object(apps_module, "_powershell",
                               lambda *a, **kw: "NOTFOUND"), \
             mock.patch.object(reo_system, "control", lambda a, n=0: "Done."), \
             contextlib.redirect_stdout(io.StringIO()):
            for name in sorted(tools.FUNCTIONS):
                for args in self.NONSENSE:
                    said = tools.run_tool(name, dict(args))
                    if said.startswith(f"Tool {name} failed"):
                        # a wrong keyword is the model's mistake, not a bug
                        if any(w in said for w in ("unexpected keyword",
                                                   "required positional",
                                                   "missing 1 required")):
                            continue
                        self.fail(f"{name}({args}) -> {said}")

    def test_a_null_is_read_as_nothing_said(self):
        self.assertEqual(tools.asked_for(None), "")
        self.assertEqual(tools.asked_for(None, "fallback"), "fallback")

    def test_a_number_is_read_as_the_number(self):
        """"type 123" arrives as a number."""
        self.assertEqual(tools.asked_for(123), "123")
        self.assertEqual(tools.asked_for(1.5), "1.5")

    def test_a_true_is_not_read_as_one(self):
        """bool is an int, and typing "True" is not what was meant."""
        self.assertEqual(tools.asked_for(True), "")

    def test_the_spacing_in_what_it_types_is_left_alone(self):
        """He asks reo to type code, and the indentation is the code."""
        code = "def f():\n    return 1\n"
        self.assertEqual(tools.asked_for(code), code)

    def test_each_one_says_what_is_missing(self):
        for name, args, wanted in [
                ("search_web", {"query": None}, "what to search"),
                ("look_up", {"topic": None}, "what to look up"),
                ("open_website", {"url": None}, "which website"),
                ("read_page", {"url": None}, "which page"),
                ("press_key", {"key": None}, "which key"),
                ("type_text", {"text": None}, "what to type"),
                ("run_command", {"command": None}, "which command"),
                ("open_search", {"query": None}, "what to search")]:
            said = tools.run_tool(name, args)
            self.assertIn(wanted, said.lower(), f"{name}: {said}")

    def test_an_empty_command_is_not_run(self):
        """It was. The shell printed "[shell] running:" and ran nothing."""
        ran = []
        with mock.patch.object(shell_module, "run_command",
                               lambda c, confirmed=False: ran.append(c) or "ran"):
            tools.run_tool("run_command", {"command": "   "})
        self.assertEqual(ran, [])


class SayingNamesProperly(unittest.TestCase):
    """Found by driving the real brain through twenty commands and reading
    what it said: "Opened https://youtube. com." and "Put the earlier
    print_primes. py back."

    _undouble split on every dot rather than on a full stop followed by a
    space, then rejoined the pieces with a space. Writing files and saying
    their names is one of the things reo does most, and it had been
    mispronouncing every one of them. 451 tests did not notice."""

    def test_a_filename_keeps_its_extension(self):
        for said in ["Wrote game.py to your folder.",
                     "Put the earlier print_primes.py back.",
                     "The file is at C:/Users/kollu/notes.txt.",
                     "I made notes.md and todo.txt."]:
            self.assertEqual(speak_module.said_aloud(said), said, said)

    def test_a_web_address_stays_in_one_piece(self):
        for said in ["Opened https://youtube.com.",
                     "Opened mail.google.com.",
                     "It is on en.wikipedia.org."]:
            self.assertEqual(speak_module.said_aloud(said), said, said)

    def test_a_number_with_a_decimal_point_stays_a_number(self):
        for said in ["That is 3.5 metres.", "Version 2.0.1 is out.",
                     "It took 0.27 seconds."]:
            self.assertEqual(speak_module.said_aloud(said), said, said)

    def test_an_abbreviation_is_left_alone(self):
        self.assertEqual(speak_module.said_aloud("Use e.g. a shorter name."),
                         "Use e.g. a shorter name.")

    def test_the_answer_said_twice_is_still_said_once(self):
        """Which is what _undouble is for, and what the loose split was
        buying. It needs a space after the full stop, and gpt-oss puts one."""
        self.assertEqual(speak_module.said_aloud("The capital of Peru is Lima. Lima."),
                         "The capital of Peru is Lima.")
        self.assertEqual(speak_module.said_aloud("Lima. Lima."), "Lima.")

    def test_two_real_sentences_are_both_kept(self):
        for said in ["I like Rome. Rome is warm.",
                     "Chrome is open. It took a moment."]:
            self.assertEqual(speak_module.said_aloud(said), said, said)


class TheWakeWordPath(unittest.TestCase):
    """He runs reo on the hotkey, so none of this is exercised day to day -
    which is exactly how it rots. These are smoke tests: they check the parts
    hold together, not that the models are any good."""

    def test_the_phrase_engine_names_itself_readably(self):
        from reo import phrase
        # lower case on purpose: it is shown inside quotes, as say "hey jarvis"
        self.assertEqual(phrase.pretty("hey_jarvis"), "hey jarvis")
        self.assertTrue(phrase.pretty(config.WAKE_PHRASE))

    def test_what_the_user_is_told_to_say_matches_the_engine(self):
        with mock.patch.object(config, "WAKE_ENGINE", "phrase"):
            self.assertEqual(main.woken_by(), "hey jarvis")
        with mock.patch.object(config, "WAKE_ENGINE", "whisper"):
            self.assertEqual(main.woken_by(), config.WAKE_NAME)

    def test_the_names_it_answers_to_include_the_near_misses(self):
        """A weak microphone turns "reo" into "rio" and "rayo"."""
        self.assertIn(config.WAKE_NAME, config.WAKE_WORDS)
        self.assertGreater(len(config.WAKE_WORDS), 1)

    def test_the_whisper_engine_hears_its_name_in_a_sentence(self):
        """It transcribes the room and looks for the name in what came back,
        so it has to find it and hand back what followed it."""
        from reo.wake import WakeListener
        for said, name, after in [
                ("reo what time is it", True, "what time is it"),
                ("rio open chrome", True, "open chrome"),
                ("hey reo", True, ""),
                ("the weather is nice", False, None)]:
            words = normalize(said).split()
            self.assertEqual(any(WakeListener._is_name(w) for w in words),
                             name, said)
            if name:
                self.assertEqual(
                    (WakeListener._command_after_wake(normalize(said)) or "").strip(),
                    after, said)


class NotTalkingToItself(unittest.TestCase):
    """He said "open youtube" once and then watched this, with no further
    word from him at all:

        REO  All set. Enjoy!
        YOU  Enjoy!
        REO  Thanks! You're welcome.
        YOU  You're welcome
        REO  Likewise.
        YOU  Likewise
        REO  Same here.
        YOU  Same here

    The microphone is held open and the hotkey stays on between commands, so
    the ring that feeds the next recording its pre-roll was filling up with
    reo's own voice while reo was speaking. Then it handed those words to
    itself and answered them."""

    HIS_LOOP = [("All set. Enjoy!", "Enjoy!"),
                ("Thanks! You're welcome.", "You're welcome"),
                ("Likewise.", "Likewise"),
                ("Same here.", "Same here"),
                ("Glad to be of help!", "Glad to be of help!"),
                ("That's what I'm here for.", "That's what I'm here for!"),
                ("Perfect!", "Perfect")]

    # --- the ring, which is where it starts -------------------------------

    def _microphone(self, talking):
        mic = Microphone(keep_open=True)
        self.addCleanup(mic.close)
        block = np.zeros((listen.BLOCK_SAMPLES, 1), dtype=np.float32)
        card = SimpleNamespace(read=lambda n: (block, False),
                               start=lambda: None, stop=lambda: None,
                               close=lambda: None)
        with mock.patch.object(listen.sd, "InputStream", lambda **kw: card), \
             mock.patch.object(listen, "speaking", talking):
            self.assertTrue(mic.open())
            deadline = time.time() + 2
            while len(mic._recent) < 3 and time.time() < deadline:
                time.sleep(0.01)
            kept = len(mic._recent)
            mic.close()
        return kept

    def test_the_ring_keeps_the_room(self):
        self.assertGreater(self._microphone(lambda: False), 0)

    def test_the_ring_keeps_nothing_while_reo_is_speaking(self):
        """This is the fix. Everything else here is the second line."""
        self.assertEqual(self._microphone(lambda: True), 0)

    def test_speaking_says_so_while_it_speaks(self):
        self.assertFalse(speak_module.talking(settle=0))
        said = io.StringIO()
        with mock.patch.object(config, "TTS_BACKEND", "pyttsx3"), \
             mock.patch.object(speak_module, "_speak_pyttsx3",
                               lambda *a, **kw: self.assertTrue(
                                   speak_module.talking()) or False), \
             contextlib.redirect_stdout(said):
            speak_module.speak("[neutral] Hello.")
        self.assertFalse(speak_module._talking.is_set())

    def test_it_stays_true_for_a_moment_after(self):
        """A little of the clip is still on its way out of the sound card
        when the call to play it returns."""
        speak_module._talking.clear()
        speak_module._went_quiet = time.time()
        self.assertTrue(speak_module.talking())
        self.assertFalse(speak_module.talking(settle=0))

    def test_what_it_last_said_is_remembered(self):
        said = io.StringIO()
        with mock.patch.object(config, "TTS_BACKEND", "pyttsx3"), \
             mock.patch.object(speak_module, "_speak_pyttsx3",
                               lambda *a, **kw: False), \
             contextlib.redirect_stdout(said):
            speak_module.speak("[happy] All set. Enjoy!")
        self.assertEqual(speak_module.just_said(), "All set. Enjoy!")

    # --- the second line --------------------------------------------------

    def test_every_line_of_his_loop_is_caught(self):
        for said, came_back in self.HIS_LOOP:
            speak_module._last_words = said
            speak_module._went_quiet = time.time()
            self.assertTrue(main._my_own_voice(came_back),
                            f"{said!r} -> {came_back!r}")

    def test_what_he_actually_says_gets_through(self):
        for said, he_said in [("Chrome is open.", "open youtube"),
                              ("Lima.", "what about peru"),
                              ("Should I overwrite it?", "yes"),
                              ("I wrote game.py.", "run it"),
                              ("All set. Enjoy!", "")]:
            speak_module._last_words = said
            speak_module._went_quiet = time.time()
            self.assertFalse(main._my_own_voice(he_said),
                             f"{said!r} -> {he_said!r}")

    def test_it_stops_watching_after_a_while(self):
        """Past the window it is probably really him. One turn of the loop he
        saw took about six seconds, so the window is generous."""
        speak_module._last_words = "Likewise."
        speak_module._went_quiet = time.time() - main.ECHO_WINDOW - 1
        self.assertFalse(main._my_own_voice("Likewise"))

    def test_the_loop_does_not_answer_itself(self):
        """End to end: reo speaks, the microphone hands those words straight
        back, and the model must never be asked about them."""
        answered = []
        heard = iter(["Enjoy!", "You're welcome", "goodbye"])
        listener = SimpleNamespace(listen=lambda **kw: next(heard), missed="")
        toggle = SimpleNamespace(listening=True, presses=0,
                                 wait_until_listening=lambda: None,
                                 listen_now=lambda: None)
        brain = SimpleNamespace(history=[], unanswered=None)

        def handle(brain, text, *a, **kw):
            answered.append(text)
            return text != "goodbye"

        speak_module._last_words = "All set. Enjoy! You're welcome."
        speak_module._went_quiet = time.time()
        said = io.StringIO()
        with mock.patch.object(main, "handle", handle), \
             mock.patch.object(main, "speak", lambda *a, **kw: None), \
             contextlib.redirect_stdout(said):
            main._loop(listener, brain, None, toggle, "Ctrl+Alt+R", None)
        self.assertEqual(answered, ["goodbye"])
        self.assertIn("my own voice", said.getvalue())


class WhisperSayingGoodbyeForHim(unittest.TestCase):
    """He did not say bye.

        You: Search MBD in the YouTube search bar.
        Reo: Done. Results are showing.
        You: Bye
        Reo: Goodbye.

    "Bye" is one of the handful of things whisper emits when handed a quiet
    room, and it is also an exit phrase, so an invented word ended a session
    he was in the middle of using."""

    def _ask(self, text, level, voice):
        listener = SimpleNamespace(last_level=level, last_voice=voice,
                                   transcribe_seconds=0.0, heard_by="groq",
                                   cut_short=0.0)
        brain = SimpleNamespace(history=[], unanswered=None,
                                think=mock.Mock(return_value="x"))
        with mock.patch.object(main, "speak", lambda *a, **kw: None),              contextlib.redirect_stdout(io.StringIO()):
            return main.handle(brain, text, None, 0.0, listener, None)

    def test_an_invented_goodbye_does_not_end_the_session(self):
        for text in ["Bye", "goodbye", "quit", "exit"]:
            self.assertTrue(self._ask(text, 0.002, 0.1), text)

    def test_him_actually_saying_it_still_works(self):
        for text in ["Bye", "goodbye", "goodbye reo", "stop listening"]:
            self.assertFalse(self._ask(text, 0.08, 0.5), text)

    def test_two_words_is_nobody_s_hallucination(self):
        """"stop listening" is a real instruction, quietly said."""
        self.assertFalse(self._ask("stop listening", 0.002, 0.1))

    def test_whisper_s_stock_answers_are_known(self):
        for stock in ["Bye", "bye.", "Thank you.", "you",
                      "Thanks for watching!", "Okay", "Music"]:
            self.assertTrue(au_mod.from_silence(stock), stock)

    def test_what_he_really_says_is_not_mistaken_for_one(self):
        for real in ["bye reo", "thank you for that", "open youtube",
                     "okay open chrome", "you are my baby"]:
            self.assertFalse(au_mod.from_silence(real), real)

    def test_a_silent_clip_never_reaches_whisper_now(self):
        """The floor exists for exactly this and was being skipped whenever
        the voice detector was working - but silero says where speech starts
        and stops, not whether there was ever anything there."""
        # No voice detector is set up here on purpose: the floor no longer
        # asks whether there is one.
        ear = Listener.__new__(Listener)
        ear.missed = ""
        quiet = np.full(16000, 0.001, dtype=np.float32)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(ear._transcribe(quiet), "")
        self.assertEqual(ear.missed, "quiet")


if __name__ == "__main__":
    unittest.main()
