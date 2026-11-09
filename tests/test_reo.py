"""Offline tests: no microphone, no API key, no network."""
import re
import shutil
import tempfile
import unittest
import urllib.parse
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from openai import APIStatusError, RateLimitError

from reo import config, fastpath, providers, tools
from reo import memory as memory_module
from reo import pointer
from reo.memory import Memory
from reo.audio import looks_hallucinated
from reo.audio import normalize as normalize_gain
from reo.__main__ import EXIT_PHRASES
from reo.text import is_speech, normalize
from reo.brain import KNOWN_TOOLS, SYSTEM, Brain
from reo.listen import Listener
from reo.speak import VOICE_SETTINGS, split_emotion


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
    def _play_with(stream):
        from unittest import mock
        from reo import speak
        audio = np.zeros(1600, dtype=np.int16)        # 0.1s at 16kHz
        with mock.patch.object(speak.sd, "play"), \
             mock.patch.object(speak.sd, "stop") as stop, \
             mock.patch.object(speak.sd, "get_stream", return_value=stream), \
             mock.patch.object(speak, "PLAY_GRACE", 0.3):
            speak._play(audio, 16000)
        return stop

    def test_returns_when_the_clip_finishes(self):
        from unittest import mock
        stop = self._play_with(mock.Mock(active=False))
        stop.assert_not_called()

    def test_gives_up_when_the_device_never_frees(self):
        from unittest import mock
        stop = self._play_with(mock.Mock(active=True))
        stop.assert_called_once()


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
        found = providers.build(self.ENV)
        self.assertEqual(found[-1].name, "llama")
        self.assertIn("api.llama.com", found[-1].base_url)

    def test_blank_keys_are_not_providers(self):
        names = [p.name for p in providers.build(self.ENV)]
        self.assertNotIn("openrouter", names)

    def test_the_same_key_is_not_listed_twice(self):
        env = dict(self.ENV, GROQ_API_KEY="groq-one")   # same key, known slot
        keys = [p.key for p in providers.build(env)]
        self.assertEqual(len(keys), len(set(keys)))

    def test_no_keys_means_no_providers(self):
        self.assertEqual(providers.build({"LLM_API_KEY": ""}), [])

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

        class Stream:
            def __init__(self): self.left = 4
            def start(self): pass
            def stop(self): raise sd.PortAudioError("Stream is stopped", -9983)
            def close(self): raise sd.PortAudioError("Stream is stopped", -9983)
            def read(self, n):
                if self.left <= 0:
                    raise sd.PortAudioError("Stream is stopped", -9983)
                self.left -= 1
                return np.full((160, 1), 0.3, dtype=np.float32), False

        listener = self._listener()
        with mock.patch.object(sd, "InputStream", lambda **kw: Stream()):
            audio = listener._record()      # must not raise
        self.assertEqual(len(audio), 160 * 4)

    def test_giving_up_on_silence_still_closes_the_stream(self):
        import sounddevice as sd
        closed = []

        class Stream:
            def start(self): pass
            def stop(self): closed.append("stop")
            def close(self): closed.append("close")
            def read(self, n):
                return np.zeros((160, 1), dtype=np.float32), False

        listener = self._listener()
        with mock.patch.object(sd, "InputStream", lambda **kw: Stream()),              mock.patch.object(config, "WAIT_FOR_SPEECH_SECONDS", 0.2):
            self.assertIsNone(listener._record())
        self.assertEqual(closed, ["stop", "close"])

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


if __name__ == "__main__":
    unittest.main()
