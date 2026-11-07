"""Offline tests: no microphone, no API key, no network."""
import re
import unittest

import numpy as np

from reo import config, tools
from reo.audio import looks_hallucinated
from reo.audio import normalize as normalize_gain
from reo.__main__ import EXIT_PHRASES
from reo.text import is_speech, normalize
from reo.brain import SYSTEM, Brain
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
            def __enter__(self): return self
            def __exit__(self, *a): return False
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


if __name__ == "__main__":
    unittest.main()
