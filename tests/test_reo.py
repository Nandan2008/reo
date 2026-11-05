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


class ToolFailures(unittest.TestCase):
    """run_tool must hand failures back to the model, never raise."""

    def test_unknown_app_is_refused(self):
        self.assertIn("Not allowed", tools.run_tool("open_app", {"name": "spotify"}))

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
