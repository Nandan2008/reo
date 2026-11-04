"""Offline tests: no microphone, no API key, no network."""
import re
import unittest

from reo import tools
from reo.__main__ import EXIT_PHRASES, is_speech, normalize
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

    def test_no_tag(self):
        self.assertEqual(split_emotion("Just words."), ("neutral", "Just words."))

    def test_prompt_and_voice_table_agree(self):
        """Every tag the brain is told to use needs a voice setting, or it goes flat."""
        promised = set(re.findall(r"\[(\w+)]", SYSTEM))
        self.assertTrue(promised, "no tags found in the system prompt")
        self.assertEqual(promised - set(VOICE_SETTINGS), set())


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
