"""End to end, with his actual voice, through the actual Listener.

Everything in test_reo.py that touches listening fakes the hard part: the
detector is a script, or _transcribe is a lambda. That is how a bug got in
twice. Anything voice-like in the moment before the key press started the
silence countdown, so he had 1.5 seconds to begin talking instead of six,
and every unit test still passed because none of them had a voice in them.

These do. diagnose.wav is six seconds of him saying "I am from Hyderabad.
Hello. Hello." - saved by his own run of diagnose.py - and it goes in
through a stand-in sound card, through the real voice detector and the real
speech model, and what comes out the other end is checked.

It is not in the repository (it is his voice), so these skip without it.

    python -m unittest tests.test_voice -v
"""
import threading
import time
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np
import sounddevice as sd

from reo import config, listen
from reo.listen import Listener, Microphone

CLIP = Path(__file__).resolve().parent.parent / "diagnose.wav"
SIZE = listen.BLOCK_SAMPLES
# Where the speech is in that recording, from running silero over it:
# 0.86-2.22s is "I am from Hyderabad", then a 2.2s gap, then "Hello. Hello."
HIS_ROOM = (2.4, 4.3)       # the gap: his actual room, not digital silence
HIS_VOICE = (0.8, 2.3)      # the first sentence, "I am from Hyderabad"
HIS_HELLO = (4.3, 6.0)      # the second, "Hello. Hello." - different words,
                            # which is the whole point: a test whose pre-roll
                            # holds the words it then asserts on passes
                            # whatever the code does.


def his_voice():
    if not CLIP.exists():
        return None, 16000
    with wave.open(str(CLIP), "rb") as w:
        rate, channels = w.getframerate(), w.getnchannels()
        raw = w.readframes(w.getnframes())
    clip = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        clip = clip.reshape(-1, channels).mean(axis=1)
    return (clip if rate == 16000 else None), rate


class Timeline:
    """A stand-in sound card that plays a list of pieces, then silence.

    Paced, but ten times faster than life. It cannot simply hand blocks over
    as fast as they are asked for: the reader runs in its own thread and
    would race to the end of the tape before the listener had looked at the
    first block, which makes "he pressed the key again here" meaningless.
    Everything reo decides is counted in blocks, so ten times speed changes
    nothing except how long the test takes.
    """

    PACE = listen.BLOCK_SECONDS / 10

    def __init__(self, pieces):
        self.audio = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
        self.at = 0
        self.served = 0

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        pass

    def read(self, frames):
        time.sleep(self.PACE)
        self.served += 1
        if self.served > 4000:                  # never wedge a broken test
            raise sd.PortAudioError("test ran out of tape", -9983)
        block = np.zeros(SIZE, dtype=np.float32)
        if self.at < len(self.audio):
            piece = self.audio[self.at:self.at + SIZE]
            block[:len(piece)] = piece
            self.at += SIZE
        return block.reshape(-1, 1), False


def quiet(seconds):
    return np.zeros(int(16000 * seconds), dtype=np.float32)


class HearingHim(unittest.TestCase):
    """The real model, the real detector, his real voice."""

    @classmethod
    def setUpClass(cls):
        cls.clip, _ = his_voice()
        if cls.clip is None:
            raise unittest.SkipTest("no diagnose.wav to test against")
        # One model for the whole class: loading it is the slow part.
        cls.listener = Listener()

    def setUp(self):
        self.listener.language, self.listener.detected = "en", True
        self.listener.missed = ""
        self.listener.voice.reset()

    def part(self, span):
        start, end = span
        return self.clip[int(16000 * start):int(16000 * end)]

    def hear(self, after_press, before_press=None, press_again_after=None):
        """Run one recording through, and give back what reo made of it.

        before_press is what the room was doing while reo sat with the
        microphone open and nobody had pressed anything. It has to go through
        the real ring, which is the only way the pre-roll gets tested at all:
        opening the device and starting the recording in the same breath
        leaves that ring empty, and then a test of the pre-roll tests nothing.

        press_again_after is in seconds of the tape: how far into what he was
        saying he reached over and pressed the key a second time.
        """
        before_press = before_press or []
        head = np.concatenate(before_press) if before_press else np.zeros(0, dtype=np.float32)
        card = Timeline(list(before_press) + list(after_press))
        stop = {"now": False}

        mic = Microphone(keep_open=True)
        self.listener._mic = mic
        self.addCleanup(mic.close)
        with mock.patch.object(sd, "InputStream", lambda **kw: card):
            self.assertTrue(mic.open(), "the stand-in card would not open")
            # Let the room play into the ring, the way it does while reo is
            # waiting to be asked for something.
            deadline = time.time() + 10
            while card.at < len(head) and time.time() < deadline:
                time.sleep(0.002)

            if press_again_after is not None:
                mark = len(head) + 16000 * press_again_after

                def press():
                    while card.at < mark:
                        time.sleep(0.001)
                    stop["now"] = True

                threading.Thread(target=press, daemon=True).start()

            return self.listener.listen(stop_now=lambda: stop["now"])

    # --- the things that should work -------------------------------------

    def test_he_speaks_straight_away(self):
        said = self.hear([self.clip])
        self.assertIn("hyderabad", said.lower(), f"heard {said!r}")

    def test_he_takes_three_seconds_to_begin(self):
        """The bug. Reo must still be listening when he starts."""
        said = self.hear([quiet(3.0), self.clip])
        self.assertIn("hyderabad", said.lower(), f"heard {said!r}")

    def test_something_was_said_before_he_pressed_the_key(self):
        """The bug exactly. A voice in the moment before the press used to
        start the silence countdown, so reo gave up before he had opened his
        mouth. Here he presses, thinks for two and a half seconds, and talks."""
        said = self.hear(before_press=[self.part(HIS_HELLO)],
                         after_press=[quiet(2.5), self.part(HIS_VOICE)])
        # With the bug, reo stops 1.5s after the pre-roll and all it has is
        # "Hello" - the thing he actually said never gets recorded at all.
        self.assertIn("hyderabad", said.lower(), f"heard {said!r}")

    def test_his_own_room_before_he_presses_is_not_a_voice(self):
        said = self.hear(before_press=[self.part(HIS_ROOM)],
                         after_press=[quiet(2.0), self.clip])
        self.assertIn("hyderabad", said.lower(), f"heard {said!r}")

    def test_the_first_word_is_not_clipped(self):
        """What the pre-roll is for: he is already talking as he reaches for
        the key, so the start of it lands in the ring and not on the floor."""
        said = self.hear(before_press=[self.clip[:int(16000 * 1.6)]],
                         after_press=[self.clip[int(16000 * 1.6):]])
        self.assertIn("from hyderabad", said.lower(), f"heard {said!r}")

    def test_pressing_the_key_again_ends_it_and_keeps_the_words(self):
        said = self.hear([self.clip], press_again_after=2.4)
        self.assertTrue(said.strip(), "pressing to stop threw the words away")
        self.assertIn("hyderabad", said.lower(), f"heard {said!r}")

    def test_the_pause_in_the_middle_does_not_split_the_sentence(self):
        """There is a 2.2s gap in that recording. With a 1.5s safety net it
        ends there - which is right, they are two sentences - but the first
        one has to come back whole."""
        said = self.hear([self.clip])
        self.assertIn("hyderabad", said.lower(), f"heard {said!r}")

    # --- and the things that should not ----------------------------------

    def test_silence_is_not_turned_into_words(self):
        said = self.hear([quiet(8.0)])
        self.assertEqual(said, "", f"invented {said!r} out of nothing")
        self.assertIn(self.listener.missed, ("nothing", "silent", "noise"))

    def test_his_empty_room_is_not_turned_into_words(self):
        """Whisper does not return nothing for near-silence, it invents. Given
        his room it once produced "I've been on 30 YouTube videos again and
        again and again"."""
        room = np.concatenate([self.part(HIS_ROOM)] * 4)
        said = self.hear([room])
        self.assertEqual(said, "", f"invented {said!r} out of room noise")

    def test_he_presses_the_key_and_says_nothing(self):
        said = self.hear([quiet(7.0)])
        self.assertEqual(said, "")
        self.assertEqual(self.listener.missed, "nothing")

    def test_a_level_is_reported_either_way(self):
        """So "can you hear me" has something true to say."""
        self.hear([self.clip])
        self.assertGreater(self.listener.last_level, 0.01)
        self.assertGreater(self.listener.last_voice, 0.5)


if __name__ == "__main__":
    unittest.main()
