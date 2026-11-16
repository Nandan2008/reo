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
        # One model for the whole class: loading it is the slow part. But
        # Listener() also opens the real microphone and holds it, which these
        # tests never want and never closed - and an open PortAudio stream at
        # interpreter shutdown can wedge the process. One run sat there for
        # fifty eight minutes.
        with mock.patch.object(config, "KEEP_MIC_OPEN", False):
            cls.listener = Listener()
        cls.addClassCleanup(cls.listener.mic.close)

    def setUp(self):
        # The local path, so these stay about endpointing and are the same
        # every run. One test below does the hosted one on purpose.
        local = mock.patch.object(config, "HEARING", "local")
        local.start()
        self.addCleanup(local.stop)
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

    def test_groq_hears_him_too(self):
        """The path he actually runs. Skipped rather than failed when the
        network or the key is not there - that is a bad day, not a bug."""
        with mock.patch.object(config, "HEARING", "hosted"):
            self.listener._ears_client = None
            said = self.hear([self.clip])
        if not said.strip():
            self.skipTest("groq did not answer; the local path covers the rest")
        self.assertIn("hyderabad", said.lower(), f"heard {said!r}")

    def heard_for(self):
        """How long reo kept listening.

        Not last_voice, which counts only the blocks the detector called
        speech: 22 seconds of his two sentences joined comes out as 11.8
        seconds of voice, because of the gaps between the words. And not the
        card's position either - the microphone is held open and its reader
        thread keeps pulling tape into the ring after reo has stopped taking
        it, so that reads as the whole tape however early reo stopped.
        """
        return self.listener.last_seconds

    def talking_for(self, seconds):
        """A tape of him talking without stopping for that long.

        His two sentences joined, repeated. Measured with silero: the longest
        unvoiced run inside that is 0.70s, and 0.9s is what ends a recording,
        so as far as reo is concerned this is one person not drawing breath.
        """
        cycle = np.concatenate([self.part(HIS_VOICE), self.part(HIS_HELLO)])
        return np.tile(cycle, int(16000 * seconds / len(cycle)) + 1)[
            :int(16000 * seconds)]

    def test_he_is_not_cut_off_where_he_used_to_be(self):
        """Twice in one session reo stopped recording while he was still
        talking and answered the first half of what he had said. Twenty
        seconds was the local model's limit - past 28s it takes longer to
        recognise a clip than he took to say it - and it was being applied
        to him even when groq was going to do the hearing.

        This runs the local path, which setUp pins, so what it holds is the
        24 second limit. The hosted 45 is checked in test_reo, where the
        measurements it comes from are written down.
        """
        self.hear([self.talking_for(22.0), quiet(2.0)])
        self.assertEqual(self.listener.cut_short, 0.0,
                         "cut off at 22 seconds of continuous speech")
        self.assertGreater(self.heard_for(), 20.0,
                           f"stopped listening after {self.heard_for():.1f}s")
        # What comes back as words is not asserted. Twenty-two seconds of his
        # two sentences on a loop is repetitive enough for the hallucination
        # guard to throw away, which is that guard doing its job on a tape no
        # person would produce. Whether reo was still listening is the point.
        self.assertEqual(self.listener.cut_short, 0.0)

    def test_there_is_still_a_limit_and_he_is_told_about_it(self):
        """Eight seconds here, so the test is quick; the real one is 24 on
        this laptop and 45 when groq is listening."""
        with mock.patch.object(config, "MAX_LOCAL_RECORD_SECONDS", 8.0):
            self.hear([self.talking_for(20.0)])
        self.assertEqual(self.listener.cut_short, 8.0)
        # Within a block of the limit, and nowhere near the 20s of tape.
        self.assertAlmostEqual(self.heard_for(), 8.0, delta=0.5)

        # and it does not stay set, or every command after a long one would
        # be reported as cut off too
        self.hear([self.clip])
        self.assertEqual(self.listener.cut_short, 0.0)

    def test_an_ordinary_command_is_not_marked_as_cut_off(self):
        self.hear([self.clip])
        self.assertEqual(self.listener.cut_short, 0.0)


    def test_a_level_is_reported_either_way(self):
        """So "can you hear me" has something true to say."""
        self.hear([self.clip])
        self.assertGreater(self.listener.last_level, 0.01)
        self.assertGreater(self.listener.last_voice, 0.5)


if __name__ == "__main__":
    unittest.main()
