"""Telling his voice from everyone else's.

He asked for this out loud, and reo had to say no: "Remember my voice from
now on, you should only respond to me." It is also the fix for the thing
that has gone wrong twice - reo answering something he did not say. Once it
was reo's own voice coming back off the speaker, and once it was whisper
inventing "Bye" out of a quiet room.

How it works: a 192-number voiceprint from the speech itself, compared
against the one he enrolled. Same voice gives a high number, anyone else
gives a low one.

Measured here before any of it was built, on Windows' two voices saying five
different sentences each:

    same voice, different words   0.684 to 0.855
    different voices              0.214 to 0.283

That is a gap of 0.40, at clips of 1.8 to 3.7 seconds - which is the length
of the things he actually says. The model is speechbrain's ECAPA-TDNN,
converted to ONNX so it runs on onnxruntime alone: no torch, 84MB, Apache-2.
It is downloaded on first use, like the speech model.

The one thing this must never do is make reo deaf to him. So:

  - It does nothing at all until he enrols, and "forget my voice" undoes it.
  - A clip too short to judge is accepted, not refused. The embedding needs
    real speech in it; his own diagnose.wav has a 1.7 second stretch that is
    mostly silence, and it scores 0.086 against his own voice.
  - When it does turn something down it says so out loud, so reo is never
    quietly ignoring him.
  - Enrolment checks itself. If his own takes do not agree with each other,
    it refuses to switch on and says why, rather than locking him out.
"""
import json
import threading
from datetime import date
from pathlib import Path

import numpy as np

from . import config

PATH = Path(__file__).resolve().parent.parent / "voiceprint.json"
MODEL = ("kenrunio/transcriber-speaker-onnx", "ecapa-any-length.onnx")

# Above this, two clips are the same person. Measured separation was 0.28
# against 0.68 on clean audio, so this sits well clear of both - deliberately
# nearer the stranger's end, because turning him away is the worse mistake.
ALIKE = config.VOICE_MATCH
# Seconds of actual speech needed before the question is worth asking.
ENOUGH = 1.2
# Takes to enrol from, and how well they must agree before this is trusted.
TAKES = 3
AGREEMENT = 0.55


class Voiceprint:
    def __init__(self, path=PATH):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._session = None
        self._me = self._load()

    # --- the model --------------------------------------------------------

    def _model(self):
        """Loaded on first use, and never again. None if it cannot be had -
        no network on a fresh machine is not a reason to stop listening."""
        if self._session is None:
            try:
                import onnxruntime as ort
                from huggingface_hub import hf_hub_download

                options = ort.SessionOptions()
                options.inter_op_num_threads = 1
                options.intra_op_num_threads = 1
                self._session = ort.InferenceSession(
                    hf_hub_download(*MODEL), sess_options=options,
                    providers=["CPUExecutionProvider"])
            except Exception as e:
                print(f"  [voice] cannot check voices ({type(e).__name__}: {e})")
                self._session = False
        return self._session or None

    def _print_of(self, audio):
        """The voiceprint of one clip, or None if it could not be taken."""
        session = self._model()
        if session is None:
            return None
        sound = np.asarray(audio, dtype=np.float32).reshape(-1)
        if not len(sound):
            return None
        try:
            out = session.run(None, {"wav": sound.reshape(1, -1),
                                     "lens": np.array([1.0], np.float32)})[0]
        except Exception as e:
            print(f"  [voice] could not read that voice ({type(e).__name__}: {e})")
            return None
        vector = np.asarray(out, dtype=np.float32).reshape(-1)
        size = float(np.linalg.norm(vector))
        return vector / size if size else None

    # --- what is stored ---------------------------------------------------

    def _load(self):
        try:
            kept = json.loads(self.path.read_text(encoding="utf-8"))
            vector = np.asarray(kept["print"], dtype=np.float32)
        except (OSError, ValueError, KeyError, TypeError):
            return None
        size = float(np.linalg.norm(vector))
        return vector / size if size else None

    @property
    def enrolled(self) -> bool:
        return self._me is not None

    def forget(self) -> str:
        with self._lock:
            self._me = None
            try:
                self.path.unlink()
            except OSError:
                pass
        return "Forgotten. I will answer anyone again."

    # --- learning it ------------------------------------------------------

    def learn(self, takes) -> tuple[bool, str]:
        """Store his voice from several recordings. Says whether it worked.

        The takes are checked against each other first. If they disagree,
        nothing is stored: a voiceprint that cannot recognise its own owner
        would simply stop reo answering him.
        """
        prints = [p for p in (self._print_of(take) for take in takes)
                  if p is not None]
        if len(prints) < 2:
            return False, ("I could not get a clear enough recording of your "
                           "voice. Try again somewhere quieter.")
        agreement = min(float(np.dot(a, b))
                        for i, a in enumerate(prints) for b in prints[i + 1:])
        if agreement < AGREEMENT:
            return False, (f"Those recordings did not sound alike enough to "
                           f"be sure - {agreement:.2f} where I want "
                           f"{AGREEMENT:.2f}. I have left this off rather "
                           f"than risk ignoring you.")
        mine = np.mean(prints, axis=0)
        mine = mine / float(np.linalg.norm(mine))
        with self._lock:
            self._me = mine
            try:
                self.path.write_text(json.dumps({
                    "print": [round(float(x), 6) for x in mine],
                    "takes": len(prints),
                    "agreement": round(agreement, 3),
                    "since": date.today().isoformat(),
                }), encoding="utf-8")
            except OSError as e:
                return False, f"I could not save that ({e})."
        return True, ("Got your voice. I will only answer you now. Say forget "
                      "my voice to undo it.")

    # --- using it ---------------------------------------------------------

    def is_him(self, audio, voiced_seconds: float):
        """True, False, or None for "not enough to go on".

        None is the important one. A clip without much speech in it gives an
        embedding that means nothing - his own diagnose.wav has a stretch
        that scores 0.086 against the rest of his own voice - and guessing
        from that would make reo deaf at random.
        """
        if self._me is None or voiced_seconds < ENOUGH:
            return None
        theirs = self._print_of(audio)
        if theirs is None:
            return None
        self.last_score = float(np.dot(self._me, theirs))
        return self.last_score >= ALIKE

    last_score = 0.0


voiceprint = Voiceprint()
