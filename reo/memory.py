"""What reo keeps between runs.

Until now reo forgot everything the moment it closed: your name, how you like
to be answered, which of the three Hyderabad addresses is yours. The model has
plenty of general knowledge and no knowledge of you at all, and no amount of
web searching fixes that.

Facts live in a small JSON file beside the log and ride along in the system
prompt, so reo simply knows them rather than having to go and ask. That costs
tokens on every request, which is why the number carried is capped.
"""
import json
import re
import threading
from datetime import date
from pathlib import Path

from . import config

PATH = Path(__file__).resolve().parent.parent / "memory.json"
KEEP = 60           # facts stored on disk
CARRY = 24          # facts put in front of the model each request
LONGEST = 200       # one spoken sentence of fact, not an essay


# He said: "from now on if anyone asks who are you you should tell you are
# Nandan's baby". That is not a fact about him, it is an instruction about
# reo, and reo has to be able to find it again when asked - which means it
# has to be stored in a shape about.py can recognise.
#
# Asking the model to store it in that shape was tried first and does not
# work. Told to use "Reo says: " it called remember("If anyone asks who are
# you, say you are Nandan's baby") instead, and the instruction was lost
# among the facts. It is reliable about calling remember, so the shaping is
# done here, where it either happens or does not.
AN_INSTRUCTION = re.compile(
    r"\b(?:say|says|saying|tell|answer|answering|respond|reply)\b[^,]{0,28}?"
    r"\b(?:you|i)(?:\s+are|\s+am|\s*['\u2019]?re|\s*['\u2019]?m"
    r"|\s+should\s+be)\s+(.+)$", re.IGNORECASE)
# ... and that it is reo being described rather than him. Both persons: the
# model says "who you are" one run and "who I am" the next.
ABOUT_ITSELF = re.compile(
    r"\b(?:who|what)\s+(?:you\s+are|are\s+you|i\s+am|am\s+i)\b",
    re.IGNORECASE)


def _as_standing(fact: str) -> str:
    """"Reo says: I am X" if this is him saying how reo should answer."""
    about_itself = ABOUT_ITSELF.search(fact)
    found = AN_INSTRUCTION.search(fact)
    if not (about_itself and found):
        return fact
    answer = found.group(1).strip().strip('"\'').rstrip(".")
    if not answer:
        return fact
    return f"{config.WAKE_NAME.capitalize()} says: I am {answer}"


class Memory:
    def __init__(self, path=PATH):
        self.path = Path(path)
        self._lock = threading.Lock()
        self.facts = self._load()

    def _load(self) -> list[dict]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []       # no file yet, or someone edited it into nonsense
        if not isinstance(data, list):
            return []
        return [f for f in data if isinstance(f, dict) and f.get("text")][-KEEP:]

    def _save(self) -> None:
        try:
            self.path.write_text(json.dumps(self.facts, indent=1), encoding="utf-8")
        except OSError as e:
            print(f"  [memory] could not save: {e}")

    # --- what the model calls ---------------------------------------------

    def remember(self, fact: str) -> str:
        fact = _as_standing(" ".join(str(fact).split()))[:LONGEST]
        if not fact:
            return "Nothing to remember."
        with self._lock:
            # Superseding is checked first. _find treats one fact containing
            # another as the same fact, so "User's name is Nandan Sai" was
            # being swallowed by "User's name is Nandan" already being there
            # and the correction was silently dropped.
            older = self._supersedes(fact)
            if older is not None and older["text"].lower() == fact.lower():
                return f"Already knew that: {older['text']}"
            match = None if older is not None else self._find(fact)
            if match is not None:
                return f"Already knew that: {match['text']}"
            if older is not None:
                was = older["text"]
                older["text"] = fact
                older["since"] = date.today().isoformat()
                self._save()
                return f"Changed that: {was} -> {fact}"
            self.facts.append({"text": fact, "since": date.today().isoformat()})
            del self.facts[:-KEEP]
            self._save()
        return f"Remembered: {fact}"

    def _supersedes(self, fact: str):
        """An existing fact this one replaces: same opening, different end.

        He ended up with "User's name is Nandan", "User's name is Byron" and
        "User's name is Nanan Sai" all in front of the model at once, because
        each was a different enough string to count as new. Reo then had
        three names for him and picked whichever it liked. A fact that opens
        the same way as one already held is a correction of it, not another
        one to keep beside it.
        """
        opening = " ".join(fact.lower().split()[:3])
        if len(opening.split()) < 3:
            return None
        for existing in self.facts:
            if " ".join(existing["text"].lower().split()[:3]) == opening:
                return existing
        return None

    def forget(self, about: str) -> str:
        about = " ".join(str(about).split()).lower()
        if not about:
            return "Nothing to forget."
        with self._lock:
            going = [f for f in self.facts if about in f["text"].lower()]
            if not going:
                return f"Nothing remembered about {about}."
            self.facts = [f for f in self.facts if f not in going]
            self._save()
        return "Forgot: " + "; ".join(f["text"] for f in going)

    def recall(self, about: str = "") -> str:
        about = " ".join(str(about).split()).lower()
        with self._lock:
            found = [f["text"] for f in self.facts if not about or about in f["text"].lower()]
        if not found:
            return f"Nothing remembered about {about}." if about else "Nothing remembered yet."
        return "; ".join(found)

    # --- what the brain puts in the prompt ---------------------------------

    def about_reo(self) -> list[str]:
        """Facts he has stated about reo rather than about himself.

        He said, in his own words, "from now on if anyone asks who are you
        you should tell you are Nandan's baby" - and reo could not obey him,
        because the identity questions are answered from about.py without
        the model or the memory being consulted at all. He then asked "whose
        baby are you" eight times and got eight different answers.

        Stored as "Reo says: ..." and already in the first person, so there
        is no grammar to do here. Turning "Reo is Nandan's baby" round in
        code produced "I is Nandan's baby", and the list of verbs that would
        need is endless; the model can simply write the sentence.

        One at a time, on purpose: _supersedes sees the same three opening
        words every time, so a new instruction replaces the last rather than
        joining it. He should not have to unpick a pile of them by voice.
        """
        mark = f"{config.WAKE_NAME.lower()} says:"
        with self._lock:
            return [f["text"][len(mark):].strip() for f in self.facts
                    if f["text"].lower().startswith(mark)]

    def for_prompt(self) -> str:
        """The facts, short enough to resend on every request."""
        with self._lock:
            recent = self.facts[-CARRY:]
        if not recent:
            return ""
        return " Remembered about this user: " + "; ".join(f["text"] for f in recent) + "."

    def _find(self, fact: str):
        """Caller holds the lock. Same fact said twice, in either direction."""
        lowered = fact.lower()
        for existing in self.facts:
            text = existing["text"].lower()
            if text == lowered or text in lowered or lowered in text:
                return existing
        return None


memory = Memory()
