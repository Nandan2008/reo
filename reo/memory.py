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
import threading
from datetime import date
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / "memory.json"
KEEP = 60           # facts stored on disk
CARRY = 24          # facts put in front of the model each request
LONGEST = 200       # one spoken sentence of fact, not an essay


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
        fact = " ".join(str(fact).split())[:LONGEST]
        if not fact:
            return "Nothing to remember."
        with self._lock:
            match = self._find(fact)
            if match is not None:
                return f"Already knew that: {match['text']}"
            self.facts.append({"text": fact, "since": date.today().isoformat()})
            del self.facts[:-KEEP]
            self._save()
        return f"Remembered: {fact}"

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
