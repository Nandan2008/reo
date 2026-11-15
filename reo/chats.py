"""Every exchange, kept, so reo does not start each session a stranger.

Until now closing reo threw the conversation away. Ask it something on
Tuesday and follow it up on Wednesday and it had no idea what "it" was - and
"what did we talk about yesterday" was simply unanswerable.

Two different things are needed and they want different storage.

Carrying on: the last few exchanges go back into the brain's history at
startup, so a follow-up after a restart still has something to follow. That
costs tokens on every request, so it is a handful of turns, not a diary.

Looking back: everything ever said is on disk and can be searched, which
costs nothing until reo actually goes looking. That is what answers "what
did I tell you about my sister" a week later.

One line of JSON per exchange, appended. Not a single JSON array: reo is
killed by closing its window, and a half-written array is an unreadable
file where a half-written last line is one lost exchange.
"""
import json
import threading
from datetime import datetime
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / "chats.jsonl"
# How many exchanges to carry into a new session. Each is two messages and
# every one of them is resent on every request, against an 8000 token minute.
CARRY = 3
# Past this, the trail is trimmed from the front.
KEEP = 4000
FOUND = 6           # exchanges handed back from a search at most


class Chats:
    def __init__(self, path=PATH):
        self.path = Path(path)
        self._lock = threading.Lock()

    def add(self, said: str, answered: str) -> None:
        """Record one exchange. Never raises: losing a line of history is not
        worth taking the assistant down for."""
        if not (said or "").strip():
            return
        row = {"when": datetime.now().isoformat(timespec="seconds"),
               "you": said.strip(), "reo": (answered or "").strip()}
        try:
            with self._lock:
                with open(self.path, "a", encoding="utf-8") as file:
                    file.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError as e:
            print(f"  [chats] could not save: {e}")

    def _rows(self) -> list[dict]:
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except (OSError, ValueError):
            return []
        rows = []
        for line in lines[-KEEP:]:
            try:
                row = json.loads(line)
            except ValueError:
                continue        # a half-written last line, from being killed
            if isinstance(row, dict) and row.get("you"):
                rows.append(row)
        return rows

    def carry_over(self, how_many: int = CARRY) -> list[dict]:
        """The last few exchanges, as messages to start the history with."""
        rows = self._rows()[-how_many:]
        if not rows:
            return []
        messages = []
        today = datetime.now().date().isoformat()
        for row in rows:
            said = row["you"]
            # Say when, if it was not today. Without it "open it again" means
            # something reo thinks happened a minute ago.
            when = (row.get("when") or "")[:10]
            if when and when != today:
                said = f"[on {when}] {said}"
            messages.append({"role": "user", "content": said})
            messages.append({"role": "assistant", "content": row.get("reo") or ""})
        return messages

    def search(self, about: str) -> str:
        """Earlier exchanges that mention this. For the model to call."""
        words = [w for w in " ".join(str(about).lower().split()).split() if len(w) > 2]
        if not words:
            return "Say what to look for."
        hits = []
        for row in reversed(self._rows()):
            haystack = (row.get("you", "") + " " + row.get("reo", "")).lower()
            if all(word in haystack for word in words):
                hits.append(row)
            if len(hits) >= FOUND:
                break
        if not hits:
            return f"Nothing in our earlier conversations about {about}."
        lines = []
        for row in reversed(hits):
            when = (row.get("when") or "")[:16].replace("T", " ")
            lines.append(f"{when} - you: {row['you']}\n  reo: {row.get('reo', '')}")
        return "\n".join(lines)

    def how_many(self) -> int:
        return len(self._rows())


chats = Chats()
