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
import re
import threading
from datetime import date, datetime, timedelta
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / "chats.jsonl"
# How many exchanges to carry into a new session. Each is two messages and
# every one of them is resent on every request, against an 8000 token minute.
CARRY = 3
# Past this, the trail is trimmed from the front.
KEEP = 4000
FOUND = 6           # exchanges handed back from a search at most


A_DATE = re.compile(r"^(\d{4}-\d{2}-\d{2})$")


def _a_day(about: str):
    """The day this is asking for, as 2026-11-10, or None if it is words.

    "yesterday" could in principle be a word to look for, but someone saying
    it to a voice assistant means the day, every time.
    """
    said = about.strip().lower()
    found = A_DATE.match(said)
    if found:
        return found.group(1)
    if said in ("today", "earlier today"):
        return date.today().isoformat()
    if said in ("yesterday", "last night"):
        return (date.today() - timedelta(days=1)).isoformat()
    return None


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

    def search(self, about: str = "", day: str = "") -> str:
        """Earlier exchanges. For the model to call.

        Two ways in, because "what did I tell you about my sister" and "what
        did we talk about a week ago" are not the same question and only the
        first has a word to look for. Asked the second, this took a word or
        nothing, so reo called it with nothing, got "Say what to look for."
        back, and handed the question to him - which is reo asking him what
        he wanted to know.

        `day` is the model's own idea. Given only words to pass, it invented
        the parameter anyway - past_chats({'about': '', 'day': '2026-11-10'})
        - and when that was ignored it fell back to asking for everything and
        answered "what did we talk about yesterday" with a conversation from
        the week before. It had the right shape; it just had nowhere to put it.

        Words narrow a day when both are given. Neither gives the most
        recent, which is what "what have we been talking about" means.
        """
        rows = self._rows()
        if not rows:
            return "Nothing has been said between us yet."
        about = " ".join(str(about or "").split())
        day = " ".join(str(day or "").split())

        # A day passed as the words is still a day: no reason to refuse it.
        wanted = _a_day(day) or _a_day(about)
        if wanted is None and day:
            return ('Give the day as 2026-11-10, or as "today" or '
                    '"yesterday".')
        everything = rows
        if wanted is not None:
            rows = [r for r in rows if (r.get("when") or "")[:10] == wanted]
            if not rows and not about:
                # Told only "nothing was said", reo searched again with no
                # day at all, got the most recent, and answered "yesterday
                # we covered Peru's capital" about a conversation from today.
                # Adding "do not answer with another day's" did not stop it.
                #
                # So the days that do have something on them come back here
                # instead. An unscoped second search then tells reo nothing
                # it was not just given, and the true answer - that nothing
                # was said, and when things were - is all there is to say.
                days = sorted({(r.get("when") or "")[:10] for r in everything
                               if r.get("when")})
                listed = ", ".join(days[-8:]) or "none"
                return (f"Nothing was said on {wanted}. The days anything was "
                        f"said on: {listed}.")
            if _a_day(about) is not None:
                about = ""              # the day was all it passed

        words = [w for w in about.lower().split() if len(w) > 2]
        if not words:
            return self._as_lines(rows[-FOUND:])

        hits = self._matching(rows, words)
        # A day narrowing a word search is not something he asked for unless
        # he named a day, and reo volunteers it: asked what he had said about
        # Goa it searched for Goa on today, found nothing, and said there was
        # no note about Goa - when Goa was the whole of a conversation the
        # week before. So a word that finds nothing inside a day is looked
        # for outside it before giving up.
        wider = False
        if not hits and wanted is not None:
            hits = self._matching(everything, words)
            wider = bool(hits)
        if not hits:
            where = f" on {wanted}" if wanted else ""
            return f"Nothing in our earlier conversations{where} about {about}."
        found = self._as_lines(hits)
        if wider:
            found = f"Nothing about {about} on {wanted}, but from another day:\n{found}"
        return found

    @staticmethod
    def _matching(rows: list[dict], words: list[str]) -> list[dict]:
        hits = []
        for row in reversed(rows):
            haystack = (row.get("you", "") + " " + row.get("reo", "")).lower()
            if all(word in haystack for word in words):
                hits.append(row)
            if len(hits) >= FOUND:
                break
        return list(reversed(hits))

    @staticmethod
    def _as_lines(rows: list[dict]) -> str:
        """The exchanges found, with a line saying that is what they are.

        Without it reo mixed them up with the conversation in front of it.
        Asked what was said a week ago it searched correctly, got the right
        exchanges back, and then answered with what had been said today -
        which was also in its history, undated, because it happened today.
        """
        lines = []
        for row in rows:
            when = (row.get("when") or "")[:16].replace("T", " ")
            lines.append(f"{when} - you: {row['you']}\n  reo: {row.get('reo', '')}")
        days = sorted({(row.get("when") or "")[:10] for row in rows})
        if len(days) == 1:
            when = f"on {days[0]}"
        elif days:
            # Named, not "earlier". Asked what was said yesterday, when
            # nothing was, reo answered from a mixed result and dated it
            # yesterday; it cannot do that against a stated span.
            when = f"from {days[0]} to {days[-1]}"
        else:
            when = "earlier"
        return (f"Found in your saved conversations {when}. Answer from these "
                f"and not from anything else:\n" + "\n".join(lines))

    def how_many(self) -> int:
        return len(self._rows())


chats = Chats()
