"""What this session has cost, in requests, tokens and seconds.

Borrowed from OpenJarvis, which puts a SYSTEM panel beside the conversation
showing requests, tokens and what the work would have cost elsewhere. Useful
here for a different reason: the timings make slowness visible while it happens
instead of after the fact, in a log.
"""
import threading

# Groq's free tier bills nothing, but the tokens are real and worth seeing.
FREE = True


class Session:
    def __init__(self):
        self._lock = threading.RLock()
        self.requests = 0
        self.tokens = 0
        self.tool_calls = 0
        self.heard = 0.0
        self.thought = 0.0
        self.spoke = 0.0
        self.replies = 0
        self.skipped_model = 0      # answered by the fast path

    def add_request(self, tokens: int = 0) -> None:
        with self._lock:
            self.requests += 1
            self.tokens += tokens or 0

    def add_tools(self, count: int) -> None:
        with self._lock:
            self.tool_calls += count

    def add_reply(self, heard: float, thought: float, spoke: float,
                  used_model: bool = True) -> None:
        with self._lock:
            self.replies += 1
            self.heard += heard
            self.thought += thought
            self.spoke += spoke
            if not used_model:
                self.skipped_model += 1

    def _average(self) -> float:
        """Caller already holds the lock. Taking it again deadlocks: a plain
        Lock is not reentrant, and rows() froze here on its first call, which
        left every panel in the window empty."""
        if not self.replies:
            return 0.0
        return (self.heard + self.thought + self.spoke) / self.replies

    @property
    def average(self) -> float:
        with self._lock:
            return self._average()

    def how_long_i_take(self) -> str:
        """Where this session's time went, out loud.

        Asked "why are you slow", reo used to go and ask a model, which knows
        nothing about it. It is the one thing in the world reo has first-hand
        knowledge of, and it was already counting it for the window's panel.

        The voice note is the actionable part: the hosted voice measured at
        +1.15s a reply against the offline one on the same five sentences,
        because Orpheus speaks more slowly, not because of the network. That
        is the one lever here he can actually pull.
        """
        from . import config

        with self._lock:
            replies = self.replies
            heard, thought, spoke = self.heard, self.thought, self.spoke
            free = self.skipped_model
        if not replies:
            return "You have not asked me anything yet, so I have nothing to compare."

        parts = {"listening": heard / replies, "thinking": thought / replies,
                 "speaking": spoke / replies}
        worst = max(parts, key=parts.get)
        # The breakdown is printed rather than read out. Spoken in full it
        # ran to about twenty-five seconds, which is a strange thing to sit
        # through from something being asked why it is slow.
        print(f"  [time] over {replies} repl{'y' if replies == 1 else 'ies'}: "
              f"heard {parts['listening']:.1f}s, thought {parts['thinking']:.1f}s, "
              f"spoke {parts['speaking']:.1f}s, "
              f"total {(heard + thought + spoke) / replies:.1f}s"
              + (f", {free} without the model" if free else ""))
        if config.TTS_BACKEND == "orpheus":
            print("  [time] the hosted voice costs about 1.15s of that per "
                  "reply - measured against the offline one on the same five "
                  "sentences. Set REO_TTS=pyttsx3 in .env for the quick one.")

        said = (f"About {(heard + thought + spoke) / replies:.0f} seconds a "
                f"reply, and most of that is {worst}.")
        if worst == "speaking" and config.TTS_BACKEND == "orpheus":
            said += " My voice is the hosted one; the offline one is quicker."
        elif worst == "listening":
            said += " That is mostly you talking, which I cannot speed up."
        return said

    def rows(self):
        """(label, value) pairs for the SYSTEM panel."""
        with self._lock:
            replies = self.replies or 1
            return [
                ("REQUESTS", str(self.requests)),
                ("TOKENS", f"{self.tokens:,}" if self.tokens < 10000
                           else f"{self.tokens / 1000:.1f}K"),
                ("TOOL CALLS", str(self.tool_calls)),
                ("NO MODEL", str(self.skipped_model)),
                ("", ""),
                ("HEARD", f"{self.heard / replies:.1f}s"),
                ("THOUGHT", f"{self.thought / replies:.1f}s"),
                ("SPOKE", f"{self.spoke / replies:.1f}s"),
                ("PER REPLY", f"{self._average():.1f}s"),
            ]


session = Session()
