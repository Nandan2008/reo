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
        self._lock = threading.Lock()
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

    @property
    def average(self) -> float:
        with self._lock:
            if not self.replies:
                return 0.0
            return (self.heard + self.thought + self.spoke) / self.replies

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
                ("PER REPLY", f"{self.average:.1f}s"),
            ]


session = Session()
