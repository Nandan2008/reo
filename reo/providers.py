"""Which model reo talks to, and where to go when one says no.

Groq's free tier allows 8000 tokens a minute, which one tool-heavy question can
spend by itself - reo spent a whole evening apologising out loud instead of
answering. A second key is the cheap fix: when a provider refuses, ask the next
one rather than giving up, and come back to the first a minute later when its
per-minute window has rolled over.

Every provider here speaks the OpenAI wire format, so only the key, the URL and
the model name change.
"""
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from . import config

# A refusal is almost always a per-minute window, so a minute is long enough to
# wait before trying that key again. Not a ban, just a bench.
COOLDOWN_SECONDS = 60.0

# OpenAI-compatible services reo knows the address of, so .env only needs a key.
# name: (env var, base url, default model, env var for the model)
KNOWN = {
    "llama": ("LLAMA_API_KEY", "https://api.llama.com/compat/v1",
              "Llama-4-Maverick-17B-128E-Instruct-FP8", "LLAMA_MODEL"),
    "groq": ("GROQ_API_KEY", "https://api.groq.com/openai/v1",
             "openai/gpt-oss-120b", "GROQ_MODEL"),
    "openrouter": ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1",
                   "meta-llama/llama-3.3-70b-instruct:free", "OPENROUTER_MODEL"),
    "gemini": ("GEMINI_API_KEY",
               "https://generativelanguage.googleapis.com/v1beta/openai/",
               "gemini-2.0-flash", "GEMINI_MODEL"),
}


# Ollama runs on this machine and has no key at all, so it is found by
# asking rather than by reading one out of .env - a laptop that has it today
# may not tomorrow. Nothing is sent anywhere and nothing is metered.
OLLAMA = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
# Asking costs a local round trip, so the answer is kept for a while. Long
# enough that it is not asked per reply, short enough that starting ollama
# after reo does not mean restarting reo.
LOCAL_RECHECK = 120.0
_local_seen: tuple[float, object] = (0.0, False)


def _ollama(path: str, body=None, timeout: float = 1.5):
    """Ask the local ollama something, or None if it is not there."""
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        OLLAMA + path, data=data,
        headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as answer:
            return json.loads(answer.read() or b"{}")
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
        return None


def local_model(env=None) -> str | None:
    """A model on this machine that can call tools, or None.

    The capability matters more than the name. reo is nothing but tool calls,
    and a model without them answers every single question with a sentence
    about how it cannot do that - his llama3 refuses outright, with
    "llama3:latest does not support tools".
    """
    env = os.environ if env is None else env
    wanted = (env.get("REO_OLLAMA_MODEL") or "").strip()
    listed = _ollama("/api/tags")
    if not listed:
        return None
    names = [m.get("name") for m in listed.get("models", []) if m.get("name")]
    if wanted:
        names = [n for n in names if n == wanted or n.split(":")[0] == wanted.split(":")[0]]
    for name in names:
        shown = _ollama("/api/show", {"model": name}, timeout=4.0)
        if shown and "tools" in (shown.get("capabilities") or []):
            return name
    return None


def local(env=None) -> Provider | None:
    """Ollama as a provider, if it is running here with a usable model."""
    global _local_seen
    when, found = _local_seen
    if time.time() - when > LOCAL_RECHECK:
        name = local_model(env)
        found = Provider("ollama", "ollama-needs-no-key",
                         OLLAMA + "/v1", name) if name else None
        _local_seen = (time.time(), found)
    return found or None


@dataclass(frozen=True)
class Provider:
    name: str
    key: str
    base_url: str
    model: str

    @property
    def label(self) -> str:
        """What the window shows: the model, without the vendor prefix."""
        return self.model.split("/")[-1]


def _host_name(url: str) -> str:
    """A short name for a provider we only know by its URL."""
    host = url.split("//")[-1].split("/")[0]
    parts = [p for p in host.split(".") if p not in ("api", "www", "com", "ai", "io", "net")]
    return parts[0] if parts else host


def build(env=None) -> list[Provider]:
    """The providers to try, in order.

    LLM_API_KEY stays first because it is the one deliberately configured.
    LLM_API_KEYS adds more keys for that same service, and any KNOWN provider
    with a key in the environment is appended as a further fallback.
    """
    env = os.environ if env is None else env
    found: list[Provider] = []
    seen = set()

    def add(name, key, base_url, model):
        key = (key or "").strip()
        if not key or (base_url, key) in seen:
            return
        seen.add((base_url, key))
        found.append(Provider(name, key, base_url, model))

    primary_url = env.get("LLM_BASE_URL") or config.LLM_BASE_URL
    primary_model = env.get("LLM_MODEL") or config.LLM_MODEL
    primary_name = _host_name(primary_url)
    add(primary_name, env.get("LLM_API_KEY"), primary_url, primary_model)
    # Spare keys for the same service. Numbered so the log says which one spoke.
    for i, extra in enumerate(env.get("LLM_API_KEYS", "").split(","), start=2):
        add(f"{primary_name}-{i}", extra, primary_url, primary_model)

    for name, (var, base_url, model, model_var) in KNOWN.items():
        add(name, env.get(var), base_url, env.get(model_var) or model)

    # The machine's own model goes last: it answers when everything else has
    # hit its limit, which turns "I have used up my allowance" into an answer.
    # REO_PREFER_LOCAL=1 puts it first instead - slower and less sure with
    # twenty six tools, but nothing leaves the laptop and nothing is counted.
    here = local(env)
    if here is not None:
        if env.get("REO_PREFER_LOCAL", "0") not in ("0", "false", "no"):
            found.insert(0, here)
        else:
            found.append(here)

    return found


class Pool:
    """The keys reo may use, and which of them is currently sulking."""

    def __init__(self, providers=None):
        self.providers = list(build() if providers is None else providers)
        self._until: dict[int, float] = {}

    def __len__(self) -> int:
        return len(self.providers)

    def current(self) -> tuple[int, Provider] | None:
        """The first provider not in cooldown.

        If every one is benched the cooldown was only ever a guess, so offer the
        one due back soonest rather than refusing to answer at all.
        """
        if not self.providers:
            return None
        now = time.time()
        for i, provider in enumerate(self.providers):
            if self._until.get(i, 0.0) <= now:
                return i, provider
        i = min(range(len(self.providers)), key=lambda j: self._until.get(j, 0.0))
        return i, self.providers[i]

    def bench(self, index: int, seconds: float = COOLDOWN_SECONDS) -> None:
        self._until[index] = time.time() + seconds

    def benched(self, index: int) -> bool:
        return self._until.get(index, 0.0) > time.time()

    def free(self) -> int:
        """How many providers are usable right now."""
        return sum(1 for i in range(len(self.providers)) if not self.benched(i))


# Models only one service hosts, whatever else is configured.
EXCLUSIVE = {"canopylabs/orpheus": "api.groq.com",
             "whisper-large": "api.groq.com"}


def hosting(model: str, env=None) -> Provider | None:
    """The key that can serve this particular model.

    Orpheus, the expressive voice, lives at Groq alone, so it still needs the
    Groq key after reo's thinking has moved to another provider.
    """
    found = [p for p in build(env) if p.name != "ollama"]   # ollama has no voice
    host = next((h for prefix, h in EXCLUSIVE.items() if model.startswith(prefix)), None)
    if host is None:
        return found[0] if found else None
    return next((p for p in found if host in p.base_url), None)


_pool: Pool | None = None


def pool() -> Pool:
    """One pool for the whole process, so the brain and the window agree about
    which keys are spent."""
    global _pool
    if _pool is None:
        _pool = Pool()
    return _pool


# The provider that last answered, so the window's MODEL row names what is
# actually speaking instead of whatever .env happened to say.
_active: Provider | None = None


def remember(provider: Provider) -> None:
    global _active
    _active = provider


def active_label() -> str:
    if _active is not None:
        return _active.label
    return config.LLM_MODEL.split("/")[-1]
