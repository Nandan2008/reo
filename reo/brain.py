import json
from datetime import datetime

from openai import (APIConnectionError, APIStatusError, APITimeoutError,
                    BadRequestError, OpenAI, RateLimitError)

from . import config, providers
from .memory import memory
from .stats import session
from .tools import TOOLS, run_tool

SYSTEM = (
    # Every word here is resent on every request, and the free tier allows
    # 8000 tokens a minute. This used to be 520 tokens of prose.
    "You are Reo, a voice assistant on the user's Windows laptop. You are Reo and "
    "nothing else: never name a company or a model. Warm and a little wry; react "
    "to news like a person would. "
    "Spoken aloud, so: one short sentence, two only if the second carries real "
    "information. No markdown, lists or emoji. No filler, no offers of further "
    "help, no restating the question. "
    "Reply in the language you were spoken to in. "
    "Begin every reply with one tag: [neutral] [happy] [excited] [amused] [sad] "
    "[concerned] [apologetic]. Neutral only when nothing else fits. "
    "Call tools without asking permission first; the result says if something "
    "needs confirming. Trust what a tool told you even if the user pushes back. "
    "Search when an answer could have changed, never putting a remembered year in "
    "the query, and read_page a result when the snippets fall short. Use look_up "
    "for facts that do not change. "
    "Call remember when the user tells you something about themselves worth "
    "knowing next time; do not announce that you did."
)

# Convert our tool specs to the OpenAI-compatible format
OPENAI_TOOLS = [
    {"type": "function",
     "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
    for t in TOOLS
]


class Brain:
    def __init__(self, pool=None):
        self.pool = providers.pool() if pool is None else pool
        self._clients = {}
        self.history = []

    def _client(self, index: int, provider):
        """One client per key, made on first use and kept.

        Without a timeout a stalled call blocks the whole assistant: reo
        transcribed a question once and then simply never spoke again.
        """
        if index not in self._clients:
            self._clients[index] = OpenAI(api_key=provider.key, base_url=provider.base_url,
                                          timeout=config.REQUEST_TIMEOUT, max_retries=1)
        return self._clients[index]

    def _ask(self, provider, client, with_tools: bool):
        """One API call. gpt-oss sometimes reaches for a built-in tool we do not
        expose (notably 'refuse' when declining), which the server rejects with a
        400. Answering without tools lets it say its piece instead of crashing."""
        today = datetime.now().strftime("%A, %d %B %Y")
        # Facts about this user travel with every request, so reo knows them
        # rather than having to go and ask. Capped in memory.py: this is resent
        # each time and the free tier counts every token.
        known = memory.for_prompt()
        kwargs = dict(
            model=provider.model,
            messages=[{"role": "system",
                       "content": f"{SYSTEM} Today is {today}.{known}"}] + self.history,
            max_tokens=500,
            temperature=0.3,
        )
        if with_tools:
            kwargs.update(tools=OPENAI_TOOLS, tool_choice="auto")
        # gpt-oss reasons before it answers, and reo answers in one sentence.
        # Measured over three questions: 104 output tokens on the default
        # against 37 on "low", for the same answers. Same wall time, but a
        # third of the budget against an 8000 token minute.
        if "gpt-oss" in provider.model and config.REASONING_EFFORT:
            kwargs["reasoning_effort"] = config.REASONING_EFFORT
        try:
            response = client.chat.completions.create(**kwargs)
        except BadRequestError as e:
            if with_tools and "was not in request.tools" in str(e):
                print("  [brain] model reached for an unavailable tool; answering without tools")
                return self._ask(provider, client, with_tools=False)
            raise
        usage = getattr(response, "usage", None)
        session.add_request(getattr(usage, "total_tokens", 0) if usage else 0)
        providers.remember(provider)
        return response

    def _complete(self, with_tools: bool):
        """Ask whoever will answer.

        A per-minute token limit is not a failure worth speaking out loud when
        another key is sitting idle, so a refusal benches that key and the same
        question goes straight to the next one.
        """
        if not len(self.pool):
            raise RuntimeError("No model key configured: add one to .env.")
        failures = []
        for _ in range(len(self.pool)):
            index, provider = self.pool.current()
            try:
                return self._ask(provider, self._client(index, provider), with_tools)
            except (RateLimitError, APIConnectionError, APITimeoutError) as e:
                self.pool.bench(index)
                failures.append(f"{provider.name}: {type(e).__name__}")
            except APIStatusError as e:
                # 4xx other than 429 is our own mistake and will fail the same
                # way on every key; only pass server trouble along.
                if e.status_code < 500:
                    raise
                self.pool.bench(index, seconds=15)
                failures.append(f"{provider.name}: {e.status_code}")
            spare = self.pool.current()
            if spare is not None and not self.pool.benched(spare[0]):
                print(f"  [brain] {provider.name} unavailable; trying {spare[1].name}")
        raise RuntimeError("Every model key refused: " + ", ".join(failures))

    def think(self, text: str) -> str:
        self.history.append({"role": "user", "content": text})
        try:
            for _ in range(10):  # max tool steps per command
                resp = self._complete(with_tools=True)
                msg = resp.choices[0].message

                if not msg.tool_calls:
                    reply = (msg.content or "").strip()
                    if not reply and resp.choices[0].finish_reason == "length":
                        # reasoning models can spend max_tokens before saying anything
                        reply = "I ran out of room before I could answer. Ask me again more simply."
                    self.history.append({"role": "assistant", "content": reply})
                    return reply

                self.history.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for tc in msg.tool_calls
                    ],
                })
                session.add_tools(len(msg.tool_calls))
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    print(f"  [tool] {tc.function.name}({args})")
                    self.history.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": run_tool(tc.function.name, args),
                    })
            return "Sorry, that took too many steps."
        except Exception:
            # Drop the unfinished turn so a bad response doesn't poison later requests
            while self.history and self.history[-1]["role"] != "user":
                self.history.pop()
            if self.history:
                self.history.pop()
            raise
        finally:
            self._trim_history()

    def _trim_history(self):
        """Keep recent turns, cutting only at a user message so tool calls stay paired."""
        if len(self.history) <= config.MAX_HISTORY_MESSAGES:
            return
        start = len(self.history) - config.MAX_HISTORY_MESSAGES
        for i in range(start, len(self.history)):
            if self.history[i]["role"] == "user":
                self.history = self.history[i:]
                return
        self.history = []
