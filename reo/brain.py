import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from openai import (APIConnectionError, APIStatusError, APITimeoutError,
                    BadRequestError, OpenAI, RateLimitError)

from . import config, providers
from .memory import memory
from .stats import session
from .tools import READ_ONLY, TOOLS, run_tool

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
    "Never claim you did something unless a tool reported doing it. If a tool "
    "could not, say so plainly. "
    "Where a choice does not matter, pick one instead of asking. "
    "You cannot see the screen: to play a video use play_youtube, and never "
    "guess coordinates to click. "
    "Search when an answer could have changed, never putting a remembered year in "
    "the query, and read_page a result when the snippets fall short. Use look_up "
    "for facts that do not change. "
    "Call remember when the user tells you something about themselves worth "
    "knowing next time; do not announce that you did."
)

# Every tool reo can actually run. A name outside this set must never enter
# the history: the server checks past tool_calls against the tools sent with
# each new request.
KNOWN_TOOLS = {t["name"] for t in TOOLS}

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
        """One API call."""
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
            # A history that already holds a call to a tool we do not offer.
            # Answering without tools does NOT help: the call is still in the
            # messages, and the server rejects that with "Tool choice is none,
            # but model called a tool". Take the call out and ask again.
            if "was not in request.tools" in str(e) and self._forget_invented_calls():
                print("  [brain] dropped a call to a tool reo does not have; asking again")
                return self._ask(provider, client, with_tools)
            raise
        usage = getattr(response, "usage", None)
        session.add_request(getattr(usage, "total_tokens", 0) if usage else 0)
        providers.remember(provider)
        return response

    def _forget_invented_calls(self) -> bool:
        """Strip history turns that call a tool reo does not have, along with
        the results paired with them. Returns whether anything was removed, so
        the caller knows whether asking again could possibly help."""
        doomed = set()
        for message in self.history:
            calls = message.get("tool_calls") or []
            if any(c["function"]["name"] not in KNOWN_TOOLS for c in calls):
                doomed.add(id(message))
                answered = {c["id"] for c in calls}
                doomed.update(id(m) for m in self.history
                              if m.get("tool_call_id") in answered)
        if not doomed:
            return False
        self.history = [m for m in self.history if id(m) not in doomed]
        return True

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

                # A call to a tool reo does not have must never reach the
                # history. Groq validates past tool_calls against the tools
                # sent with every later request, so one invented name turns
                # every following question into a 400 - and the retry without
                # tools is rejected too, because the call is still in there.
                # That is what "Something went wrong reaching my brain" was.
                runnable = [tc for tc in msg.tool_calls if tc.function.name in KNOWN_TOOLS]
                invented = sorted({tc.function.name for tc in msg.tool_calls
                                   if tc.function.name not in KNOWN_TOOLS})
                if invented:
                    print(f"  [brain] no such tool: {', '.join(invented)}; not recording it")
                if not runnable:
                    spoken = (msg.content or "").strip()
                    if spoken:
                        self.history.append({"role": "assistant", "content": spoken})
                        return spoken
                    self.history.append({
                        "role": "user",
                        "content": "That tool does not exist. Use one you were given, "
                                   "or just answer."})
                    continue

                self.history.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for tc in runnable
                    ],
                })
                session.add_tools(len(runnable))
                wanted = []
                for tc in runnable:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    print(f"  [tool] {tc.function.name}({args})")
                    wanted.append((tc, args))

                # Searches take a second or two each and the model often asks
                # for several at once. Looking things up can happen at the same
                # time; pressing keys and moving windows cannot, because the
                # order is the point. So: all-or-nothing, and only when every
                # call in the batch is a lookup.
                together = len(wanted) > 1 and all(
                    tc.function.name in READ_ONLY for tc, _ in wanted)
                if together:
                    with ThreadPoolExecutor(max_workers=len(wanted)) as pool:
                        results = list(pool.map(
                            lambda job: run_tool(job[0].function.name, job[1]), wanted))
                else:
                    results = [run_tool(tc.function.name, args) for tc, args in wanted]

                for (tc, _), result in zip(wanted, results):
                    self.history.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
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
