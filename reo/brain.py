import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from openai import (APIConnectionError, APIStatusError, APITimeoutError,
                    BadRequestError, OpenAI, RateLimitError)

from . import config, providers
from .memory import memory
from .reply import tighten
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
    "knowing next time; do not announce that you did. "
    "You can write files. Asked for code, a script, a list or anything else "
    "worth keeping, put the whole of it in write_file and then say one short "
    "sentence about what you wrote. Never read code or a long text aloud: "
    "write it down and say where it is. Use read_file before changing "
    "something you wrote earlier, and open_file when he wants to see it."
)

# Every tool reo can actually run. A name outside this set must never enter
# the history: the server checks past tool_calls against the tools sent with
# each new request.
KNOWN_TOOLS = {t["name"] for t in TOOLS}

# Rounds of tool calls one command may take. The last one is spent answering,
# with the tools left out of the request.
TOOL_STEPS = 10

# Convert our tool specs to the OpenAI-compatible format
OPENAI_TOOLS = [
    {"type": "function",
     "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
    for t in TOOLS
]


# A tool argument longer than this is not kept in the history.
#
# This is the one that actually bit him. Asked to write a game, reo puts the
# whole program in write_file's arguments - two thousand tokens - and that
# argument then sits in the history and is resent on every request for the
# next four turns, against a cap of eight thousand tokens a minute. Twice in
# a row, all reo could say was that it had used up its allowance, and the
# second time was only expensive because of the first.
#
# The result paired with the call already says what happened ("Wrote
# game.py"), and reo has read_file for when it needs the contents back - its
# prompt already tells it to use that before changing something it wrote. So
# the long argument is replaced by a note of its length, once it has been run.
LONGEST_ARGUMENT = 400


def _slim(arguments: str) -> str:
    """Drop the body of a long tool argument, keeping its shape.

    Left alone if it is not JSON: a malformed argument is the model's to see.
    """
    try:
        parsed = json.loads(arguments or "{}")
    except (ValueError, TypeError):
        return arguments
    if not isinstance(parsed, dict):
        return arguments
    slimmed, cut = {}, False
    for name, value in parsed.items():
        if isinstance(value, str) and len(value) > LONGEST_ARGUMENT:
            slimmed[name] = (value[:80].rstrip()
                             + f"... [{len(value)} characters, sent in full]")
            cut = True
        else:
            slimmed[name] = value
    return json.dumps(slimmed, ensure_ascii=False) if cut else arguments


# Groq says how long the wait is - "Please try again in 7m32.86s" - and reo
# was throwing that away and saying only "give it a moment". A moment is not
# seven minutes, and he has no way to tell the difference except by asking
# again and being refused again.
WAIT_IN = re.compile(r"try again in ((?:\d+h)?(?:\d+m)?[\d.]+m?s)", re.I)


def _how_long(message: str) -> str:
    """"another forty seconds" from whatever the 429 said, or "" if it said
    nothing useful. Rounded and spoken, not "32.86s"."""
    found = WAIT_IN.search(message or "")
    if not found:
        return ""
    text = found.group(1)
    seconds = 0.0
    for amount, unit in re.findall(r"([\d.]+)(h|ms|m|s)", text):
        try:
            value = float(amount)
        except ValueError:
            continue
        seconds += value * {"h": 3600, "m": 60, "s": 1, "ms": 0.001}[unit]
    if seconds <= 0:
        return ""
    if seconds < 90:
        return f"another {max(round(seconds), 1)} seconds"
    if seconds < 5400:
        minutes = round(seconds / 60)
        return f"another {minutes} minute" + ("s" if minutes != 1 else "")
    hours = round(seconds / 3600)
    return f"another {hours} hour" + ("s" if hours != 1 else "")


class OutOfAllowance(RuntimeError):
    """Every key has hit its per-minute limit.

    Worth its own name because it is the one failure that is not a fault:
    nothing is broken, the free tier is simply used up for the next minute.
    Writing code makes it far more likely - a program is two thousand output
    tokens against an eight thousand token minute - and "something went wrong
    reaching my brain" sends him looking for a problem that is not there.

    Carries how long the wait is, where the server said.
    """

    def __init__(self, message: str, wait: str = ""):
        super().__init__(message)
        self.wait = wait


class Brain:
    def __init__(self, pool=None):
        self.pool = providers.pool() if pool is None else pool
        self._clients = {}
        self.history = []
        # The last question that got no answer, so "try again" has something
        # to try. Cleared the moment one does get answered.
        self.unanswered = None

    def _client(self, index: int, provider):
        """One client per key, made on first use and kept.

        Without a timeout a stalled call blocks the whole assistant: reo
        transcribed a question once and then simply never spoke again.
        """
        if index not in self._clients:
            # max_retries=0 on purpose. The client waits out whatever
            # retry-after a 429 carries, which measured at 33 seconds of
            # silence before reo said anything at all. The pool below is the
            # retry layer: another key answers at once, and with no other key
            # there is nothing to wait for that saying so would not do better.
            patient = (config.LOCAL_TIMEOUT if provider.name == "ollama"
                       else config.REQUEST_TIMEOUT)
            self._clients[index] = OpenAI(api_key=provider.key, base_url=provider.base_url,
                                          timeout=patient, max_retries=0)
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
            max_tokens=config.MAX_REPLY_TOKENS,
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
        failures, limited, wait = [], 0, ""
        for _ in range(len(self.pool)):
            index, provider = self.pool.current()
            try:
                return self._ask(provider, self._client(index, provider), with_tools)
            except (RateLimitError, APIConnectionError, APITimeoutError) as e:
                self.pool.bench(index)
                limited += isinstance(e, RateLimitError)
                if isinstance(e, RateLimitError):
                    wait = wait or _how_long(str(e))
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
        if limited and limited == len(failures):
            raise OutOfAllowance(", ".join(failures), wait)
        raise RuntimeError("Every model key refused: " + ", ".join(failures))

    def think(self, text: str) -> str:
        self.history.append({"role": "user", "content": text})
        spent = 0.0          # seconds the tools have taken between them
        try:
            for step in range(TOOL_STEPS):
                # The tools go on every request, always.
                #
                # They are not cheap - 1151 tokens of the 8000 a minute,
                # measured against the server's own count, 1620 with them
                # against 469 without - so for a while reo left them out of
                # the last step and of anything past the tool budget, on the
                # grounds that it was not allowed to call one anyway.
                #
                # That was wrong, and measurably: asked five ordinary
                # questions with no tools in the request, gpt-oss tried to
                # call one twice, and groq answers that with 400 "Tool choice
                # is none, but model called a tool". Which reaches him as
                # "Something went wrong reaching my brain" - a line that is
                # in his log from November. A third question came back empty.
                # With the tools attached, nothing failed.
                #
                # The budget is still enforced, by refusing to run them and
                # saying so, which is the next thing below.
                resp = self._complete(with_tools=True)
                msg = resp.choices[0].message

                if not msg.tool_calls:
                    reply = tighten((msg.content or "").strip())
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
                    spoken = tighten((msg.content or "").strip())
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
                         "function": {"name": tc.function.name,
                                      "arguments": _slim(tc.function.arguments)}}
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
                # And a ceiling on the lot of them. Every tool has its own
                # timeout, but nothing added them up, and the model can ask
                # for round after round: one reply in the log took 18.5s,
                # which from where he is sitting is reo having died.
                #
                # This is how the budget is enforced, now that the tools go
                # on every request: the model may still ask, and is simply
                # told no. The calls get an answer each - the history has to
                # stay paired - it is just an answer that says to stop
                # fetching and speak.
                # Running out of steps is the same situation as running out of
                # budget, and it used to end in "Sorry, that took too many
                # steps." - which is in his log three times, for "Claude
                # app.", "So open it for me." and "Max level in this browser
                # on the video I am playing". Giving him nothing after ten
                # round trips is the worst of both. One step is held back so
                # there is always a turn left in which to say something.
                last_chance = step >= TOOL_STEPS - 2
                if spent > config.TOOL_BUDGET or last_chance:
                    why = ("that is ten rounds of tools" if last_chance
                           else f"{spent:.0f}s of tools is enough")
                    print(f"  [brain] {why}; answering with what it has")
                    results = ["That is taking too long. Answer now, out loud, "
                               "with what you already have." for _ in wanted]
                else:
                    began = time.time()
                    together = len(wanted) > 1 and all(
                        tc.function.name in READ_ONLY for tc, _ in wanted)
                    if together:
                        with ThreadPoolExecutor(max_workers=len(wanted)) as pool:
                            results = list(pool.map(
                                lambda job: run_tool(job[0].function.name, job[1]), wanted))
                    else:
                        results = [run_tool(tc.function.name, args) for tc, args in wanted]
                    spent += time.time() - began

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
