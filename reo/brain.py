import json
from datetime import datetime

from openai import BadRequestError, OpenAI

from . import config
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
    "the query, and read_page a result when the snippets fall short."
)

# Convert our tool specs to the OpenAI-compatible format
OPENAI_TOOLS = [
    {"type": "function",
     "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
    for t in TOOLS
]


class Brain:
    def __init__(self):
        # Without a timeout a stalled call blocks the whole assistant: reo
        # transcribed a question and then simply never spoke again.
        self.client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL,
                             timeout=config.REQUEST_TIMEOUT, max_retries=1)
        self.history = []

    def _complete(self, with_tools: bool):
        """One API call. gpt-oss sometimes reaches for a built-in tool we do not
        expose (notably 'refuse' when declining), which the server rejects with a
        400. Answering without tools lets it say its piece instead of crashing."""
        today = datetime.now().strftime("%A, %d %B %Y")
        kwargs = dict(
            model=config.LLM_MODEL,
            messages=[{"role": "system", "content": f"{SYSTEM} Today is {today}."}] + self.history,
            max_tokens=500,
            temperature=0.3,
        )
        if with_tools:
            kwargs.update(tools=OPENAI_TOOLS, tool_choice="auto")
        try:
            response = self.client.chat.completions.create(**kwargs)
            usage = getattr(response, "usage", None)
            session.add_request(getattr(usage, "total_tokens", 0) if usage else 0)
            return response
        except BadRequestError as e:
            if with_tools and "was not in request.tools" in str(e):
                print("  [brain] model reached for an unavailable tool; answering without tools")
                return self._complete(with_tools=False)
            raise

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
