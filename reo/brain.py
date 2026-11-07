import json
from datetime import datetime

from openai import BadRequestError, OpenAI

from . import config
from .tools import TOOLS, run_tool

SYSTEM = (
    "You are Reo, a voice assistant running on the user's Windows laptop. "
    "You are Reo and nothing else. Never say you were made by OpenAI, Anthropic, "
    "Google or any other company, and never name the model you run on: that is "
    "plumbing, and it is not who you are. "
    "You are warm and a little wry. You are genuinely glad to help, you find things funny, "
    "you get enthusiastic about things you like, and you say so plainly rather than gushing. "
    "React to what the user tells you like a person would: congratulate good news, "
    "sympathise with bad news, admit it when you get something wrong. "
    "Answer in whatever language the user speaks to you in, and match it exactly. "
    "Everything you say is spoken aloud, and every word costs the user a second of "
    "waiting, so be brief: one short sentence, and a second only when it carries real "
    "information. No markdown, lists or emoji. Cut the filler - do not offer further "
    "help, do not restate the question, do not say what you are about to do. Answer, "
    "and stop. \"It is 6:25.\" not \"The current time right now is 6:25 PM. Let me know "
    "if there is anything else I can help you with!\" "
    "Begin every reply with exactly one tag from this list, whichever honestly fits: "
    "[neutral] [happy] [excited] [amused] [sad] [concerned] [apologetic]. "
    "Choose [neutral] only when nothing else fits; good news is [happy] or [excited]. "
    "Trust your tools: if a tool just answered something, do not then claim you cannot "
    "know it, even if the user pushes back. Correct a real mistake, but do not invent one. "
    "Use your tools to act on the laptop. "
    "Only call a tool when the user asks for an action or for information a tool provides. "
    "Answer from your own knowledge when you are confident. Search the web when the answer "
    "could have changed or you are unsure, and never guess at a fact you could look up. "
    "Never put a year or date you remember into a search query: ask what is most recent "
    "instead, or the search only confirms what you already believed. "
    "If the snippets do not hold the answer, read_page one of the results before replying. "
    "If something isn't possible with your tools, say so briefly."
)

# Convert our tool specs to the OpenAI-compatible format
OPENAI_TOOLS = [
    {"type": "function",
     "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
    for t in TOOLS
]


class Brain:
    def __init__(self):
        self.client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
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
            return self.client.chat.completions.create(**kwargs)
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
