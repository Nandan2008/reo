import json

from openai import OpenAI

from . import config
from .tools import TOOLS, run_tool

SYSTEM = (
    "You are Reo, a voice assistant running on the user's Windows laptop. "
    "Everything you say is spoken aloud, so reply in one or two short, natural sentences "
    "with no markdown, lists, or emoji. Use your tools to act on the laptop. "
    "Only call a tool when the user asks for an action or for information a tool provides. "
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

    def think(self, text: str) -> str:
        self.history.append({"role": "user", "content": text})
        try:
            for _ in range(5):  # max tool steps per command
                resp = self.client.chat.completions.create(
                    model=config.LLM_MODEL,
                    messages=[{"role": "system", "content": SYSTEM}] + self.history,
                    tools=OPENAI_TOOLS,
                    tool_choice="auto",
                    max_tokens=500,
                    temperature=0.3,
                )
                msg = resp.choices[0].message

                if not msg.tool_calls:
                    reply = (msg.content or "").strip()
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
