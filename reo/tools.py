"""Actions reo can take. Only whitelisted, non-destructive actions for now."""
import datetime
import subprocess
import urllib.parse
import webbrowser

import psutil

# Spoken name -> Windows command. Only these can be launched.
APPS = {
    "notepad": "notepad",
    "calculator": "calc",
    "file explorer": "explorer",
    "explorer": "explorer",
    "chrome": "chrome",
    "edge": "msedge",
    "vs code": "code",
    "vscode": "code",
    "settings": "ms-settings:",
    "task manager": "taskmgr",
    "paint": "mspaint",
}


def open_app(name: str) -> str:
    target = APPS.get(name.lower().strip())
    if not target:
        return f"Not allowed or unknown app '{name}'. Allowed: {', '.join(sorted(APPS))}"
    subprocess.Popen(f'start "" {target}', shell=True)  # target comes only from the whitelist
    return f"Opened {name}."


def open_website(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    webbrowser.open(url)
    return f"Opened {url}."


def search_web(query: str) -> str:
    """Look something up and hand the text back, so reo can answer out loud."""
    try:
        from ddgs import DDGS
        results = DDGS().text(query, max_results=4)
    except Exception as e:
        return f"Could not search: {e}"
    if not results:
        return f"No results for {query}."
    return "\n".join(f"{r['title']}: {r['body']}" for r in results)


def open_search(query: str) -> str:
    """Put the results on screen when the user wants to look themselves."""
    webbrowser.open("https://www.google.com/search?q=" + urllib.parse.quote(query))
    return f"Opened a search for {query}."


def get_datetime() -> str:
    return datetime.datetime.now().strftime("%A, %d %B %Y, %I:%M %p")


def get_battery() -> str:
    b = psutil.sensors_battery()
    if b is None:
        return "No battery info available."
    return f"{b.percent:.0f}% ({'charging' if b.power_plugged else 'on battery'})"


FUNCTIONS = {
    "open_app": open_app,
    "open_website": open_website,
    "search_web": search_web,
    "open_search": open_search,
    "get_datetime": get_datetime,
    "get_battery": get_battery,
}

TOOLS = [
    {
        "name": "open_app",
        "description": "Open an application on the laptop. Allowed: " + ", ".join(sorted(APPS)),
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "App name, e.g. 'notepad'"}},
            "required": ["name"],
        },
    },
    {
        "name": "open_website",
        "description": "Open a website in the default browser.",
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "e.g. youtube.com"}},
            "required": ["url"],
        },
    },
    {
        "name": "search_web",
        "description": ("Look up current information on the web and read the results. "
                        "Use this for anything you are unsure of or that may have changed: "
                        "news, prices, scores, dates, who holds an office now."),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "open_search",
        "description": "Open a web search in the browser for the user to read themselves.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "get_datetime",
        "description": "Get the current date and time.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_battery",
        "description": "Get the laptop battery level and charging status.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


def run_tool(name: str, args: dict) -> str:
    fn = FUNCTIONS.get(name)
    if fn is None:
        return f"Unknown tool {name}."
    try:
        return fn(**args)
    except Exception as e:  # report failures to the model instead of crashing
        return f"Tool {name} failed: {e}"
