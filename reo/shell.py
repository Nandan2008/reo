"""Running commands on the machine, with the brakes that voice control needs.

Two things make this riskier here than in a chat window. Speech transcription
hallucinates - this assistant has produced whole sentences out of room noise - and
reo reads web pages, where text can be written to steer a model. Either can put a
command in front of this function that the user never asked for.

So: a small set of commands can never run, a larger set has to be confirmed out
loud first, and everything is written to the log.
"""
import re
import subprocess

from . import config

TIMEOUT = 30
OUTPUT_LIMIT = 2000

# Never, whatever anyone says. Losing these is not recoverable by apologising.
FORBIDDEN = [
    (r"\bformat\s+[a-z]:", "formatting a drive"),
    (r"\bdiskpart\b", "repartitioning disks"),
    (r"\bbcdedit\b", "editing the boot configuration"),
    (r"\bvssadmin\b.*\bdelete\b", "deleting shadow copies"),
    (r"\bcipher\b.*/w", "wiping free space"),
    (r"(remove-item|rm|del|rd|rmdir)\b[^|]*\b(c:\\?(\s|$)|c:\\windows|c:\\users(\s|\\\*|$))",
     "deleting a system or user root folder"),
    (r"-recurse[^|]*\bc:\\(\s|$|\*)", "recursively deleting the system drive"),
    (r"\bnet\s+user\b[^|]*\/(add|delete)", "adding or deleting a Windows account"),
    (r"\breg\b\s+delete\b[^|]*hklm", "deleting machine-wide registry keys"),
]

# Allowed, but it has to say what it is about to do and hear yes first.
NEEDS_CONFIRMING = [
    (r"\b(remove-item|del|erase|rd|rmdir|rm)\b", "delete files"),
    (r"\bmove-item\b|\bmove\b\s+[a-z]:", "move files"),
    (r"\b(stop-process|taskkill|kill)\b", "force a program to close"),
    (r"\b(shutdown|restart-computer|stop-computer|logoff)\b", "shut down or restart"),
    (r"\bset-executionpolicy\b", "change the script execution policy"),
    (r"\b(uninstall|msiexec)\b", "uninstall software"),
    (r"\bnew-itemproperty\b|\bset-itemproperty\b|\breg\b\s+add", "change the registry"),
    (r"\bdisable-\w+", "disable a Windows feature"),
    (r"\b(curl|wget|invoke-webrequest|iwr)\b", "download something"),
    (r"\bstart-process\b[^|]*-verb\s+runas", "run something as administrator"),
    (r">\s*[a-z]:\\|\bout-file\b|\bset-content\b", "overwrite a file"),
]


def _matches(command: str, patterns):
    lowered = command.lower()
    for pattern, description in patterns:
        if re.search(pattern, lowered):
            return description
    return None


def run_command(command: str, confirmed: bool = False) -> str:
    """Run a PowerShell command. Destructive ones need confirmed=True."""
    if not config.ALLOW_SHELL:
        return ("Running commands is switched off. The user can set REO_ALLOW_SHELL=1 "
                "in .env to allow it.")

    blocked = _matches(command, FORBIDDEN)
    if blocked:
        print(f"  [shell] REFUSED ({blocked}): {command}")
        return (f"Refused: that would mean {blocked}, which reo will not do. "
                "Tell the user to run it themselves if they really want it.")

    risky = _matches(command, NEEDS_CONFIRMING)
    if risky and not confirmed:
        print(f"  [shell] needs confirming ({risky}): {command}")
        return (f"This would {risky}. Say exactly what it will do, ask the user to "
                "confirm, and only call this again with confirmed=true if they say yes.")

    print(f"  [shell] running: {command}")
    try:
        done = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True, text=True, timeout=TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return f"That took longer than {TIMEOUT} seconds, so reo stopped it."
    except Exception as e:
        return f"Could not run it: {e}"

    output = (done.stdout or "").strip() or (done.stderr or "").strip()
    if len(output) > OUTPUT_LIMIT:
        output = output[:OUTPUT_LIMIT] + " ...(truncated)"
    if done.returncode != 0:
        return f"Exit code {done.returncode}. {output}" if output else f"Exit code {done.returncode}."
    return output or "Done."
