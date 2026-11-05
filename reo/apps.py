"""Opening programs the way Windows finds them.

reo used to hunt for an .exe under C:\\Program Files. That misses anything from
the Store, because WindowsApps is protected and cannot be listed - it decided the
Claude app was not installed while that app was running on screen.

Windows keeps its own index. Get-StartApps is what the Start menu searches, and it
answers instantly. If the program is already running, "open it" nearly always means
bring it to the front.
"""
import subprocess

TIMEOUT = 25

PS_OPEN = r"""
$pattern = $env:REO_APP
$shell = New-Object -ComObject WScript.Shell

# Already running? Then bring it forward rather than starting a second copy.
$running = @(Get-Process | Where-Object {
    $_.MainWindowTitle -and ($_.Name -like $pattern -or $_.MainWindowTitle -like $pattern)
})
if ($running.Count -ge 1) {
    $null = $shell.AppActivate($running[0].Id)
    Write-Output ('FOCUSED:' + $running[0].MainWindowTitle)
    exit
}

# Otherwise ask the Start menu index, which knows about Store apps too.
$apps = @(Get-StartApps | Where-Object { $_.Name -like $pattern })
if ($apps.Count -eq 0) { Write-Output 'NOTFOUND'; exit }

$exact = @($apps | Where-Object { $_.Name -eq $env:REO_APP_EXACT })
$app = if ($exact.Count -ge 1) { $exact[0] } elseif ($apps.Count -eq 1) { $apps[0] } else { $null }
if (-not $app) {
    Write-Output ('MANY:' + (($apps | ForEach-Object { $_.Name }) -join '|'))
    exit
}

Start-Process ('shell:AppsFolder\' + $app.AppID)
Write-Output ('STARTED:' + $app.Name)
"""

PS_LIST = r"""
Get-StartApps | Select-Object -ExpandProperty Name | Sort-Object
"""


def _powershell(script: str, env: dict = None) -> str:
    import os

    environment = dict(os.environ)
    if env:
        environment.update(env)
    try:
        done = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=TIMEOUT, env=environment,
        )
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    return (done.stdout or done.stderr or "").strip()


def list_apps() -> str:
    """Everything the Start menu can launch."""
    names = [n for n in _powershell(PS_LIST).splitlines() if n.strip()]
    return "\n".join(names[:120]) if names else "Could not read the app list."


def open_app(name: str) -> str:
    """Focus the program if it is running, otherwise start it."""
    if not name:
        return "Say which app to open."

    result = _powershell(PS_OPEN, {"REO_APP": f"*{name}*", "REO_APP_EXACT": name})

    if result == "TIMEOUT":
        return "Opening that timed out."
    if result == "NOTFOUND":
        return (f"Nothing installed matches {name!r}. Use list_apps to see what is "
                "available, then try the name it uses.")
    if result.startswith("MANY:"):
        return ("Several apps match that, so nothing was opened. Ask which:\n"
                + "\n".join(result[5:].split("|")))
    if result.startswith("FOCUSED:"):
        return f'{result[8:]} was already open, so reo brought it to the front.'
    if result.startswith("STARTED:"):
        return f"Opened {result[8:]}."
    return f"Could not open it: {result[:200]}"
