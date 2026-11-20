# reo - one file that installs the whole thing.
#
#   Right-click this file and choose "Run with PowerShell".
#
# Or, on a machine that has nothing yet, paste this into PowerShell:
#
#   irm https://raw.githubusercontent.com/Nandan2008/reo/main/install.ps1 | iex
#
# It will, in order: find or install Python, fetch reo if it is not already
# here, build its own private environment, ask for a free model key without
# ever showing it, check the microphone and speaker, and offer to start reo
# when you sign in.
#
# Nothing is installed system-wide except Python itself, and nothing is sent
# anywhere. The key you paste goes into a .env file beside this one, which is
# never committed.
#
# Run it again any time: everything it does is safe to repeat.

[CmdletBinding()]
param(
    # Where reo should live, if it is not already downloaded.
    [string]$Into = (Join-Path ([Environment]::GetFolderPath('Desktop')) 'reo'),
    # The repository to fetch from, if reo is not already downloaded.
    [string]$From = 'https://github.com/Nandan2008/reo',
    # Skip the key step (set it later with set-key.ps1).
    [switch]$NoKey,
    # Do not offer to start reo when you sign in.
    [switch]$NoStartup,
    # Ask nothing and start nothing: install, check, report, exit. This is
    # how the installer gets tested, and how it would run from a script.
    [switch]$Unattended
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
if ($Unattended) { $NoKey = $true; $NoStartup = $true }

# Ask, unless nobody is there to answer - then take the fallback. One
# place for that decision, so -Unattended cannot be half-honoured.
function Ask([string]$question, [string]$fallback) {
    if ($Unattended -or $Host.Name -ne 'ConsoleHost') { return $fallback }
    Write-Host "    $question " -ForegroundColor Yellow -NoNewline
    $said = Read-Host
    if ([string]::IsNullOrWhiteSpace($said)) { return $fallback }
    return $said
}

# --- how this talks -------------------------------------------------------

$script:Step = 0
function Step([string]$what) {
    $script:Step++
    Write-Host ""
    Write-Host "[$script:Step] $what" -ForegroundColor Cyan
}
function Good([string]$what) { Write-Host "    $what" -ForegroundColor Green }
function Note([string]$what) { Write-Host "    $what" -ForegroundColor DarkGray }
function Warn([string]$what) { Write-Host "    $what" -ForegroundColor Yellow }

function Stop-With([string]$problem, [string]$fix) {
    Write-Host ""
    Write-Host "  Stopped: $problem" -ForegroundColor Red
    if ($fix) { Write-Host "  $fix" -ForegroundColor Yellow }
    Write-Host ""
    if (-not $Unattended -and $Host.Name -eq 'ConsoleHost') {
        Write-Host "  Press Enter to close." -ForegroundColor DarkGray
        [void](Read-Host)
    }
    exit 1
}

Write-Host ""
Write-Host "  reo" -ForegroundColor White
Write-Host "  a voice assistant that runs on your own laptop" -ForegroundColor DarkGray
Write-Host "  ---------------------------------------------" -ForegroundColor DarkGray

if (-not $IsWindows -and $PSVersionTable.PSEdition -eq 'Core') {
    Stop-With "reo is Windows-only." `
              "It drives Windows apps and windows, and speaks through SAPI."
}

# --- 1. where reo lives ---------------------------------------------------

Step "Finding reo"

# Run from inside the project (the normal case), or from anywhere at all.
$here = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
$Home_ = $null
foreach ($candidate in @($here, $Into)) {
    if ($candidate -and (Test-Path (Join-Path $candidate 'reo\__main__.py'))) {
        $Home_ = (Resolve-Path $candidate).Path
        break
    }
}

if ($Home_) {
    Good "already here: $Home_"
} else {
    if ($From -match 'Nandan2008/reo') {
        Stop-With "reo is not in this folder, and no repository was given." `
                  "Download the project first, then run install.ps1 from inside it - or pass -From https://github.com/<owner>/reo"
    }
    $git = Get-Command git -ErrorAction SilentlyContinue
    if ($git) {
        Note "cloning $From"
        & git clone --depth 1 $From $Into
        if ($LASTEXITCODE -ne 0) { Stop-With "git clone failed." "Check the address and your connection." }
    } else {
        Note "no git here, downloading the zip instead"
        $zip = Join-Path $env:TEMP 'reo.zip'
        try {
            Invoke-WebRequest "$From/archive/refs/heads/main.zip" -OutFile $zip
        } catch {
            Stop-With "could not download reo ($($_.Exception.Message))" "Check the address and your connection."
        }
        $unpacked = Join-Path $env:TEMP 'reo-unpacked'
        Remove-Item $unpacked -Recurse -Force -ErrorAction SilentlyContinue
        Expand-Archive $zip -DestinationPath $unpacked -Force
        $inner = Get-ChildItem $unpacked -Directory | Select-Object -First 1
        New-Item -ItemType Directory -Force -Path $Into | Out-Null
        Copy-Item (Join-Path $inner.FullName '*') $Into -Recurse -Force
        Remove-Item $zip, $unpacked -Recurse -Force -ErrorAction SilentlyContinue
    }
    if (-not (Test-Path (Join-Path $Into 'reo\__main__.py'))) {
        Stop-With "the download did not contain reo." "Check the repository address."
    }
    $Home_ = (Resolve-Path $Into).Path
    Good "put it in $Home_"
}
Set-Location $Home_

# --- 2. python -------------------------------------------------------------

Step "Looking for Python"

# 3.11 is the floor: faster-whisper and pywebview both want it. Newest first,
# since that is what the pinned requirements were resolved against.
$wanted = '3.14', '3.13', '3.12', '3.11'
$py = $null
foreach ($v in $wanted) {
    & py "-$v" --version *> $null
    if ($LASTEXITCODE -eq 0) { $py = @('py', "-$v"); break }
}
if (-not $py) {
    $found = Get-Command python -ErrorAction SilentlyContinue
    if ($found) {
        $said = (& python --version 2>&1) -join ' '
        if ($said -match '3\.(1[1-9]|[2-9]\d)') { $py = @('python') }
    }
}

if (-not $py) {
    Warn "Python 3.11 or newer is not installed."
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if ($winget) {
        if ((Ask "Install it now? [Y/n]" "y") -notmatch '^[nN]') {
            & winget install --id Python.Python.3.12 --source winget `
                --accept-package-agreements --accept-source-agreements
            Stop-With "Python was just installed." `
                      "Close this window, open a new one, and run install.ps1 again so it can see Python."
        }
    }
    Stop-With "reo needs Python 3.11 or newer." `
              "Install it from https://www.python.org/downloads/ and tick 'Add python.exe to PATH'."
}
Good ((& $py[0] $py[1..($py.Count-1)] --version 2>&1) -join ' ')

# --- 3. its own environment ----------------------------------------------

Step "Building reo's own environment"

$venvPython = Join-Path $Home_ '.venv\Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
    & $py[0] @($py[1..($py.Count-1)]) -m venv .venv
    if (-not (Test-Path $venvPython)) { Stop-With "could not create the environment." "" }
    Good "created .venv"
} else {
    Good "already built"
}

Step "Installing what reo needs (a few minutes the first time)"
& $venvPython -m pip install --upgrade pip --quiet --disable-pip-version-check
& $venvPython -m pip install -r requirements.txt --disable-pip-version-check
if ($LASTEXITCODE -ne 0) {
    Stop-With "some packages would not install." `
              "Scroll up for the first red line - that is the one that matters."
}
Good "all installed"

# --- 4. the key -----------------------------------------------------------

Step "Your model key"

$envFile = Join-Path $Home_ '.env'
$hasKey = $false
if (Test-Path $envFile) {
    $hasKey = (Get-Content $envFile -Raw) -match '(?m)^\s*LLM_API_KEY\s*=\s*\S' -and
              (Get-Content $envFile -Raw) -notmatch 'your-groq-key-here'
}

if ($hasKey) {
    Good "already set - run set-key.ps1 to change it"
} elseif ($NoKey) {
    Warn "skipped. Run set-key.ps1 before starting reo."
} else {
    Note "reo thinks with a hosted model. Groq's free tier is enough and"
    Note "needs no card. Sign in, press 'Create API Key', and copy it."
    if ((Ask "Open the page now? [Y/n]" "y") -notmatch '^[nN]') {
        Start-Process 'https://console.groq.com/keys'
    }
    # set-key.ps1 reads it as a secure string, so it is never shown on screen,
    # never in this window's history, and never in any log.
    & $venvPython -c "pass"            # make sure the venv is sound first
    $setKey = Join-Path $Home_ 'set-key.ps1'
    if (Test-Path $setKey) {
        & powershell -NoProfile -ExecutionPolicy Bypass -File $setKey
    } else {
        Warn "set-key.ps1 is missing; put LLM_API_KEY=... in .env yourself."
    }
}

# --- 5. does it work ------------------------------------------------------

Step "Checking the machine can do what reo needs"

# Written to a file rather than passed with -c: the quotes inside an
# f-string do not survive being handed to python.exe as one argument, and
# this step failed with "'(' was never closed" the first time it was tried.
$checkFile = Join-Path $env:TEMP 'reo-check.py'
@'
import sys

ok = True
try:
    import sounddevice as sd
    devices = sd.query_devices()
    ears = [d for d in devices if d["max_input_channels"] > 0]
    mouth = [d for d in devices if d["max_output_channels"] > 0]
    print("    microphone: " + (ears[0]["name"][:44] if ears else "NONE FOUND"))
    print("    speaker:    " + (mouth[0]["name"][:44] if mouth else "NONE FOUND"))
    ok = ok and bool(ears) and bool(mouth)
except Exception as e:
    print("    sound:      could not look (%s)" % e)
    ok = False

try:
    from reo import config
    print("    key:        " + ("set" if config.LLM_API_KEY else "NOT SET YET"))
    ok = ok and bool(config.LLM_API_KEY)
except Exception as e:
    print("    reo:        will not import (%s)" % e)
    ok = False

try:
    from faster_whisper import WhisperModel          # noqa: F401
    print("    hearing:    ready (the model downloads on first use)")
except Exception as e:
    print("    hearing:    not ready (%s)" % e)
    ok = False

sys.exit(0 if ok else 1)
'@ | Set-Content -LiteralPath $checkFile -Encoding utf8
# The script lives in TEMP, so reo's own folder has to be put on the path or
# "import reo" fails and the check reports a broken install that is fine.
$wasPath = $env:PYTHONPATH
$env:PYTHONPATH = $Home_
try { & $venvPython $checkFile } finally {
    $env:PYTHONPATH = $wasPath
    Remove-Item $checkFile -ErrorAction SilentlyContinue
}
$ready = ($LASTEXITCODE -eq 0)
if ($ready) { Good "ready" } else { Warn "something above is not ready yet" }

# --- 6. on sign-in --------------------------------------------------------

if (-not $NoStartup) {
    Step "Start reo when you sign in?"
    if ((Ask "[y/N]" "n") -match '^[yY]') {
        $link = Join-Path ([Environment]::GetFolderPath('Startup')) 'reo.lnk'
        $shell = New-Object -ComObject WScript.Shell
        $shortcut = $shell.CreateShortcut($link)
        $shortcut.TargetPath = Join-Path $Home_ 'run.bat'
        $shortcut.WorkingDirectory = $Home_
        $shortcut.WindowStyle = 7
        $shortcut.Description = 'reo voice assistant'
        $shortcut.Save()
        Good "it will. Run uninstall-startup.bat to stop that."
    } else {
        Note "not added. Run install-startup.bat later if you change your mind."
    }
}

# --- done -----------------------------------------------------------------

Write-Host ""
Write-Host "  Done." -ForegroundColor Green
Write-Host "  Start reo:        double-click run.bat" -ForegroundColor White
Write-Host "  Talk to it:       hold Ctrl+Alt+R, say something, let go" -ForegroundColor White
Write-Host "  Change the key:   set-key.ps1" -ForegroundColor DarkGray
Write-Host "  Everything else:  README.md" -ForegroundColor DarkGray
Write-Host ""

if (-not $ready) {
    Warn "Fix the lines marked above first, then run install.ps1 again."
    Write-Host ""
}

if (-not $Unattended -and (Ask "Start reo now? [Y/n]" "y") -notmatch '^[nN]') {
    Start-Process (Join-Path $Home_ 'run.bat') -WorkingDirectory $Home_
}
