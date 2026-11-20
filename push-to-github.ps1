# Put reo on GitHub. One command, after you have signed in once.
#
#   1.  gh auth login          (once, ever - it opens a browser)
#   2.  .\push-to-github.ps1
#
# It creates the repository, fills in the placeholder links in README.md and
# install.ps1 with the real address, and pushes everything.
#
# Public by default, because the installer's one-line command and the link in
# the README only work if people can reach it. Pass -Private for the other.
#
#   .\push-to-github.ps1 -Private
#   .\push-to-github.ps1 -Name my-assistant
#
# What does NOT go up, now and ever, because .gitignore says so: your API key
# (.env), the log, what reo remembers about you (memory.json), everything you
# have said to it (chats.jsonl), your voiceprint, and diagnose.wav.

[CmdletBinding()]
param(
    [string]$Name = 'reo',
    [switch]$Private,
    [switch]$Yes
)

# Continue, not Stop: gh and git write perfectly ordinary things to stderr,
# and with Stop that becomes a PowerShell error before this script can say
# anything useful about it. Every call below checks $LASTEXITCODE instead.
$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot

function Stop-With([string]$problem, [string]$fix) {
    Write-Host ""
    Write-Host "  Stopped: $problem" -ForegroundColor Red
    if ($fix) { Write-Host "  $fix" -ForegroundColor Yellow }
    Write-Host ""
    exit 1
}

# --- the tool -------------------------------------------------------------

$gh = (Get-Command gh -ErrorAction SilentlyContinue).Source
foreach ($guess in @("$env:ProgramFiles\GitHub CLI\gh.exe",
                     "$env:LOCALAPPDATA\Programs\GitHub CLI\gh.exe")) {
    if (-not $gh -and (Test-Path $guess)) { $gh = $guess }
}
if (-not $gh) {
    Stop-With "the GitHub CLI is not installed." `
              "winget install --id GitHub.cli   then open a new window."
}

& $gh auth status 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Stop-With "you are not signed in to GitHub." `
              "Run: gh auth login      then run this again."
}
$who = (& $gh api user --jq .login 2>$null | Out-String).Trim()
if (-not $who) { Stop-With "could not read your GitHub account." "" }

# --- what is about to happen ---------------------------------------------

$visibility = if ($Private) { 'private' } else { 'public' }
$where = "https://github.com/$who/$Name"
$count = (& git rev-list --count HEAD).Trim()
$files = (& git ls-files).Count

Write-Host ""
Write-Host "  About to create a $visibility repository:" -ForegroundColor Cyan
Write-Host "      $where" -ForegroundColor White
Write-Host "  and push $count commits, $files files, as $who." -ForegroundColor Cyan
Write-Host ""
Write-Host "  Not included: .env, reo.log, memory.json, chats.jsonl," -ForegroundColor DarkGray
Write-Host "  voiceprint.json, diagnose.wav - see .gitignore." -ForegroundColor DarkGray
if ($visibility -eq 'public') {
    Write-Host ""
    Write-Host "  Public means anyone can read it, and that is hard to undo:" -ForegroundColor Yellow
    Write-Host "  things that have been public get copied and cached." -ForegroundColor Yellow
}
Write-Host ""

if (-not $Yes) {
    Write-Host "  Go ahead? [y/N] " -ForegroundColor Yellow -NoNewline
    if ((Read-Host) -notmatch '^[yY]') {
        Write-Host "  Nothing done." -ForegroundColor DarkGray
        exit 0
    }
}

# --- a last look for anything that should not go out ----------------------

$leaked = & git grep -I -l -E 'gsk_[A-Za-z0-9]{10,}|sk-[A-Za-z0-9]{20,}' -- ':!*.example' 2>$null
if ($leaked) {
    Stop-With "something that looks like an API key is in a tracked file:`n    $leaked" `
              "Take it out and commit that before pushing."
}

# --- do it ----------------------------------------------------------------

$already = (& git remote get-url origin 2>$null | Out-String).Trim()
if (-not $already) {
    & $gh repo create $Name --$visibility --source=. --remote=origin
    if ($LASTEXITCODE -ne 0) { Stop-With "could not create the repository." "" }
} else {
    Write-Host "  origin is already set to $already" -ForegroundColor DarkGray
}

# The README and the installer both carry OWNER/reo as a placeholder. Nobody
# can use the one-line install until they point somewhere real.
$changed = $false
foreach ($file in @('README.md', 'install.ps1')) {
    if (-not (Test-Path $file)) { continue }
    $text = Get-Content $file -Raw
    $fixed = $text -replace 'OWNER/reo', "$who/$Name"
    if ($fixed -ne $text) {
        Set-Content $file -Value $fixed -Encoding utf8 -NoNewline
        $changed = $true
        Write-Host "  filled in the real address in $file" -ForegroundColor DarkGray
    }
}
if ($changed) {
    & git add README.md install.ps1
    & git commit -q -m "Point the install link at the repository it actually lives in"
}

$branch = (& git rev-parse --abbrev-ref HEAD).Trim()
& git push -u origin $branch
if ($LASTEXITCODE -ne 0) { Stop-With "the push failed - the message above says why." "" }

Write-Host ""
Write-Host "  Done. $where" -ForegroundColor Green
Write-Host "  Anyone can now install it with:" -ForegroundColor DarkGray
Write-Host "      irm https://raw.githubusercontent.com/$who/$Name/$branch/install.ps1 | iex" -ForegroundColor White
Write-Host ""
