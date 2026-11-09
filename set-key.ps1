# Put a model key into .env without it ever appearing on screen, in this
# window's history, or in anything claude can read back.
#
#   powershell -ExecutionPolicy Bypass -File set-key.ps1
#
# Each key is typed straight into .env. More than one is useful: a key that hits
# its per-minute limit is benched for a minute and the next one answers, which
# is the difference between reo answering and reo apologising.

$ErrorActionPreference = "Stop"
$envFile = Join-Path $PSScriptRoot ".env"

$slots = @(
    @{ Var = "LLM_API_KEY";        Name = "Groq";       Url = "https://console.groq.com/keys" }
    @{ Var = "LLAMA_API_KEY";      Name = "Llama API";  Url = "https://llama.developer.meta.com" }
    @{ Var = "OPENROUTER_API_KEY"; Name = "OpenRouter"; Url = "https://openrouter.ai/keys" }
    @{ Var = "GEMINI_API_KEY";     Name = "Gemini";     Url = "https://aistudio.google.com/apikey" }
    @{ Var = "LLM_API_KEYS";       Name = "spare keys for the service above, comma separated"; Url = "" }
)

if (-not (Test-Path $envFile)) {
    $example = Join-Path $PSScriptRoot ".env.example"
    if (Test-Path $example) { Copy-Item $example $envFile } else { New-Item -ItemType File $envFile | Out-Null }
    Write-Host "Created .env"
}

$lines = @(Get-Content -LiteralPath $envFile)

function Set-EnvLine([string]$name, [string]$value) {
    $script:lines = @($script:lines | Where-Object { $_ -notmatch "^\s*$([regex]::Escape($name))\s*=" })
    $script:lines += "$name=$value"
}

Write-Host ""
Write-Host "Enter a key to set it, or press Enter to leave it alone."
Write-Host "Nothing you type is echoed or logged." -ForegroundColor DarkGray
Write-Host ""

$changed = 0
foreach ($slot in $slots) {
    $label = "  $($slot.Name)"
    if ($slot.Url) { $label += "  [$($slot.Url)]" }
    Write-Host $label -ForegroundColor Cyan
    $secure = Read-Host "  $($slot.Var)" -AsSecureString
    $plain = [Runtime.InteropServices.Marshal]::PtrToStringBSTR(
        [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure))
    if ($plain.Trim()) {
        Set-EnvLine $slot.Var $plain.Trim()
        Write-Host "  set ($($plain.Trim().Length) characters)" -ForegroundColor Green
        $changed++
    }
    $plain = $null
    Write-Host ""
}

if ($changed -eq 0) {
    Write-Host "Nothing changed."
    exit
}

Set-Content -LiteralPath $envFile -Value $lines -Encoding utf8
Write-Host "Wrote $changed key(s) to .env" -ForegroundColor Green
Write-Host ".env is gitignored, so the keys stay on this machine."
Write-Host "Restart reo (run.bat) to pick them up."
