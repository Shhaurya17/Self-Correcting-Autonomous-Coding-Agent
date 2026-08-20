# Runs the agent in Docker, reading GEMINI_API_KEY from .env so it
# doesn't need to be retyped on every run.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$envFile = ".env"
if (-not (Test-Path $envFile)) {
    Write-Error ".env not found. Copy .env.example to .env and fill in GEMINI_API_KEY."
    exit 1
}

$apiKeyLine = Get-Content $envFile | Where-Object { $_ -match "^GEMINI_API_KEY=" }
if (-not $apiKeyLine) {
    Write-Error "GEMINI_API_KEY not set in .env."
    exit 1
}
$apiKey = ($apiKeyLine -split "=", 2)[1].Trim()
if (-not $apiKey) {
    Write-Error "GEMINI_API_KEY is empty in .env."
    exit 1
}

docker run --rm -it `
    -e GEMINI_API_KEY=$apiKey `
    -v "${PSScriptRoot}\workspace:/workspace" `
    coding-agent
