param(
    [switch]$SkipRepair,
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Write-Step {
    param([string]$Message)
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Fail {
    param([string]$Message)
    Write-Host "`n[ERROR] $Message" -ForegroundColor Red
    exit 1
}

function Get-FreePort {
    param([int]$StartPort)

    $port = $StartPort
    while ($port -le ($StartPort + 50)) {
        $listener = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue
        if (-not $listener) {
            return $port
        }
        $port++
    }

    throw "No free port found near $StartPort."
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = (Resolve-Path (Join-Path $scriptDir '..')).Path
$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
$reflexExe = Join-Path $projectRoot '.venv\Scripts\reflex.exe'
$guiDir = Join-Path $projectRoot 'src\sportsbet\gui'

Write-Step 'Checking local environment'
if (-not (Test-Path $venvPython)) {
    Fail "Virtual environment not found: $venvPython`nRun: py -3.12 -m venv .venv"
}
if (-not (Test-Path $reflexExe)) {
    Fail 'Reflex executable not found. Run in project root: .\.venv\Scripts\python.exe -m pip install -e ".[gui]"'
}
if (-not (Test-Path $guiDir)) {
    Fail "GUI directory not found: $guiDir"
}

$nodeCmd = Get-Command node -ErrorAction SilentlyContinue
if (-not $nodeCmd) {
    $nodeHome = 'C:\Program Files\nodejs'
    $nodeExe = Join-Path $nodeHome 'node.exe'
    if (Test-Path $nodeExe) {
        $env:Path = "$nodeHome;$env:Path"
        $nodeCmd = Get-Command node -ErrorAction SilentlyContinue
    }
}
if (-not $nodeCmd) {
    Fail "Node.js not found.`nInstall: winget install OpenJS.NodeJS.LTS"
}

if (-not $SkipRepair) {
    Write-Step 'Repairing GUI dependency compatibility (safe idempotent pins)'
    & $venvPython -m pip install --disable-pip-version-check --quiet `
        "reflex==0.7.0" `
        "reflex-ag-grid>=0.0.10" `
        "nest-asyncio>=1.6.0" `
        "pytz" `
        "fastapi<0.119" `
        "pydantic<2.12" `
        "sqlmodel==0.0.22"
}

$pythonVersion = (& $venvPython -V).Trim()
$nodeVersion = (& node --version).Trim()
$frontendPort = Get-FreePort -StartPort 3000
$backendPort = Get-FreePort -StartPort 8000

Write-Step 'Launching sports-betting GUI'
Write-Host "Python : $pythonVersion"
Write-Host "Node   : $nodeVersion"
Write-Host "Frontend URL: http://localhost:$frontendPort"
Write-Host "Backend  URL: http://localhost:$backendPort"

if (-not $NoBrowser) {
    Start-Process "http://localhost:$frontendPort" | Out-Null
}

Push-Location $guiDir
try {
    & $reflexExe run --frontend-port $frontendPort --backend-port $backendPort
}
finally {
    Pop-Location
}

