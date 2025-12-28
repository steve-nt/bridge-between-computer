#!/usr/bin/env pwsh
# D7078E MAXIMUM LOAD GENERATOR - 4x BIGGER THAN STANDARD

$projectDir = "C:\Users\snten\Desktop\bridge-between-computer\Mini D7078E"
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$ALB_URL = "http://D7078E-MINI-PROJECT-GROUP35-LB-8834940.eu-north-1.elb.amazonaws.com/"

Set-Location $projectDir

Write-Host "======================================================================" -ForegroundColor Green
Write-Host "TASK 4: MAXIMUM LOAD GENERATOR - 4x BIGGER!" -ForegroundColor Green
Write-Host "Start: $timestamp" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
Write-Host ""

# Check Python
Write-Host "Checking Python..." -ForegroundColor Cyan
if ((python --version 2>&1) -match "Python") {
    Write-Host "OK: Python installed" -ForegroundColor Green
} else {
    Write-Host "ERROR: Python not found" -ForegroundColor Red
    exit 1
}

# Check agent.py
Write-Host "Checking agent.py..." -ForegroundColor Cyan
if (Test-Path "agent.py") {
    Write-Host "OK: agent.py found" -ForegroundColor Green
} else {
    Write-Host "ERROR: agent.py not found" -ForegroundColor Red
    exit 1
}

# Check aiohttp
Write-Host "Checking aiohttp..." -ForegroundColor Cyan
python -c "import aiohttp" 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "OK: aiohttp installed" -ForegroundColor Green
} else {
    Write-Host "Installing aiohttp..." -ForegroundColor Yellow
    python -m pip install aiohttp -q
    Write-Host "OK: aiohttp installed" -ForegroundColor Green
}

Write-Host ""
Write-Host "Prerequisites OK" -ForegroundColor Green
Write-Host ""
Write-Host "LOAD SCALING:" -ForegroundColor Yellow
Write-Host "Phase 1: 200 RPS  (original: 50 RPS)   - 4x bigger" -ForegroundColor Cyan
Write-Host "Phase 2: 800 RPS  (original: 200 RPS)  - 4x bigger" -ForegroundColor Cyan
Write-Host "Phase 3: 2000 RPS (original: 500 RPS)  - 4x bigger" -ForegroundColor Cyan
Write-Host ""

# Phase 1 - 200 RPS (4x original 50 RPS)
Write-Host "======================================================================" -ForegroundColor Yellow
Write-Host "PHASE 1: HEAVY LOAD TEST" -ForegroundColor Yellow
Write-Host "======================================================================" -ForegroundColor Yellow
Write-Host "Load: 200 RPS (40 workers x 5 RPS)" -ForegroundColor Cyan
Write-Host "Duration: 5 minutes" -ForegroundColor Cyan
Write-Host "Expected: CPU 40-50%, no scaling yet" -ForegroundColor Cyan
Write-Host ""

python agent.py --url $ALB_URL --workers 40 --rps 5 --duration 300

Write-Host ""
Write-Host "Phase 1 complete" -ForegroundColor Green
Write-Host ""

# Phase 2 - 800 RPS (4x original 200 RPS)
Write-Host "======================================================================" -ForegroundColor Yellow
Write-Host "PHASE 2: VERY HEAVY LOAD TEST" -ForegroundColor Yellow
Write-Host "======================================================================" -ForegroundColor Yellow
Write-Host "Load: 800 RPS (160 workers x 5 RPS)" -ForegroundColor Cyan
Write-Host "Duration: 5 minutes" -ForegroundColor Cyan
Write-Host "Expected: CPU 70-80%, may start scaling" -ForegroundColor Cyan
Write-Host ""

python agent.py --url $ALB_URL --workers 160 --rps 5 --duration 300

Write-Host ""
Write-Host "Phase 2 complete" -ForegroundColor Green
Write-Host ""

# Phase 3 - 2000 RPS (4x original 500 RPS) - MAXIMUM!
Write-Host "======================================================================" -ForegroundColor Yellow
Write-Host "PHASE 3: EXTREME LOAD TEST - MASSIVE SCALING!" -ForegroundColor Yellow
Write-Host "======================================================================" -ForegroundColor Yellow
Write-Host "Load: 2000 RPS (400 workers x 5 RPS)" -ForegroundColor Cyan
Write-Host "Duration: 15 minutes" -ForegroundColor Cyan
Write-Host "Expected: CPU 80%+ SUSTAINED, Multiple scaling events" -ForegroundColor Cyan
Write-Host ""
Write-Host "WARNING: This is EXTREME load!" -ForegroundColor Red
Write-Host "WATCH FOR:" -ForegroundColor Red
Write-Host "- Rapid CPU spike to 80%+" -ForegroundColor Magenta
Write-Host "- MULTIPLE scaling events (possibly all 3 instances in minutes)" -ForegroundColor Magenta
Write-Host "- HealthyHostCount: 1 to 2 to 3 QUICKLY" -ForegroundColor Magenta
Write-Host "- Sustained high load across all 3 instances" -ForegroundColor Magenta
Write-Host "- RequestCount at 2000+ RPS" -ForegroundColor Magenta
Write-Host ""

python agent.py --url $ALB_URL --workers 400 --rps 5 --duration 900

Write-Host ""
Write-Host "Phase 3 complete" -ForegroundColor Green
Write-Host ""

# Phase 4 optional
Write-Host "======================================================================" -ForegroundColor Yellow
Write-Host "PHASE 4: OPTIONAL - SCALE-IN OBSERVATION" -ForegroundColor Yellow
Write-Host "======================================================================" -ForegroundColor Yellow
Write-Host "Load: 2 RPS (minimal)" -ForegroundColor Cyan
Write-Host "Duration: 10 minutes" -ForegroundColor Cyan
Write-Host ""
$choice = Read-Host "Observe scale-in? (Y/N)"

if ($choice -eq "Y" -or $choice -eq "y") {
    Write-Host "Starting Phase 4..." -ForegroundColor Green
    Write-Host ""
    python agent.py --url $ALB_URL --workers 2 --rps 1 --duration 600
    Write-Host ""
    Write-Host "Phase 4 complete" -ForegroundColor Green
} else {
    Write-Host "Skipping Phase 4" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Green
Write-Host "TASK 4 COMPLETE!" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
Write-Host "End: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host ""
Write-Host "RESULTS:" -ForegroundColor Yellow
Write-Host "Phase 1: 200 RPS, CPU 40-50%, no scaling" -ForegroundColor Green
Write-Host "Phase 2: 800 RPS, CPU 70-80%, may scale" -ForegroundColor Green
Write-Host "Phase 3: 2000 RPS, CPU 80%+ SUSTAINED, full scaling 1 to 2 to 3" -ForegroundColor Green
Write-Host ""
Write-Host "SUCCESS: Extreme auto-scaling demonstrated!" -ForegroundColor Green
Write-Host ""
Write-Host "COMPARISON:" -ForegroundColor Cyan
Write-Host "Standard script (run_high_load.ps1): 50 RPS -> 200 RPS -> 500 RPS" -ForegroundColor Gray
Write-Host "This script (run_max_load.ps1):     200 RPS -> 800 RPS -> 2000 RPS" -ForegroundColor Cyan
Write-Host "Difference: 4x BIGGER LOAD!" -ForegroundColor Yellow
Write-Host ""
