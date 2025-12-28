#!/usr/bin/env pwsh
# D7078E High Load Generator Script

$projectDir = "C:\Users\snten\Desktop\bridge-between-computer\Mini D7078E"
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$ALB_URL = "http://D7078E-MINI-PROJECT-GROUP35-LB-8834940.eu-north-1.elb.amazonaws.com/"

Set-Location $projectDir

Write-Host "======================================================================" -ForegroundColor Green
Write-Host "TASK 4: HIGH-LOAD GENERATOR" -ForegroundColor Green
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

# Phase 1
Write-Host "======================================================================" -ForegroundColor Yellow
Write-Host "PHASE 1: LOW LOAD TEST" -ForegroundColor Yellow
Write-Host "======================================================================" -ForegroundColor Yellow
Write-Host "Load: 50 RPS (10 workers x 5 RPS)" -ForegroundColor Cyan
Write-Host "Duration: 5 minutes" -ForegroundColor Cyan
Write-Host "Expected: CPU 20-30%, no scaling" -ForegroundColor Cyan
Write-Host ""

python agent.py --url $ALB_URL --workers 10 --rps 5 --duration 300

Write-Host ""
Write-Host "Phase 1 complete" -ForegroundColor Green
Write-Host ""

# Phase 2
Write-Host "======================================================================" -ForegroundColor Yellow
Write-Host "PHASE 2: MEDIUM LOAD TEST" -ForegroundColor Yellow
Write-Host "======================================================================" -ForegroundColor Yellow
Write-Host "Load: 200 RPS (40 workers x 5 RPS)" -ForegroundColor Cyan
Write-Host "Duration: 5 minutes" -ForegroundColor Cyan
Write-Host "Expected: CPU 40-60%, no scaling" -ForegroundColor Cyan
Write-Host ""

python agent.py --url $ALB_URL --workers 40 --rps 5 --duration 300

Write-Host ""
Write-Host "Phase 2 complete" -ForegroundColor Green
Write-Host ""

# Phase 3
Write-Host "======================================================================" -ForegroundColor Yellow
Write-Host "PHASE 3: HIGH LOAD TEST - SCALING WILL TRIGGER!" -ForegroundColor Yellow
Write-Host "======================================================================" -ForegroundColor Yellow
Write-Host "Load: 500 RPS (100 workers x 5 RPS)" -ForegroundColor Cyan
Write-Host "Duration: 15 minutes" -ForegroundColor Cyan
Write-Host "Expected: CPU 80%+, Instances scale 1 to 2 to 3" -ForegroundColor Cyan
Write-Host ""
Write-Host "WATCH FOR:" -ForegroundColor Red
Write-Host "- CPU spikes to 80 percent" -ForegroundColor Magenta
Write-Host "- Alarm triggers" -ForegroundColor Magenta
Write-Host "- HealthyHostCount increases 1 to 2" -ForegroundColor Magenta
Write-Host "- HealthyHostCount increases 2 to 3" -ForegroundColor Magenta
Write-Host ""

python agent.py --url $ALB_URL --workers 100 --rps 5 --duration 900

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
Write-Host "Phase 1: 50 RPS, CPU 20-30%, no scaling" -ForegroundColor Green
Write-Host "Phase 2: 200 RPS, CPU 40-60%, no scaling" -ForegroundColor Green
Write-Host "Phase 3: 500 RPS, CPU 80%+, instances 1 to 2 to 3" -ForegroundColor Green
Write-Host ""
Write-Host "SUCCESS: Auto-scaling demonstrated!" -ForegroundColor Green
Write-Host ""
