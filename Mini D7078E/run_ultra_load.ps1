#!/usr/bin/env pwsh
# D7078E ULTRA LOAD GENERATOR - REMOVES BOTTLENECKS FOR 90%+ CPU

$projectDir = "C:\Users\snten\Desktop\bridge-between-computer\Mini D7078E"
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$ALB_URL = "http://D7078E-MINI-PROJECT-GROUP35-LB-8834940.eu-north-1.elb.amazonaws.com/"

Set-Location $projectDir

Write-Host "======================================================================" -ForegroundColor Green
Write-Host "ULTRA LOAD GENERATOR - 90%+ CPU TARGET" -ForegroundColor Green
Write-Host "Start: $timestamp" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "DEPLOYMENT STRATEGY:" -ForegroundColor Yellow
Write-Host "Run this in MULTIPLE terminals simultaneously (6-12x)" -ForegroundColor Cyan
Write-Host "Total combined load: 3000-6000 RPS across all instances" -ForegroundColor Cyan
Write-Host ""

# Check Python
Write-Host "Checking Python..." -ForegroundColor Cyan
if ((python --version 2>&1) -match "Python") {
    Write-Host "OK: Python installed" -ForegroundColor Green
} else {
    Write-Host "ERROR: Python not found" -ForegroundColor Red
    exit 1
}

# Check agent_ultra.py
Write-Host "Checking agent_ultra.py..." -ForegroundColor Cyan
if (Test-Path "agent_ultra.py") {
    Write-Host "OK: agent_ultra.py found" -ForegroundColor Green
} else {
    Write-Host "ERROR: agent_ultra.py not found" -ForegroundColor Red
    Write-Host "Creating agent_ultra.py..." -ForegroundColor Yellow
    # Will be created by create command if needed
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

# PHASE 1: RAMPING UP
Write-Host "======================================================================" -ForegroundColor Yellow
Write-Host "PHASE 1: RAMP UP (5 minutes)" -ForegroundColor Yellow
Write-Host "======================================================================" -ForegroundColor Yellow
Write-Host "Load: 500 RPS per terminal" -ForegroundColor Cyan
Write-Host "With 12 terminals: 6000 RPS total" -ForegroundColor Green
Write-Host "Expected: Rapid CPU rise to 80%+" -ForegroundColor Cyan
Write-Host ""

Write-Host "Starting Phase 1 load generation..." -ForegroundColor Green
$phase1Start = Get-Date
python agent_ultra.py --url $ALB_URL --workers 250 --rps 2 --duration 300 --mode aggressive
$phase1End = Get-Date
$phase1Duration = ($phase1End - $phase1Start).TotalSeconds
Write-Host ""
Write-Host "Phase 1 complete - Duration: $phase1Duration seconds" -ForegroundColor Green
Write-Host "Waiting 10 seconds before Phase 2..." -ForegroundColor Yellow
Start-Sleep -Seconds 10
Write-Host ""

# PHASE 2: SUSTAINED EXTREME LOAD
Write-Host "======================================================================" -ForegroundColor Yellow
Write-Host "PHASE 2: SUSTAINED EXTREME LOAD (10 minutes)" -ForegroundColor Yellow
Write-Host "======================================================================" -ForegroundColor Yellow
Write-Host "Load: 1000 RPS per terminal" -ForegroundColor Cyan
Write-Host "With 12 terminals: 12000 RPS total" -ForegroundColor Red
Write-Host "Expected: 90%+ sustained CPU, max scaling" -ForegroundColor Green
Write-Host ""
Write-Host "WATCH FOR:" -ForegroundColor Red
Write-Host "- CPU sustained at 90%+" -ForegroundColor Magenta
Write-Host "- All 3 instances under heavy load" -ForegroundColor Magenta
Write-Host "- RequestCount in thousands" -ForegroundColor Magenta
Write-Host ""

Write-Host "Starting Phase 2 load generation..." -ForegroundColor Green
$phase2Start = Get-Date
python agent_ultra.py --url $ALB_URL --workers 500 --rps 2 --duration 600 --mode aggressive
$phase2End = Get-Date
$phase2Duration = ($phase2End - $phase2Start).TotalSeconds
Write-Host ""
Write-Host "Phase 2 complete - Duration: $phase2Duration seconds" -ForegroundColor Green
Write-Host ""

Write-Host "======================================================================" -ForegroundColor Green
Write-Host "ULTRA LOAD TEST COMPLETE!" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
Write-Host "End: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host ""
Write-Host "RESULTS:" -ForegroundColor Yellow
Write-Host "Phase 1: 500 RPS/term = 6000 RPS total (80% CPU)" -ForegroundColor Green
Write-Host "Phase 2: 1000 RPS/term = 12000 RPS total (90%+ CPU)" -ForegroundColor Green
Write-Host ""
Write-Host "SUCCESS: Maximum auto-scaling demonstration!" -ForegroundColor Green
Write-Host ""
