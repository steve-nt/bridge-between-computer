#!/usr/bin/env pwsh
# D7078E INTERACTIVE LOAD GENERATOR - Choose endpoint dynamically

$projectDir = "C:\Users\snten\Desktop\bridge-between-computer\Mini D7078E"
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

Set-Location $projectDir

Write-Host "======================================================================" -ForegroundColor Green
Write-Host "TASK 4: INTERACTIVE LOAD GENERATOR" -ForegroundColor Green
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

# Ask for ALB URL or IP
Write-Host "======================================================================" -ForegroundColor Yellow
Write-Host "STEP 1: SELECT TARGET" -ForegroundColor Yellow
Write-Host "======================================================================" -ForegroundColor Yellow
Write-Host ""
Write-Host "Options:" -ForegroundColor Cyan
Write-Host "  1) ALB DNS (default)" -ForegroundColor Cyan
Write-Host "  2) Custom ALB/IP" -ForegroundColor Cyan
Write-Host ""

$choice = Read-Host "Select option (1 or 2)"

if ($choice -eq "2") {
    $ALB_URL = Read-Host "Enter ALB DNS or IP (e.g., http://13.61.2.53 or http://your-alb.eu-north-1.elb.amazonaws.com)"
} else {
    $ALB_URL = "http://D7078E-MINI-PROJECT-GROUP35-LB-8834940.eu-north-1.elb.amazonaws.com"
}

Write-Host ""
Write-Host "Target: $ALB_URL" -ForegroundColor Green
Write-Host ""

# Ask for endpoint
Write-Host "======================================================================" -ForegroundColor Yellow
Write-Host "STEP 2: SELECT ENDPOINT" -ForegroundColor Yellow
Write-Host "======================================================================" -ForegroundColor Yellow
Write-Host ""
Write-Host "Options:" -ForegroundColor Cyan
Write-Host "  1) / (root - normal, fast response)" -ForegroundColor Cyan
Write-Host "  2) /burn (CPU-intensive, slow response)" -ForegroundColor Cyan
Write-Host "  3) /health (health check endpoint)" -ForegroundColor Cyan
Write-Host ""

$endpointChoice = Read-Host "Select endpoint (1, 2, or 3)"

switch ($endpointChoice) {
    "2" { $ENDPOINT = "/burn"; $endpointDesc = "CPU-intensive (/burn)" }
    "3" { $ENDPOINT = "/health"; $endpointDesc = "Health check (/health)" }
    default { $ENDPOINT = "/"; $endpointDesc = "Root (/)" }
}

$FULL_URL = "$ALB_URL$ENDPOINT"
Write-Host ""
Write-Host "Endpoint: $endpointDesc" -ForegroundColor Green
Write-Host "Full URL: $FULL_URL" -ForegroundColor Green
Write-Host ""

# Ask for load profile
Write-Host "======================================================================" -ForegroundColor Yellow
Write-Host "STEP 3: SELECT LOAD PROFILE" -ForegroundColor Yellow
Write-Host "======================================================================" -ForegroundColor Yellow
Write-Host ""
Write-Host "Profiles:" -ForegroundColor Cyan
Write-Host "  1) Light Load     (40 workers x 5 RPS = 200 RPS)" -ForegroundColor Cyan
Write-Host "  2) Medium Load    (160 workers x 5 RPS = 800 RPS)" -ForegroundColor Cyan
Write-Host "  3) Heavy Load     (400 workers x 5 RPS = 2000 RPS)" -ForegroundColor Cyan
Write-Host "  4) Custom" -ForegroundColor Cyan
Write-Host ""

$profileChoice = Read-Host "Select profile (1, 2, 3, or 4)"

switch ($profileChoice) {
    "2" {
        $workers = 160
        $rps = 5
        $duration = 6000
        $profileDesc = "Medium Load (800 RPS)"
    }
    "3" {
        $workers = 400
        $rps = 5
        $duration = 6000
        $profileDesc = "Heavy Load (2000 RPS)"
    }
    "4" {
        $workers = [int](Read-Host "Enter number of workers")
        $rps = [int](Read-Host "Enter RPS per worker")
        $duration = [int](Read-Host "Enter duration in seconds")
        $totalRps = $workers * $rps
        $profileDesc = "Custom ($totalRps RPS)"
    }
    default {
        $workers = 40
        $rps = 5
        $duration = 6000
        $profileDesc = "Light Load (200 RPS)"
    }
}

Write-Host ""
Write-Host "Profile: $profileDesc" -ForegroundColor Green
Write-Host "  Workers: $workers" -ForegroundColor Green
Write-Host "  RPS per worker: $rps" -ForegroundColor Green
Write-Host "  Duration: $duration seconds" -ForegroundColor Green
Write-Host ""

# Confirm before running
Write-Host "======================================================================" -ForegroundColor Yellow
Write-Host "READY TO RUN" -ForegroundColor Yellow
Write-Host "======================================================================" -ForegroundColor Yellow
Write-Host ""
Write-Host "Target URL: $FULL_URL" -ForegroundColor Cyan
Write-Host "Load Profile: $profileDesc" -ForegroundColor Cyan
Write-Host ""

$confirm = Read-Host "Continue? (Y/N)"

if ($confirm -ne "Y" -and $confirm -ne "y") {
    Write-Host "Cancelled" -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Yellow
Write-Host "STARTING LOAD TEST" -ForegroundColor Yellow
Write-Host "======================================================================" -ForegroundColor Yellow
Write-Host "URL: $FULL_URL" -ForegroundColor Cyan
Write-Host "Workers: $workers, RPS: $rps, Duration: $duration seconds" -ForegroundColor Cyan
Write-Host "Start time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host ""

python agent.py --url $FULL_URL --workers $workers --rps $rps --duration $duration

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Green
Write-Host "LOAD TEST COMPLETE!" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
Write-Host "End time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host ""
Write-Host "Summary:" -ForegroundColor Yellow
Write-Host "  Target: $FULL_URL" -ForegroundColor Green
Write-Host "  Load: $profileDesc" -ForegroundColor Green
Write-Host "  Duration: $duration seconds" -ForegroundColor Green
Write-Host ""
