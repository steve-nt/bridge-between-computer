# ================================================================================
# TASK 4: LOAD GENERATOR AUTOMATION SCRIPT
# Run this script in PowerShell to execute all phases automatically
# ================================================================================

$ErrorActionPreference = "Stop"
$projectDir = "C:\Users\snten\Desktop\bridge-between-computer\Mini D7078E"
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

Write-Host "================================================================================" -ForegroundColor Green
Write-Host "TASK 4: LOAD GENERATOR - AUTOMATED EXECUTION" -ForegroundColor Green
Write-Host "Start Time: $timestamp" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Green
Write-Host ""

# Change to project directory
Set-Location $projectDir
Write-Host "Working Directory: $projectDir" -ForegroundColor Cyan
Write-Host ""

# ================================================================================
# STEP 1: BUILD DOCKER IMAGE
# ================================================================================
Write-Host "================================================================================" -ForegroundColor Yellow
Write-Host "STEP 1: BUILD DOCKER IMAGE" -ForegroundColor Yellow
Write-Host "================================================================================" -ForegroundColor Yellow
Write-Host "Building Docker image: d7078e-load-gen..." -ForegroundColor Cyan
Write-Host ""

docker build -t d7078e-load-gen .

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Docker build failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Docker image built successfully!" -ForegroundColor Green
Write-Host ""

# ================================================================================
# STEP 2: VERIFY DOCKER IMAGE
# ================================================================================
Write-Host "================================================================================" -ForegroundColor Yellow
Write-Host "STEP 2: VERIFY DOCKER IMAGE EXISTS" -ForegroundColor Yellow
Write-Host "================================================================================" -ForegroundColor Yellow
Write-Host ""

docker images d7078e-load-gen

Write-Host ""
Write-Host "Docker image verified!" -ForegroundColor Green
Write-Host ""

# ================================================================================
# STEP 3: PHASE 1 - LOW LOAD
# ================================================================================
Write-Host "================================================================================" -ForegroundColor Yellow
Write-Host "PHASE 1: LOW LOAD TEST" -ForegroundColor Yellow
Write-Host "================================================================================" -ForegroundColor Yellow
Write-Host "Running 1 container = 2 RPS (low load)" -ForegroundColor Cyan
Write-Host "Duration: 3 minutes" -ForegroundColor Cyan
Write-Host "Expected: CPU less than 30 percent, no scaling" -ForegroundColor Cyan
Write-Host ""
Write-Host "INSTRUCTIONS:" -ForegroundColor Magenta
Write-Host "1. Monitor CloudWatch dashboard in a browser" -ForegroundColor Magenta
Write-Host "2. Watch CPU stay below 30 percent" -ForegroundColor Magenta
Write-Host "3. Watch RequestCount increase slowly" -ForegroundColor Magenta
Write-Host "4. No instances should be added" -ForegroundColor Magenta
Write-Host ""
Write-Host "Starting Phase 1 (will run for 180 seconds = 3 minutes)..." -ForegroundColor Green
Write-Host ""

docker-compose up --scale agent1=1

Write-Host ""
Write-Host "Phase 1 complete!" -ForegroundColor Green
Write-Host ""

# ================================================================================
# STEP 4: PHASE 2 - MEDIUM LOAD
# ================================================================================
Write-Host "================================================================================" -ForegroundColor Yellow
Write-Host "PHASE 2: MEDIUM LOAD TEST" -ForegroundColor Yellow
Write-Host "================================================================================" -ForegroundColor Yellow
Write-Host "Running 3 containers = 6 RPS (medium load)" -ForegroundColor Cyan
Write-Host "Duration: 5 minutes" -ForegroundColor Cyan
Write-Host "Expected: CPU 50-70 percent, no scaling yet" -ForegroundColor Cyan
Write-Host ""
Write-Host "INSTRUCTIONS:" -ForegroundColor Magenta
Write-Host "1. Monitor CloudWatch dashboard" -ForegroundColor Magenta
Write-Host "2. Watch CPU rise to 50-70 percent" -ForegroundColor Magenta
Write-Host "3. Watch RequestCount increase significantly" -ForegroundColor Magenta
Write-Host "4. No scaling should occur yet (CPU less than 80 percent)" -ForegroundColor Magenta
Write-Host ""
Write-Host "Starting Phase 2 (will run for 300 seconds = 5 minutes)..." -ForegroundColor Green
Write-Host ""

docker-compose up --scale agent1=3

Write-Host ""
Write-Host "Phase 2 complete!" -ForegroundColor Green
Write-Host ""

# ================================================================================
# STEP 5: PHASE 3 - HIGH LOAD (TRIGGERS SCALING)
# ================================================================================
Write-Host "================================================================================" -ForegroundColor Yellow
Write-Host "PHASE 3: HIGH LOAD TEST - SCALING WILL TRIGGER!" -ForegroundColor Yellow
Write-Host "================================================================================" -ForegroundColor Yellow
Write-Host "Running 5 containers = 10 RPS (high load)" -ForegroundColor Cyan
Write-Host "Duration: 15 minutes" -ForegroundColor Cyan
Write-Host "Expected: CPU greater than or equal to 80 percent, SCALING: 1 to 2 to 3 instances" -ForegroundColor Cyan
Write-Host ""
Write-Host "CRITICAL OBSERVATIONS:" -ForegroundColor Magenta
Write-Host "1. Watch CloudWatch CPU rise toward 80 percent" -ForegroundColor Magenta
Write-Host "2. Watch cpu-high-alarm-80percent trigger (OK to IN_ALARM)" -ForegroundColor Magenta
Write-Host "3. Watch EC2 Instances page - new instances should appear" -ForegroundColor Magenta
Write-Host "4. Timeline of scaling events:" -ForegroundColor Magenta
Write-Host "   - approx 2 min: First instance added (1 to 2)" -ForegroundColor Magenta
Write-Host "   - approx 5 min: Second instance added (2 to 3)" -ForegroundColor Magenta
Write-Host "5. Record timestamps and CPU values!" -ForegroundColor Magenta
Write-Host ""
Write-Host "Starting Phase 3 (will run for 900 seconds = 15 minutes)..." -ForegroundColor Green
Write-Host ""

docker-compose up --scale agent1=5

Write-Host ""
Write-Host "Phase 3 complete!" -ForegroundColor Green
Write-Host ""

# ================================================================================
# STEP 6: CLEANUP AND SUMMARY
# ================================================================================
Write-Host "================================================================================" -ForegroundColor Yellow
Write-Host "CLEANUP AND SUMMARY" -ForegroundColor Yellow
Write-Host "================================================================================" -ForegroundColor Yellow
Write-Host ""

Write-Host "Stopping all containers..." -ForegroundColor Cyan
docker-compose down

Write-Host ""
Write-Host "All containers stopped!" -ForegroundColor Green
Write-Host ""

$endTime = Get-Date

Write-Host "================================================================================" -ForegroundColor Green
Write-Host "TASK 4 EXECUTION COMPLETE!" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Green
Write-Host "Start Time:   $timestamp" -ForegroundColor Cyan
Write-Host "End Time:     $endTime" -ForegroundColor Cyan
Write-Host "Total Duration: approximately 23 minutes" -ForegroundColor Cyan
Write-Host ""
Write-Host "WHAT YOU SHOULD HAVE OBSERVED:" -ForegroundColor Yellow
Write-Host "Phase 1: Low CPU (less than 30 percent), no scaling" -ForegroundColor Green
Write-Host "Phase 2: Medium CPU (50-70 percent), no scaling" -ForegroundColor Green
Write-Host "Phase 3: High CPU (80 percent or more), SCALING TRIGGERED!" -ForegroundColor Green
Write-Host "Instances increased from 1 to 2 to 3" -ForegroundColor Green
Write-Host "Alarms transitioned OK to IN_ALARM" -ForegroundColor Green
Write-Host ""
Write-Host "NEXT STEPS:" -ForegroundColor Magenta
Write-Host "1. Collect CloudWatch screenshots" -ForegroundColor Magenta
Write-Host "2. Document scaling timestamps" -ForegroundColor Magenta
Write-Host "3. Proceed to Task 5: Safe Failure Simulation" -ForegroundColor Magenta
Write-Host ""
Write-Host "================================================================================" -ForegroundColor Green
