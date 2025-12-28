================================================================================
✅ POWERSHELL LOAD TEST SCRIPTS - READY TO USE!
================================================================================

I've created a complete PowerShell solution for your D7078E Lab Task 4.

You now have:

📌 NEW FILES CREATED:
  1. run_load_test.ps1        - Main script with interactive menu
  2. start_load_test.ps1      - Quick launcher
  3. POWERSHELL_GUIDE.txt     - Complete guide (READ THIS FIRST!)
  4. QUICK_START.txt          - Copy-paste one-liners
  5. RUN_POWERSHELL_SCRIPT.txt- Step-by-step instructions
  6. START_HERE.txt           - Navigation guide

🎯 FASTEST WAY TO START (COPY-PASTE READY):

Option A: Interactive Menu (Recommended)
──────────────────────────────────────────────────────────────────────────────
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force; cd C:\Users\snten\Desktop\bridge-between-computer\Mini\ D7078E; .\run_load_test.ps1

Then select option 1, 2, or 3 when prompted.


Option B: Quick Test (Phase 1 + 2, no menu)
──────────────────────────────────────────────────────────────────────────────
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force; cd C:\Users\snten\Desktop\bridge-between-computer\Mini\ D7078E; python agent.py --url http://D7078E-MINI-PROJECT-GROUP35-LB-8834940.eu-north-1.elb.amazonaws.com/ --workers 20 --rps 5 --duration 300; Read-Host "Phase 1 done. Press Enter for Phase 2"; python agent.py --url http://D7078E-MINI-PROJECT-GROUP35-LB-8834940.eu-north-1.elb.amazonaws.com/ --workers 50 --rps 10 --duration 600

Runs Phase 1 (5 min) + Phase 2 (10 min) = 15 minutes total


Option C: Maximum Load (Instant scaling)
──────────────────────────────────────────────────────────────────────────────
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force; cd C:\Users\snten\Desktop\bridge-between-computer\Mini\ D7078E; python agent.py --url http://D7078E-MINI-PROJECT-GROUP35-LB-8834940.eu-north-1.elb.amazonaws.com/ --workers 100 --rps 20 --duration 900

Runs maximum load (2000 RPS) for 15 minutes


🎬 WHAT THE SCRIPT DOES:

Phase 1: Moderate Load
  Workers: 20, RPS per worker: 5 = 100 RPS total
  Duration: 5 minutes
  Expected: CPU 30-50%, no scaling yet

Phase 2: High Load (TRIGGERS SCALING!)
  Workers: 50, RPS per worker: 10 = 500 RPS total
  Duration: 10 minutes
  Expected: CPU 80%+, instances scale 1 → 2 → 3

Phase 3: Maximum Load (Optional)
  Workers: 100, RPS per worker: 20 = 2000 RPS total
  Duration: 15 minutes
  Expected: All 3 instances handling maximum load

Phase 4: Reduce Load (Optional, observe scale-in)
  Workers: 1, RPS per worker: 1 = 1 RPS total
  Duration: 10 minutes
  Expected: CPU drops, instances terminate 3 → 2 → 1


📊 METRICS YOU'LL WATCH:

CPU Utilization: 0% → 30% → 50% → 80%+ (scales at 80%)
RequestCount: 0 → 100 → 500 → 2000+ per second
HealthyHostCount: 1 → 2 → 3 (new instances launching)


⏱️ TIMELINE EXAMPLE:

00:00-05:00 Phase 1 - CPU rises 0% → 30-50%, no scaling
05:00-07:00 Phase 2 starts - CPU rises 50% → 80%
07:00       Alarm triggers (CPU hit 80%)
07:30       First instance launches (Hosts: 1 → 2)
08:30       CPU drops as load distributes, then rises again
09:00       Second instance launches (Hosts: 2 → 3)
09:30-15:00 Phase 2 continues - CPU stable with 3 instances


✅ BEFORE YOU RUN:

☑️ Open CloudWatch Dashboard in browser
   AWS Console → CloudWatch → Dashboards → D7078E-Mini-Project-Dashboard

☑️ Keep it visible while script runs (best: side-by-side windows)

☑️ Python installed (run: python --version)

☑️ aiohttp installed (script installs if missing)

☑️ agent.py exists in same folder (it does ✓)

☑️ ALB is running and responding

☑️ Scaling policies are active (Task 3 completed)


🚀 QUICKEST START:

1. Open PowerShell (just regular, not admin)

2. Copy-paste this line (Option A above):

Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force; cd C:\Users\snten\Desktop\bridge-between-computer\Mini\ D7078E; .\run_load_test.ps1

3. Press Enter

4. Select option 1 (Quick Test) when menu appears

5. Open CloudWatch Dashboard in browser

6. Watch it scale 1 → 2 → 3 instances! ✓


📸 WHAT TO SCREENSHOT:

During the test, take screenshots of:
  ☐ Start (baseline)
  ☐ CPU at 50%
  ☐ CPU at 80% (alarm point)
  ☐ First instance launching
  ☐ Second instance launching
  ☐ All 3 instances running
  ☐ Peak metrics
  ☐ CloudWatch Alarm (IN_ALARM state)
  ☐ ASG Activity (scaling events)
  ☐ EC2 Instances (instance count)


❓ IF SOMETHING GOES WRONG:

Check POWERSHELL_GUIDE.txt → Troubleshooting section
Or check RUN_POWERSHELL_SCRIPT.txt → Troubleshooting section


📚 DOCUMENTATION FILES:

START_HERE.txt ................. Navigation guide (read first)
POWERSHELL_GUIDE.txt ........... Complete overview
QUICK_START.txt ................ One-liner commands
RUN_POWERSHELL_SCRIPT.txt ...... Step-by-step instructions
LOAD_COMMANDS.txt .............. Direct Python commands
HIGH_LOAD_GENERATOR.txt ........ Load theory
CREATE_DASHBOARD.txt ........... Dashboard setup


🎯 YOUR CHOICE:

Quick (just run it):
→ Copy Option A command above, paste, press Enter

Want to understand first:
→ Read POWERSHELL_GUIDE.txt (10 min read)
→ Then copy Option A command

Want all details:
→ Read START_HERE.txt
→ Then read POWERSHELL_GUIDE.txt
→ Then copy Option A command


✨ SUMMARY:

✓ All scripts created
✓ All documentation written
✓ All commands ready to copy-paste
✓ Just need to run it!

The hardest part is done. Now just execute!


NEXT STEP: Copy one of the commands above and paste into PowerShell!

================================================================================
