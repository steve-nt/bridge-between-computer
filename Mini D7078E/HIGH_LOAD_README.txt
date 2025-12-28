================================================================================
HIGH-LOAD TASK 4 SCRIPT - QUICK START
================================================================================

SCRIPT FILE: run_high_load_task4.ps1

LOAD COMPARISON:
Original (run_task4.ps1):
  Phase 1: 2 RPS
  Phase 2: 6 RPS
  Phase 3: 10 RPS

NEW (run_high_load_task4.ps1):
  Phase 1: 50 RPS (25x more)
  Phase 2: 200 RPS (33x more)
  Phase 3: 500 RPS (50x more) ← 🚀 MUCH MORE AGGRESSIVE!

================================================================================
HOW TO RUN
================================================================================

Open PowerShell and paste this:

& 'C:\Users\snten\Desktop\bridge-between-computer\Mini D7078E\run_high_load_task4.ps1'

That's it! The script will:
1. Check prerequisites (Python, aiohttp, ALB connectivity)
2. Run Phase 1 (50 RPS for 5 min) - Low load, baseline
3. Run Phase 2 (200 RPS for 5 min) - Medium load, approaching threshold
4. Run Phase 3 (500 RPS for 15 min) - HIGH LOAD, TRIGGERS SCALING! ✨
5. Ask if you want Phase 4 (scale-in observation)

Total time: ~25-35 minutes

================================================================================
EXPECTED RESULTS
================================================================================

Phase 1 (50 RPS, 5 min):
  CPU: 20-30%
  Hosts: 1
  Status: No scaling

Phase 2 (200 RPS, 5 min):
  CPU: 40-60%
  Hosts: 1
  Status: No scaling yet

Phase 3 (500 RPS, 15 min) ← KEY PHASE
  ⏱️ 0-2 min: CPU rises 50%→80%
  ⏱️ 2-3 min: cpu-high-alarm-80percent triggers 🚨
  ⏱️ 3-5 min: First instance launches (1→2) ✨
  ⏱️ 5-7 min: CPU rises again, alarm fires
  ⏱️ 7-9 min: Second instance launches (2→3) ✨
  ⏱️ 9-15 min: All 3 instances running, balanced load
  Final CPU: 40-60% (distributed across 3)
  Final Hosts: 3

Phase 4 (2 RPS, 10 min) - OPTIONAL
  CPU: drops to 0-5%
  After 5 min: cpu-low-alarm-30percent triggers
  Hosts: 3 → 2 → 1
  Watch scale-in in action

================================================================================
WHAT TO WATCH
================================================================================

CloudWatch Dashboard (D7078E-Mini-Project-Dashboard):

1. CPU Utilization
   - Should spike much faster than original
   - Should hit 80%+ by minute 2-3 of Phase 3

2. RequestCount
   - Shows requests per second
   - Phase 3 should show ~500 RPS (huge spike!)

3. HealthyHostCount
   - This is the proof of scaling!
   - Should jump: 1 → 2 → 3 during Phase 3

CloudWatch Alarms:
  - Go to: CloudWatch > Alarms > cpu-high-alarm-80percent
  - Watch it transition: OK → IN_ALARM → OK

ASG Activity:
  - Go to: EC2 > Auto Scaling Groups > D7078E-MINI-PROJECT-GROUP35-ASG
  - Click Activity tab
  - See scaling events in real-time

EC2 Instances:
  - Go to: EC2 > Instances
  - Watch instance count increase 1 → 2 → 3

================================================================================
SCREENSHOT CHECKLIST
================================================================================

Take screenshots at these moments (especially during Phase 3):

Before Test:
  ☐ Baseline dashboard (1 instance, low metrics)

Phase 1 (Low Load):
  ☐ CPU at 20-30%, Hosts: 1

Phase 2 (Medium Load):
  ☐ CPU at 40-60%, Hosts: 1

Phase 3 (High Load) - MOST IMPORTANT:
  ☐ CPU rising toward 80%
  ☐ CPU at 80% (alarm point)
  ☐ Alarm state: IN_ALARM
  ☐ First instance launching (Hosts: 1→2)
  ☐ RequestCount showing 500+ RPS spike
  ☐ Second instance launching (Hosts: 2→3)
  ☐ All 3 instances running and healthy
  ☐ ASG Activity showing scaling events
  ☐ EC2 Instances showing 3 running

Phase 4 (Optional, Scale-in):
  ☐ CPU dropping to 0-5%
  ☐ Instances terminating (3→2→1)

Final:
  ☐ Dashboard with all metrics visible

These screenshots prove scaling works!

================================================================================
COMPARISON TO ORIGINAL
================================================================================

Original Script (run_task4.ps1):
  • Uses Docker containers
  • Low load (max 10 RPS)
  • May not consistently trigger scaling
  • Takes ~23 minutes

New Script (run_high_load_task4.ps1):
  • Direct Python load generation
  • AGGRESSIVE load (max 500 RPS)
  • WILL trigger scaling consistently
  • Takes ~25-35 minutes (includes optional scale-in)
  • 50x MORE LOAD in Phase 3!
  • Easier to see results

================================================================================
IF THINGS DON'T WORK
================================================================================

Script won't run:
  → Make sure Python is installed: python --version
  → Make sure aiohttp is available: pip install aiohttp
  → Make sure agent.py exists in same folder

CPU doesn't reach 80%:
  → This script uses MUCH more aggressive load
  → Should definitely trigger scaling
  → Check ALB is responding: open $ALB_URL in browser
  → Check ASG max is 3 instances (not lower)

No instances scale:
  → Check scaling policies exist in AWS
  → Check alarms exist and are enabled
  → Wait 2+ minutes (alarms need time to evaluate)
  → Check Instance type (t2.micro may struggle - consider t2.small)

Dashboard not updating:
  → Refresh browser
  → Wait 30-60 seconds for metrics to appear
  → Make sure dashboard widgets are configured

================================================================================
KEY DIFFERENCES FROM ORIGINAL
================================================================================

Original uses docker-compose:
  • docker-compose up --scale agent1=1 (2 RPS)
  • docker-compose up --scale agent1=3 (6 RPS)
  • docker-compose up --scale agent1=5 (10 RPS)

New uses Python directly:
  • python agent.py --workers 10 --rps 5 (50 RPS)
  • python agent.py --workers 40 --rps 5 (200 RPS)
  • python agent.py --workers 100 --rps 5 (500 RPS)

Result: MUCH MORE LOAD → FASTER, MORE OBVIOUS SCALING!

================================================================================
TIMELINE (Estimated)
================================================================================

00:00-05:00  Phase 1: 50 RPS, CPU 20-30%, no scaling
05:00-10:00  Phase 2: 200 RPS, CPU 40-60%, no scaling
10:00-12:00  Phase 3: CPU rises 50%→80%, alarm about to trigger
12:00-13:00  First alarm triggers, instance launches
13:00-15:00  CPU drops as load distributes, then rises again
15:00-16:00  Second alarm triggers, instance launches
16:00-25:00  All 3 instances running, balanced load
25:00-35:00  Phase 4 (optional): Scale-in observation
35:00        Complete!

Key moment: Minute 12 is when you'll see the first scaling event!

================================================================================
SUCCESS CRITERIA
================================================================================

You've successfully completed the high-load test when:

✓ Phase 1 runs with low CPU (20-30%)
✓ Phase 2 runs with medium CPU (40-60%)
✓ Phase 3 CPU spikes to 80%+
✓ cpu-high-alarm-80percent triggers (goes IN_ALARM)
✓ HealthyHostCount increases: 1→2→3 (VISIBLE IN DASHBOARD!)
✓ All 3 instances show healthy in ASG
✓ ASG Activity shows "Launching 1 new instance" (twice)
✓ RequestCount shows 500+ RPS during Phase 3
✓ You have screenshots of all events

If all above are true: TASK 4 COMPLETE! ✅

Much more dramatic than original because:
  • Load is 50x higher
  • Scaling happens much faster
  • Results are very obvious
  • Clear evidence of auto-scaling working

================================================================================
RUN NOW!
================================================================================

Copy this command:

& 'C:\Users\snten\Desktop\bridge-between-computer\Mini D7078E\run_high_load_task4.ps1'

Paste into PowerShell and press Enter!

Watch your infrastructure scale 1→2→3 with AGGRESSIVE load! 🚀

Good luck!

================================================================================
