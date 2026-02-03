================================================================================
MAXIMUM LOAD SCRIPT - 4x BIGGER LOAD
================================================================================

YES! I created an even bigger load script: run_max_load.ps1

This script generates 4x MORE load than the standard high-load script!

================================================================================
LOAD COMPARISON
================================================================================

ORIGINAL (run_task4.ps1):
  Phase 1: 2 RPS
  Phase 2: 6 RPS
  Phase 3: 10 RPS

STANDARD (run_high_load.ps1):
  Phase 1: 50 RPS
  Phase 2: 200 RPS
  Phase 3: 500 RPS

MAXIMUM (run_max_load.ps1) - 4x BIGGER!
  Phase 1: 200 RPS (4x standard)
  Phase 2: 800 RPS (4x standard)
  Phase 3: 2000 RPS (4x standard) ← EXTREME!

================================================================================
HOW TO RUN
================================================================================

Open PowerShell and run:

.\run_max_load.ps1

That's it! It will:
1. Check prerequisites
2. Run Phase 1: 200 RPS for 5 minutes
3. Run Phase 2: 800 RPS for 5 minutes
4. Run Phase 3: 2000 RPS for 15 minutes ← EXTREME SCALING!
5. Ask about Phase 4 (optional)

================================================================================
WHAT TO EXPECT WITH 2000 RPS
================================================================================

Phase 3 (2000 RPS - the extreme part):

Timeline:
├─ 0-1 min: CPU rises rapidly 0% → 50%
├─ 1-2 min: CPU spikes 50% → 80%+ (very fast!)
├─ 2 min: Alarm triggers immediately! 🚨
├─ 2-3 min: First instance launches (1→2) ✨
├─ 3 min: CPU may drop slightly, then rises again
├─ 4 min: Second alarm triggers (CPU still 80%+) 🚨
├─ 4-5 min: Second instance launches (2→3) ✨
├─ 5-15 min: All 3 instances running, balanced
└─ 15 min: Phase complete

RESULT: RAPID, OBVIOUS scaling 1→2→3!

Much faster and more dramatic than standard script!

================================================================================
METRICS YOU'LL SEE
================================================================================

CPU Utilization:
  Phase 1: 40-50%
  Phase 2: 70-80%
  Phase 3: 80%+ SUSTAINED (EXTREME!)

RequestCount:
  Phase 1: ~200 requests/second
  Phase 2: ~800 requests/second
  Phase 3: ~2000 requests/second (MASSIVE!)

HealthyHostCount:
  Phase 1: 1 (no scaling)
  Phase 2: 1 or 2 (might start scaling)
  Phase 3: 1 → 2 → 3 (RAPID SCALING!) ✨

Instances running at end: 3 (maximum)

================================================================================
ADVANTAGES OVER STANDARD SCRIPT
================================================================================

Standard (500 RPS):
  ✓ Shows scaling
  ✓ CPU hits 80%
  ✓ Clear proof

Maximum (2000 RPS):
  ✓ EXTREME scaling
  ✓ Sustained 80%+ CPU
  ✓ VERY dramatic changes
  ✓ All instances used to full capacity
  ✓ Shows true maximum performance
  ✓ Undeniable proof of auto-scaling
  ✓ Much more impressive results

================================================================================
WHEN TO USE EACH
================================================================================

Use run_high_load.ps1 (500 RPS) if:
  • You want to demonstrate scaling
  • You have a slower machine
  • You want to be conservative

Use run_max_load.ps1 (2000 RPS) if:
  • You want MAXIMUM impact
  • You want EXTREME scaling demonstration
  • You want to stress-test to the limit
  • You want most impressive results
  • You want 4x more load! ← RECOMMENDED FOR IMPACT!

================================================================================
QUICK START
================================================================================

To run the MAXIMUM load script:

.\run_max_load.ps1

Expected time: 25-35 minutes

Expected result: Dramatic, obvious, impressive scaling!

================================================================================
COMPARISON TABLE
================================================================================

Metric              Standard Script     Maximum Script
─────────────────────────────────────────────────────────
Phase 1 Load        50 RPS             200 RPS (4x)
Phase 2 Load        200 RPS            800 RPS (4x)
Phase 3 Load        500 RPS            2000 RPS (4x)
CPU Phase 3         80% occasional      80%+ SUSTAINED
Scaling Speed       Moderate            VERY FAST
Scaling Visibility  Clear               EXTREME
Impact              Good                EXCELLENT
Impressiveness      Good                VERY IMPRESSIVE

================================================================================
REQUIREMENTS
================================================================================

Same as standard script:
✓ Python installed
✓ aiohttp installed (script auto-installs)
✓ agent.py in same folder
✓ ALB running and healthy
✓ CloudWatch Dashboard created

No additional requirements!

================================================================================
GO RUN IT!
================================================================================

Command:

.\run_max_load.ps1

This will generate 4x MORE load than the standard script!

Result: EXTREME auto-scaling demonstration! 🚀

Duration: ~30 minutes

Proof: HealthyHostCount 1→2→3 with sustained 2000 RPS!

================================================================================
