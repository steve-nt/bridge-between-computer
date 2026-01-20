# TASK 5 & 6 Quick Start Checklist

## 🎯 Your Goal
Complete **Task 5** (Failure Simulation) and **Task 6** (Evidence Collection) to demonstrate automatic failover and recovery in your ASG.

---

## ✅ PRE-FLIGHT CHECKLIST (Before You Start)

### Infrastructure Ready?
- [ ] 3 instances running in ASG (verify: EC2 > Instances)
- [ ] All 3 instances "Healthy" in Target Group
- [ ] Load generators running (traffic going to ALB)
- [ ] CloudWatch dashboard open and visible
- [ ] AWS CLI configured (test: `aws sts get-caller-identity`)
- [ ] Region: eu-north-1 selected

### Have You Set Up SSM?
- [ ] Created IAM role: `D7078E-EC2-SSM-Role`
- [ ] Updated Launch Template with SSM role
- [ ] Terminated old instances (ASG relaunched with role)
- [ ] New instances online in Fleet Manager
- [ ] SSM send-command works (test on one instance)

If not, follow: `SSM_SETUP_DETAILED.txt`

---

## 🚀 TASK 5: FAILURE SIMULATION (15-20 minutes)

### Step 1: Document Pre-Failure State (2 minutes)

```bash
# Take these screenshots:
[ ] Screenshot 1: EC2 Instances showing all 3 running
[ ] Screenshot 2: Target Group showing all 3 "Healthy"
[ ] Screenshot 3: CloudWatch CPU showing ~60-80% (from load)
[ ] Screenshot 4: ASG Activity showing no recent scaling
```

### Step 2: Trigger Failure (2 minutes)

**Option A: SSM Stress Command (RECOMMENDED)**

```bash
# Get one instance ID:
INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:aws:autoscaling:groupName,Values=D7078E-MINI-PROJECT-GROUP35-ASG" \
  --region eu-north-1 \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text)

echo "Stressing instance: $INSTANCE_ID"

# Run stress command:
aws ssm send-command \
  --instance-ids $INSTANCE_ID \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["sudo apt-get update -y || true","sudo apt-get install -y stress-ng || true","stress-ng -c 0 -l 90 -t 120s"]' \
  --region eu-north-1

# Note the timestamp here:
echo "Failure triggered at: $(date)"
```

**Option B: FIS Experiment**

```
1. Go to: Fault Injection Simulator > Create experiment
2. Name: D7078E-Task5-CPU-Stress
3. Target: Your 3 ASG instances
4. Action: CPU (or stop instances)
5. Duration: 120 seconds
6. Click "Create and Start"
7. Note the start time
```

[ ] Failure simulation started
[ ] Exact timestamp: _______________

### Step 3: Monitor Failure Timeline (5-7 minutes)

**Minute 0-1: Health Check Failure**

Keep CloudWatch open, watch for:
- [ ] CPU spike on one instance to 90%+ (within 30 seconds)
- [ ] HealthyHostCount dropdown (3 → 2)
- [ ] Target Group shows one instance as "Unhealthy"

**Take screenshot when HealthyHostCount = 2**

**Minute 1-2: ASG Launches Replacement**

Watch for:
- [ ] ASG Activity shows "Launching 1 instances"
- [ ] New instance appears in EC2 Instances (Pending state)
- [ ] HealthyHostCount still = 2 (replacement not ready yet)

**Take screenshot showing new instance launching**

**Minute 2-3: Replacement Becomes Healthy**

Watch for:
- [ ] New instance transitions to "Running"
- [ ] New instance passes health checks
- [ ] HealthyHostCount rises: 2 → 3 (RECOVERY!)
- [ ] CPU stabilizes (stress ends at 120 seconds)

**Take screenshot when HealthyHostCount = 3**

**Minute 3+: Cleanup**

Final state:
- [ ] Old stressed instance terminates
- [ ] 3 new healthy instances in ASG
- [ ] RequestCount back to normal
- [ ] CPU normalized

**Take final "recovered" screenshot**

### Step 4: Document Exact Timeline (2 minutes)

Open a text editor and record:

```
FAILURE SIMULATION TIMELINE
===========================

Timestamp Format: YYYY-MM-DD HH:MM:SS UTC

Stress command started:     2026-01-05 16:30:00
CPU spike observed:         2026-01-05 16:30:15 (CPU: 90%+)
Health check failed:        2026-01-05 16:30:45 (HealthyHostCount: 3→2)
ASG launched replacement:   2026-01-05 16:31:00 ("Launching 1 instances")
New instance running:       2026-01-05 16:32:15 (state: Running)
New instance healthy:       2026-01-05 16:33:00 (HealthyHostCount: 2→3) ✓ RECOVERY
Old instance terminated:    2026-01-05 16:33:30
System recovered:           2026-01-05 16:34:00 (All metrics normal)

Total recovery time:        4 minutes
Critical path:              Health failure → Detection → Replacement → Healthy

What worked:
- Health checks detected failure within 45 seconds
- ASG launched replacement immediately
- No manual intervention required
- Load automatically rerouted to healthy instances

Observations:
- During 2-minute stress, instance CPU went from ~70% to 90%+
- Health check detected in 1 evaluation cycle (~30 seconds)
- ASG decision and launch: ~15 seconds
- New instance boot time: ~1 minute
- Total loss of capacity: ~2 minutes (3 → 2 instances)
```

[ ] Timeline document completed

---

## 📋 TASK 6: EVIDENCE COLLECTION (10-15 minutes)

### Step 1: Export SSM Command Log

```bash
# Get the command ID from earlier output, then:

COMMAND_ID="abc12345-1234-1234-1234-123456789012"  # Replace with yours
INSTANCE_ID="i-0abc1234def567890"  # Replace with yours

# Get command details:
aws ssm get-command-invocation \
  --command-id $COMMAND_ID \
  --instance-id $INSTANCE_ID \
  --region eu-north-1 \
  --output json > ssm_command_log.json

cat ssm_command_log.json
```

[ ] SSM command log exported
[ ] File: `ssm_command_log.json`

### Step 2: Export CloudTrail Events

```bash
# Get SSM SendCommand event:
aws cloudtrail lookup-events \
  --lookup-attributes AttributeKey=EventName,AttributeValue=SendCommand \
  --region eu-north-1 \
  --max-results 10 \
  --output json > cloudtrail_ssm_logs.json

# Get ASG activity via CloudTrail:
aws cloudtrail lookup-events \
  --lookup-attributes AttributeKey=EventName,AttributeValue=Launch \
  --region eu-north-1 \
  --max-results 10 \
  --output json > cloudtrail_launch_logs.json

cat cloudtrail_ssm_logs.json
cat cloudtrail_launch_logs.json
```

[ ] CloudTrail SSM events exported
[ ] CloudTrail Launch events exported
[ ] Files: `cloudtrail_ssm_logs.json`, `cloudtrail_launch_logs.json`

### Step 3: Export ASG Activity

```bash
# Get ASG activity history:
aws autoscaling describe-scaling-activities \
  --auto-scaling-group-name D7078E-MINI-PROJECT-GROUP35-ASG \
  --region eu-north-1 \
  --max-records 50 \
  --output json > asg_activity_log.json

# Pretty print it:
cat asg_activity_log.json | jq '.Activities[] | {StartTime, EndTime, Description, StatusCode}'
```

Or via Console (easier):

```
1. EC2 > Auto Scaling Groups > Your ASG
2. Activity tab
3. Right-click table > "Export"
4. Save as: asg_activity_log.csv
```

[ ] ASG Activity exported
[ ] File: `asg_activity_log.json` or `.csv`

### Step 4: Export CloudWatch Metrics

```bash
# CPU metrics:
aws cloudwatch get-metric-statistics \
  --namespace AWS/EC2 \
  --metric-name CPUUtilization \
  --dimensions Name=AutoScalingGroupName,Value=D7078E-MINI-PROJECT-GROUP35-ASG \
  --start-time 2026-01-05T16:20:00Z \
  --end-time 2026-01-05T16:40:00Z \
  --period 60 \
  --statistics Average,Maximum \
  --output json > cloudwatch_cpu.json

# HealthyHostCount:
aws cloudwatch get-metric-statistics \
  --namespace AWS/ApplicationELB \
  --metric-name HealthyHostCount \
  --dimensions Name=LoadBalancer,Value=app/D7078E-MINI-PROJECT-GROUP35-LB/xxxxx \
  --start-time 2026-01-05T16:20:00Z \
  --end-time 2026-01-05T16:40:00Z \
  --period 60 \
  --statistics Average \
  --output json > cloudwatch_healthy_hosts.json
```

Or via Console (easier):

```
1. CloudWatch > Dashboards > Your Dashboard
2. For each graph, click gear icon > "Export data"
3. Save CSV files
```

[ ] CloudWatch metrics exported
[ ] Files: `cloudwatch_cpu.json`, `cloudwatch_healthy_hosts.json`

### Step 5: Verify All Evidence Files

```bash
# Check what you have:
ls -la *.json *.csv 2>/dev/null | grep -E "(ssm|cloudtrail|asg|cloudwatch)"

# Expected files:
# - ssm_command_log.json
# - cloudtrail_ssm_logs.json
# - cloudtrail_launch_logs.json
# - asg_activity_log.json (or .csv)
# - cloudwatch_cpu.json
# - cloudwatch_healthy_hosts.json
```

[ ] All evidence files present and readable

### Step 6: Create Evidence Summary Document

```
EVIDENCE SUMMARY
================

Task 5: Failure Simulation
─────────────────────────

✓ Triggered: SSM stress-ng command on instance i-xxxxx
✓ Stress: CPU load 90% for 120 seconds
✓ Health check: Failed in ~45 seconds
✓ ASG action: Launched replacement instance
✓ Recovery: Complete in ~4 minutes
✓ Result: System resilient, no downtime

Task 6: Evidence Collection
────────────────────────────

Evidence Files:
1. ssm_command_log.json
   - Shows SSM command execution
   - Timestamp: 2026-01-05T16:30:00Z
   - Status: Success
   - CloudTrail reference: /path/to/ssm_command_log.json

2. cloudtrail_ssm_logs.json
   - AWS CloudTrail audit of SSM action
   - Timestamp: 2026-01-05T16:30:00Z
   - User: [your user]
   - EventSource: ssm.amazonaws.com

3. cloudtrail_launch_logs.json
   - CloudTrail audit of ASG launch action
   - Timestamp: 2026-01-05T16:31:00Z (ASG response)
   - Instance launched: i-yyyyy

4. asg_activity_log.json
   - Complete ASG activity history
   - Shows health check failure
   - Shows launch of replacement
   - Timestamps match observations

5. cloudwatch_cpu.json
   - CPU metrics before, during, after
   - Shows CPU spike to 90%
   - Shows recovery to normal

6. cloudwatch_healthy_hosts.json
   - HealthyHostCount metrics
   - Shows 3 → 2 transition (failure)
   - Shows 2 → 3 transition (recovery)
   - Timestamps match timeline

Key Findings:
- Automatic detection: ~45 seconds (health checks every 30 seconds)
- Automatic failover: ~15 seconds (ALB response)
- Automatic replacement: ~60 seconds (ASG launch)
- Total recovery: ~4 minutes
- System resilience: Successful (no service downtime)
- Audit trail: Complete (all actions logged to CloudTrail)
```

[ ] Evidence summary document created

---

## 📸 SCREENSHOT CHECKLIST

Collect these screenshots for your lab report:

```
BEFORE FAILURE:
[ ] Screenshot 1: EC2 showing 3 instances all "Running"
[ ] Screenshot 2: Target Group showing 3 "Healthy"
[ ] Screenshot 3: CloudWatch showing normal metrics

DURING FAILURE:
[ ] Screenshot 4: CloudWatch CPU spike on one instance
[ ] Screenshot 5: Target Group showing 1 "Unhealthy"
[ ] Screenshot 6: HealthyHostCount = 2 (failure detected)

DURING REPLACEMENT:
[ ] Screenshot 7: EC2 showing 4 instances (3 old + 1 new Pending)
[ ] Screenshot 8: ASG Activity showing "Launching 1 instances"
[ ] Screenshot 9: New instance in "Pending" state

AFTER RECOVERY:
[ ] Screenshot 10: New instance "Running" and "Healthy"
[ ] Screenshot 11: HealthyHostCount = 3 (recovery complete)
[ ] Screenshot 12: All 3 instances in Target Group as "Healthy"
[ ] Screenshot 13: CloudWatch metrics normalized
[ ] Screenshot 14: Old stressed instance terminated

Total: ~14-15 screenshots documenting complete failure/recovery cycle
```

[ ] All screenshots collected and saved

---

## 📝 FINAL DELIVERABLES CHECKLIST

### Screenshots (PNG files)
- [ ] Before failure (3 screenshots)
- [ ] During failure (3 screenshots)
- [ ] During replacement (3 screenshots)
- [ ] After recovery (5 screenshots)

### Log Files (JSON/CSV)
- [ ] `ssm_command_log.json`
- [ ] `cloudtrail_ssm_logs.json`
- [ ] `cloudtrail_launch_logs.json`
- [ ] `asg_activity_log.json` or `.csv`
- [ ] `cloudwatch_cpu.json`
- [ ] `cloudwatch_healthy_hosts.json`

### Documents
- [ ] Timeline with exact timestamps
- [ ] Evidence summary
- [ ] Analysis of failure/recovery
- [ ] Lessons learned

### Analysis Questions Answered
- [ ] How long did health check detection take?
- [ ] How long did ASG replacement take?
- [ ] What was the total recovery time?
- [ ] Were any requests lost during failover?
- [ ] How did load distribution work during failure?
- [ ] What proved the system is resilient?

---

## 🎓 WHAT YOU'VE DEMONSTRATED

By completing Tasks 5 & 6, you've shown:

✅ **Automatic Failure Detection**
   - Health checks work correctly
   - Failed instances identified within 45 seconds
   - No manual intervention needed

✅ **Automatic Instance Replacement**
   - ASG automatically launches replacement
   - New instance configured identically
   - Service continues on remaining instances

✅ **Transparent Failover**
   - ALB automatically reroutes traffic
   - Requests continue to healthy instances
   - System self-heals without downtime

✅ **Production-Grade Resilience**
   - Survives single instance failure
   - Automatic recovery in ~4 minutes
   - No data loss
   - Zero manual operations

✅ **Complete Audit Trail**
   - All actions logged to CloudTrail
   - SSM commands auditable
   - Full compliance traceability
   - Timestamps for every event

---

## 🚀 FINAL STEPS

1. **Verify all evidence files exist:**
   ```bash
   ls -la ssm_command_log.json cloudtrail_*.json asg_activity_log.* cloudwatch_*.json
   ```

2. **Review your timeline document** for accuracy

3. **Check all screenshots** are clear and properly labeled

4. **Create final analysis report** summarizing findings

5. **Collect into single folder:** `Task_5_6_Evidence/`

6. **Submit** to instructor with:
   - Screenshots (14+ images)
   - Evidence files (6+ JSON/CSV files)
   - Timeline document
   - Analysis report

---

## ⏱️ TIMING

Total time for Tasks 5 & 6:

- Task 5 preparation: 2 minutes
- Failure simulation: 2 minutes
- Monitor & observe: 7 minutes
- Timeline documentation: 2 minutes
- **Task 5 Subtotal: ~15-20 minutes**

- Log collection: 5 minutes
- Evidence export: 5 minutes
- Document creation: 5 minutes
- **Task 6 Subtotal: ~15 minutes**

**Total: ~30-35 minutes for both tasks**

---

## ✨ SUCCESS CRITERIA

You're done when:

- ✅ 3 instances scaled out and all healthy
- ✅ Failure simulated on one instance
- ✅ Health check detected failure
- ✅ ASG launched replacement
- ✅ System recovered to 3 healthy instances
- ✅ All evidence collected and documented
- ✅ Timeline complete with exact timestamps
- ✅ Screenshots show entire cycle
- ✅ CloudTrail proves all actions auditable
- ✅ Lab report prepared with all findings

---

## 📞 TROUBLESHOOTING

| Problem | Solution |
|---------|----------|
| SSM command fails with "InvalidTarget" | Verify instances in Fleet Manager show "Online" |
| Health check doesn't fail | Increase CPU stress (change -l 90 to -l 95) |
| ASG doesn't launch replacement | Check desired capacity is still 3, check ASG activity for errors |
| Instances don't show in SSM | Verify IAM role attached, wait 5 minutes, terminate and relaunch |
| CloudTrail events not showing | Wait 5 minutes for eventual consistency, check event names spelling |
| CloudWatch not showing metrics | Verify metric namespace and dimensions match, check time range |

---

**Ready? Let's go! 🚀**

