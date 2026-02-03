# D7078E Cloud Security Mini Project - Implementation Guide

## Project Overview
A hands-on lab demonstrating **Auto Scaling Groups (ASGs)**, **load balancing**, **cloud monitoring**, and **fault tolerance** in AWS using safe, controlled failure simulation.

---

## Key Deliverables Overview

### Task 5: **ASG Activity Log Screenshot (Scale-out to 3 + Replacements)**
### Task 6: **FIS/SSM Command Logs + CloudTrail Evidence**

These two tasks are your **primary deliverables** for demonstrating failure handling and recovery.

---

## TASK 5: Safe Failure Simulation - ASG Activity Log & Scaling Evidence

### Objective
Wait until ASG reaches 3 instances, then trigger a **safe failure simulation** to observe:
- ✅ Health check failure detection
- ✅ Automatic instance replacement 
- ✅ Load distribution during failure
- ✅ Automatic recovery (no manual intervention)

### Architecture During Failure
```
Before Failure:
  [Load Agents] → [ALB] → [Instance 1 ✓] [Instance 2 ✓] [Instance 3 ✓]
                                          ASG: Min=1, Desired=3, Max=3
                                          All healthy

During Failure:
  [Load Agents] → [ALB] → [Instance 1 ✓] [Instance 2 ✗ STRESSED] [Instance 3 ✓] [Replacement 🆕 Launching]
                          ├─ CPU spike to 90%
                          ├─ Health check fails
                          └─ ALB deregisters it

After Recovery:
  [Load Agents] → [ALB] → [Instance 1 ✓] [Instance 3 ✓] [Replacement ✓ Healthy]
                                         (Instance 2 terminates)
```

### Task 5 Implementation Steps

#### **Step 1: Ensure All 3 Instances Are Healthy**
Before triggering failure, verify:

```bash
# Check via AWS Console:
1. Go to: EC2 > Auto Scaling Groups > D7078E-MINI-PROJECT-GROUP35-ASG
2. Verify:
   - Running instances: 3
   - Desired capacity: 3
   - Health: All "Healthy" in Target Group

# Or via CLI:
aws autoscaling describe-auto-scaling-groups \
  --auto-scaling-group-names D7078E-MINI-PROJECT-GROUP35-ASG \
  --region eu-north-1 \
  --query 'AutoScalingGroups[0].[DesiredCapacity,MinSize,MaxSize,Instances[*].InstanceId]' \
  --output text
```

**Expected Output:**
```
3  1  3  i-1234abc  i-5678def  i-9012ghi
```

#### **Step 2: Choose Failure Simulation Method**

**OPTION A: AWS Fault Injection Simulator (FIS) - RECOMMENDED**
- ✅ Safest, designed for chaos testing
- ✅ Fully auditable (CloudTrail logs)
- ✅ Easy to control and stop
- ✅ Reversible

**OPTION B: SSM Stress Command - SIMPLER**
- ✅ No FIS setup needed
- ✅ Works immediately
- ✅ Clear cause-and-effect
- ✅ Time-limited (auto-stops after 120s)

**OPTION C: SSH + Manual Stress - NOT RECOMMENDED**
- ❌ Requires SSH key
- ❌ Less auditable
- ❌ Only use if other options fail

#### **Step 3: Set Up SSM (If Using FIS or SSM Commands)**

**Create IAM Role for SSM:**

```bash
# AWS Console:
1. IAM > Roles > Create role
2. Trusted entity: EC2
3. Permissions: AmazonSSMManagedInstanceCore
4. Name: D7078E-EC2-SSM-Role
5. Create

# Update Launch Template:
1. EC2 > Launch Templates > Your template
2. Click Actions > Create new version
3. Advanced details > IAM instance profile: D7078E-EC2-SSM-Role
4. Create launch template version

# Terminate old instances:
1. EC2 > Instances > Select all 3 instances
2. Instance State > Terminate instances
3. Wait 2-3 minutes for ASG to launch replacements

# Verify SSM connectivity:
1. Systems Manager > Fleet Manager
2. All 3 instances should show "Online" (green)
```

#### **Step 4: Trigger Failure Simulation**

**METHOD A: Using SSM Stress Command (RECOMMENDED)**

Pick one instance from your ASG and run:

```bash
# Get instance ID from EC2 console or:
aws ec2 describe-instances \
  --filters "Name=tag:aws:autoscaling:groupName,Values=D7078E-MINI-PROJECT-GROUP35-ASG" \
  --region eu-north-1 \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text

# Run stress command (replace INSTANCE_ID):
aws ssm send-command \
  --instance-ids i-0abc1234def567890 \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["sudo apt-get update -y || true","sudo apt-get install -y stress-ng || true","stress-ng -c 0 -l 90 -t 120s"]' \
  --region eu-north-1
```

**Expected Command Output:**
```json
{
  "Command": {
    "CommandId": "abc12345-1234-1234-1234-123456789012",
    "Status": "Pending",
    "InstanceIds": ["i-0abc1234def567890"]
  }
}
```

**METHOD B: Using AWS FIS (More Enterprise)**

```bash
# Create FIS experiment in Console:
1. Fault Injection Simulator > Experiments > New experiment
2. Name: D7078E-Task5-CPU-Stress
3. Service: EC2
4. Action: CPU (or stop instances)
5. Targets: Select your 3 instances
6. Duration: 120 seconds
7. IAM Role: Create or use existing
8. Create and start
```

#### **Step 5: Observe Failure & Recovery Timeline**

**MINUTE 0: Failure Begins**
```
Action: SSM stress command / FIS experiment starts
Event:  Stressed instance CPU → 90%+
Metrics: CPU spike visible in CloudWatch
```

**MINUTE 0-1: Health Check Fails**
```
Action: ALB health check fails on stressed instance
Event:  HealthyHostCount drops: 3 → 2
Status: Target Group shows instance as "Unhealthy"
ALB:    Stops sending traffic to failed instance
```

**MINUTE 1-2: ASG Launches Replacement**
```
Action: ASG detects unhealthy instance
Event:  ASG launches new instance to maintain desired=3
Status: New instance in "Pending" state
ASG Activity: "Launching 1 instances"
```

**MINUTE 2: Stress Command Ends**
```
Action: stress-ng times out (120 seconds)
Event:  Stressed instance CPU drops back to normal
Status: Instance still marked unhealthy (takes time to recover)
```

**MINUTE 2-3: Replacement Instance Boots**
```
Action: New instance boots and joins Target Group
Event:  New instance passes health checks
Status: HealthyHostCount: 2 → 3 (RECOVERY COMPLETE!)
ASG Activity: "Successfully launched 1 instances"
```

**MINUTE 3+: Cleanup**
```
Action: Old stressed instance terminates
Event:  ASG confirms 3 healthy instances
Status: System fully recovered with fresh instance
RequestCount: Returns to normal
CPU:    Normalizes across all 3 instances
```

### Task 5 Deliverables Checklist

```
☐ Screenshot 1: Before Failure
   └─ ASG Activity tab showing all 3 instances
   └─ CloudWatch showing CPU ~50-80%, HealthyHostCount=3
   └─ All 3 instances in Target Group as "Healthy"

☐ Screenshot 2: During Health Check Failure
   └─ ASG Activity showing health check failure detected
   └─ HealthyHostCount: 3 → 2 (deregistering failed instance)
   └─ Target Group showing one instance as "Unhealthy"
   └─ CPU spike on stressed instance visible in CloudWatch

☐ Screenshot 3: ASG Launching Replacement
   └─ ASG Activity: "Launching 1 instances" or "Launching 1 new instance"
   └─ EC2 Instances showing new instance in "Pending" state
   └─ HealthyHostCount still at 2 (replacement not ready yet)
   └─ Timestamp of launch event

☐ Screenshot 4: During Recovery
   └─ New instance showing in "Running" state
   └─ New instance passing health checks
   └─ HealthyHostCount rising from 2 → 3
   └─ CloudWatch showing RequestCount recovering

☐ Screenshot 5: After Recovery Complete
   └─ All 3 instances healthy in Target Group
   └─ HealthyHostCount = 3
   └─ Old stressed instance terminated
   └─ System metrics normalized

☐ Log Files:
   └─ SSM Command output (success/failure log)
   └─ CloudTrail logs showing SSM send-command action
   └─ ASG Activity export (CSV or screenshot)
   └─ CloudWatch metrics export (CPU, HealthyHostCount timeline)

☐ Timeline Document:
   └─ Exact timestamp: Stress started
   └─ Exact timestamp: Health check failed
   └─ Exact timestamp: ASG launched replacement
   └─ Exact timestamp: Replacement became healthy
   └─ Exact timestamp: Old instance terminated
   └─ TOTAL recovery time: ___ minutes
```

### Key Metrics to Capture

| Metric | Before | During | After | Notes |
|--------|--------|--------|-------|-------|
| CPU (%) | 60-80 | 90+ | 60-80 | Stressed instance spikes |
| HealthyHostCount | 3 | 2 → 3 | 3 | Recovery visible |
| RequestCount (RPS) | 500-800 | Drops | Recovers | Traffic rerouted |
| Target Group State | 3 Healthy | 2 Healthy, 1 Unhealthy | 3 Healthy | Automatic failover |
| Instance Count | 3 running | 3 running (1 stressed, 1 new) | 3 running | Seamless replacement |

---

## TASK 6: FIS/SSM Command Logs & CloudTrail Evidence

### Objective
Collect and document **evidence of failure simulation** through:
- ✅ SSM/FIS command execution logs
- ✅ CloudTrail audit trail
- ✅ ASG activity history
- ✅ CloudWatch metrics timeline

### Implementation Steps

#### **Step 1: Get SSM Command Log**

If using SSM stress command:

```bash
# Get command history:
aws ssm list-commands \
  --region eu-north-1 \
  --query 'Commands[0]' \
  --output json

# Get command output:
aws ssm get-command-invocation \
  --command-id <COMMAND_ID> \
  --instance-id i-0abc1234def567890 \
  --region eu-north-1

# Export full output:
aws ssm get-command-invocation \
  --command-id <COMMAND_ID> \
  --instance-id i-0abc1234def567890 \
  --region eu-north-1 \
  --output json > ssm_command_log.json
```

**Expected Output:**
```json
{
  "CommandId": "abc12345-1234-1234-1234-123456789012",
  "InstanceId": "i-0abc1234def567890",
  "DocumentName": "AWS-RunShellScript",
  "Status": "Success",
  "StandardOutputContent": "stress-ng: info:  [1] stress-ng --cpu 0 --load 90 ...",
  "ExecutionElapsedTime": "122",
  "ExecutionEndDateTime": "2026-01-05T16:35:00Z"
}
```

#### **Step 2: Get FIS Experiment Log**

If using AWS FIS:

```bash
# List FIS experiments:
aws fis list-experiments \
  --region eu-north-1 \
  --output json

# Get detailed experiment result:
aws fis get-experiment \
  --id EXPxxxxxxxxxxxxx \
  --region eu-north-1 \
  --output json > fis_experiment_log.json
```

**Expected Data:**
```json
{
  "experiment": {
    "id": "EXPxxxxxxxxxxxxx",
    "arn": "arn:aws:fis:eu-north-1:ACCOUNT:experiment/EXPxxxxxxxxxxxxx",
    "title": "D7078E-Task5-CPU-Stress",
    "state": {
      "status": "completed"
    },
    "startTime": "2026-01-05T16:30:00Z",
    "endTime": "2026-01-05T16:32:00Z",
    "targetAccountConfiguration": {
      "roleArn": "arn:aws:iam::ACCOUNT:role/AWS-FIS-Role"
    },
    "targets": {
      "Instances": {
        "resourceTags": {},
        "resourceIds": ["i-1234abc", "i-5678def", "i-9012ghi"]
      }
    }
  }
}
```

#### **Step 3: Get CloudTrail Evidence**

CloudTrail automatically logs all AWS API calls:

```bash
# Get CloudTrail events for SSM/FIS actions:
aws cloudtrail lookup-events \
  --lookup-attributes AttributeKey=EventName,AttributeValue=SendCommand \
  --region eu-north-1 \
  --max-results 50 \
  --output json > cloudtrail_ssm_logs.json

# Get CloudTrail events for FIS:
aws cloudtrail lookup-events \
  --lookup-attributes AttributeKey=EventName,AttributeValue=StartExperiment \
  --region eu-north-1 \
  --max-results 50 \
  --output json > cloudtrail_fis_logs.json

# Get ASG activity events:
aws cloudtrail lookup-events \
  --lookup-attributes AttributeKey=EventName,AttributeValue=LaunchInstances \
  --region eu-north-1 \
  --max-results 50 \
  --output json > cloudtrail_asg_logs.json
```

**Expected CloudTrail Entry:**
```json
{
  "EventId": "11111111-1111-1111-1111-111111111111",
  "EventName": "SendCommand",
  "EventTime": "2026-01-05T16:30:00Z",
  "Username": "user@example.com",
  "Resources": [
    {
      "ResourceType": "AWS::EC2::Instance",
      "ResourceName": "i-0abc1234def567890"
    }
  ],
  "EventSource": "ssm.amazonaws.com",
  "UserAgent": "aws-cli/2.x.x",
  "RequestParameters": {
    "instanceIds": ["i-0abc1234def567890"],
    "documentName": "AWS-RunShellScript",
    "parameters": {
      "commands": ["stress-ng -c 0 -l 90 -t 120s"]
    }
  },
  "ResponseElements": {
    "command": {
      "commandId": "abc12345-1234-1234-1234-123456789012"
    }
  }
}
```

#### **Step 4: Get ASG Activity Log**

```bash
# Export ASG activity:
aws autoscaling describe-scaling-activities \
  --auto-scaling-group-name D7078E-MINI-PROJECT-GROUP35-ASG \
  --region eu-north-1 \
  --max-records 50 \
  --output json > asg_activity_log.json

# Or via Console:
1. EC2 > Auto Scaling Groups > Your ASG
2. Activity tab
3. Right-click > Export to CSV
```

**Expected ASG Activity:**
```
Time                  | Activity
2026-01-05 16:30:15 | Health check failure detected on instance i-xxxxx
2026-01-05 16:30:30 | Removing instance from load balancer: i-xxxxx
2026-01-05 16:30:45 | Terminating instance: i-xxxxx
2026-01-05 16:31:00 | Launching 1 new instances
2026-01-05 16:32:45 | Registering new instance: i-yyyyy
2026-01-05 16:33:00 | Instance successfully joined target group
```

#### **Step 5: Get CloudWatch Metrics Timeline**

```bash
# Export CloudWatch metric data:
aws cloudwatch get-metric-statistics \
  --namespace AWS/EC2 \
  --metric-name CPUUtilization \
  --dimensions Name=AutoScalingGroupName,Value=D7078E-MINI-PROJECT-GROUP35-ASG \
  --start-time 2026-01-05T16:20:00Z \
  --end-time 2026-01-05T16:40:00Z \
  --period 60 \
  --statistics Average,Maximum \
  --output json > cloudwatch_cpu_metrics.json

# Export HealthyHostCount:
aws cloudwatch get-metric-statistics \
  --namespace AWS/ApplicationELB \
  --metric-name HealthyHostCount \
  --dimensions Name=LoadBalancer,Value=app/D7078E-MINI-PROJECT-GROUP35-LB/xxxxx Name=TargetGroup,Value=targetgroup/D7078E-MINI-PROJECT-GROUP35-TG/xxxxx \
  --start-time 2026-01-05T16:20:00Z \
  --end-time 2026-01-05T16:40:00Z \
  --period 60 \
  --statistics Average \
  --output json > cloudwatch_healthy_hosts.json
```

### Task 6 Deliverables Checklist

```
☐ SSM/FIS Logs:
   ☐ SSM Command output (if using SSM)
   ☐ FIS Experiment report (if using FIS)
   ☐ Command ID and execution timestamps
   ☐ Success/failure status

☐ CloudTrail Evidence:
   ☐ SendCommand event (SSM API call)
   ☐ StartExperiment event (FIS API call)
   ☐ LaunchInstances events (ASG launching replacement)
   ☐ TerminateInstances event (old instance cleanup)
   ☐ JSON exports from cloudtrail lookup-events

☐ ASG Activity Log:
   ☐ Health check failure detection timestamp
   ☐ Instance removal timestamp
   ☐ New instance launch timestamp
   ☐ Instance registration timestamp
   ☐ ASG Activity screenshot or CSV export

☐ CloudWatch Metrics:
   ☐ CPU utilization timeline (JSON or CSV)
   ☐ HealthyHostCount timeline
   ☐ RequestCount timeline
   ☐ Screenshots of metric graphs during failure/recovery

☐ Timeline Document:
   ☐ Complete event timeline with all timestamps
   ☐ Duration from failure start to full recovery
   ☐ Time breakdowns (detection, replacement, recovery)
```

---

## Complete Task 5-6 Workflow

### Phase 1: Preparation (5 minutes)
```
✓ Verify 3 instances healthy
✓ Load is running to generate traffic
✓ CloudWatch dashboard open
✓ Take screenshot of "before" state
```

### Phase 2: Trigger Failure (1 minute)
```
✓ Run SSM send-command OR FIS start-experiment
✓ Record exact timestamp
✓ Monitor CloudWatch for CPU spike
✓ Take screenshot of command output
```

### Phase 3: Observe Failure (1-2 minutes)
```
✓ Watch health check fail (HealthyHostCount: 3 → 2)
✓ Watch ALB deregister instance
✓ Take screenshot of Target Group changes
✓ Take screenshot of CloudWatch during spike
```

### Phase 4: Observe Replacement (2-3 minutes)
```
✓ Watch ASG launch new instance
✓ Watch new instance boot and register
✓ Watch HealthyHostCount recover: 2 → 3
✓ Take screenshots of each step
```

### Phase 5: Verify Recovery (1 minute)
```
✓ All 3 instances healthy
✓ RequestCount recovering
✓ CPU normalizing
✓ Take final "after" screenshot
```

### Phase 6: Collect Evidence (5-10 minutes)
```
✓ Export SSM command log
✓ Export CloudTrail events
✓ Export ASG activity
✓ Export CloudWatch metrics
✓ Save all JSON/CSV files
✓ Take final screenshots
```

---

## Critical Success Criteria

### For Task 5:
- ✅ ASG Activity screenshot showing scale-out to 3 instances
- ✅ ASG Activity screenshot showing instance replacement (health failure → launch replacement)
- ✅ CloudWatch metrics showing failure and recovery
- ✅ Exact timeline of events with timestamps
- ✅ Evidence that recovery was automatic (no manual intervention)

### For Task 6:
- ✅ SSM/FIS command execution logs
- ✅ CloudTrail audit trail of all operations
- ✅ ASG Activity log showing automatic replacement
- ✅ CloudWatch metrics documenting the failure and recovery
- ✅ Analysis showing system resilience and failover effectiveness

---

## Common Issues & Solutions

### Issue: "ASG didn't launch replacement"
**Solution:**
- Check ASG desired capacity is still 3
- Check ASG activity log for errors
- Verify Launch Template has correct settings
- Check EC2 limits haven't been reached

### Issue: "Instances not showing in SSM/Fleet Manager"
**Solution:**
- Verify IAM role attached to instances
- Check instances have security group allowing HTTPS (usually default)
- Wait 5 minutes after instance launch
- Terminate and let ASG relaunch

### Issue: "Health check not failing"
**Solution:**
- CPU stress not high enough (increase -l parameter)
- Health check timeout too long (may take 30+ seconds)
- Instance may be recovering during check
- Run longer stress test (increase -t parameter)

### Issue: "FIS shows 'InvalidTarget'"
**Solution:**
- Ensure instances are managed by SSM (Fleet Manager shows "Online")
- Ensure IAM role has SSM permissions
- Manually select instances instead of filtering by tags
- Restart SSM agent on instances

---

## What You'll Demonstrate

By completing Tasks 5-6, you prove:

✅ **Automatic Failure Detection**
   - Health checks detect when instance is unhealthy
   - ALB automatically removes failed instance from rotation
   - No manual intervention needed

✅ **Automatic Instance Replacement**
   - ASG automatically launches replacement when instance fails
   - New instance boots from same AMI
   - New instance joins Target Group automatically

✅ **Transparent Failover**
   - Traffic automatically rerouted to healthy instances
   - Some requests may fail briefly (during detection)
   - Most traffic continues uninterrupted

✅ **Resilience**
   - System recovers from single instance failure
   - Total recovery time: ~2-3 minutes
   - No data loss (new instance same as old)
   - Highly available (3 instances → continues at ~2/3 capacity)

✅ **Full Audit Trail**
   - CloudTrail logs all actions
   - SSM logs all commands
   - ASG logs all scaling decisions
   - Complete transparency for compliance

---

## Deliverables Summary

| Task | Deliverable | Type | Format |
|------|-------------|------|--------|
| Task 5 | ASG scaling to 3 instances | Screenshot | PNG |
| Task 5 | Health check failure & replacement | Screenshots | PNG (3-5 images) |
| Task 5 | CloudWatch metrics during failure | Screenshots | PNG (2-3 graphs) |
| Task 5 | Event timeline with timestamps | Document | TXT or PDF |
| Task 6 | SSM command log | Log file | JSON |
| Task 6 | CloudTrail events | Log file | JSON |
| Task 6 | ASG activity history | Log file | CSV or JSON |
| Task 6 | CloudWatch metrics data | Data file | CSV or JSON |
| Task 6 | Analysis report | Document | PDF |

---

## Next Steps

1. **Complete Task 4** (if not done): Run load generators to scale to 3 instances
2. **Execute Task 5**: Trigger failure simulation following timeline above
3. **Execute Task 6**: Collect all logs and evidence
4. **Create Lab Report**: Compile screenshots, logs, and analysis
5. **Submit**: Include all deliverables from tasks 1-6

---

## Additional Resources

- Task 5 Detailed Guide: `TASK_5_FAILURE_SIMULATION.txt`
- SSM Setup Guide: `SSM_SETUP_DETAILED.txt`
- CloudTrail Evidence: AWS Console > CloudTrail > Event history
- FIS Documentation: https://docs.aws.amazon.com/fis/
- ASG Documentation: https://docs.aws.amazon.com/autoscaling/

