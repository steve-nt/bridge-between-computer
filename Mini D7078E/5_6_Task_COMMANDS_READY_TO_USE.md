# Task 5 & 6: Ready-to-Use Command Templates

Use these templates to execute Tasks 5 & 6. Copy, replace bracketed values, and paste.

---

## TASK 5: FAILURE SIMULATION COMMANDS

### Pre-Flight Check

```bash
# Verify 3 instances running and healthy
aws ec2 describe-instances \
  --filters "Name=tag:aws:autoscaling:groupName,Values=D7078E-MINI-PROJECT-GROUP35-ASG" \
  --region eu-north-1 \
  --query 'Reservations[*].Instances[*].[InstanceId,State.Name,PrivateIpAddress]' \
  --output table
```

Expected output: 3 instances with State=running

### Get Instance ID to Stress

```bash
# Option 1: Get first instance
INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:aws:autoscaling:groupName,Values=D7078E-MINI-PROJECT-GROUP35-ASG" \
  --region eu-north-1 \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text)

echo "Will stress instance: $INSTANCE_ID"

# Option 2: List all instances and pick one manually
aws ec2 describe-instances \
  --filters "Name=tag:aws:autoscaling:groupName,Values=D7078E-MINI-PROJECT-GROUP35-ASG" \
  --region eu-north-1 \
  --query 'Reservations[*].Instances[*].[InstanceId,PrivateIpAddress]' \
  --output table
```

### Method A: Trigger Failure via SSM Stress Command (RECOMMENDED)

```bash
# Step 1: Set variables
INSTANCE_ID="i-0f9087a79628fd056"  # REPLACE with actual instance ID
REGION="eu-north-1"

# Step 2: Verify instance is managed by SSM
aws ssm describe-instance-information \
  --instance-information-filter-list "key=InstanceIds,valueSet=$INSTANCE_ID" \
  --region $REGION \
  --output table

aws ssm describe-instance-information --instance-information-filter-list "key=InstanceIds, valueSet=i-0f9087a79628fd056" 
    --region eu-north-1 --output table
# Expected: Should show instance as "Online"

# Step 3: Send stress command
COMMAND_ID=$(aws ssm send-command \
  --instance-ids $INSTANCE_ID \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["sudo apt-get update -y || true","sudo apt-get install -y stress-ng || true","stress-ng -c 0 -l 90 -t 120s"]' \
  --region $REGION \
  --query 'Command.CommandId' \
  --output text)

echo "Command sent with ID: $COMMAND_ID"
echo "Timestamp: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

# Step 4: Monitor command execution (optional, real-time)
while true; do
  STATUS=$(aws ssm get-command-invocation \
    --command-id $COMMAND_ID \
    --instance-id $INSTANCE_ID \
    --region $REGION \
    --query 'Status' \
    --output text 2>/dev/null)
  
  echo "Command status: $STATUS"
  
  if [ "$STATUS" = "Success" ] || [ "$STATUS" = "Failed" ]; then
    break
  fi
  
  sleep 5
done
```

### Method B: Trigger Failure via AWS FIS (Alternative)

```bash
# Step 1: Create FIS experiment (one-time setup)
# Go to AWS Console > Fault Injection Simulator > Create Experiment
# OR use AWS CLI:

FIS_EXPERIMENT=$(aws fis create-experiment-template \
  --description "D7078E Task5 CPU Stress" \
  --targets '{"Instances":{"resourceType":"ec2:instance","resourceTags":{"aws:autoscaling:groupName":"D7078E-MINI-PROJECT-GROUP35-ASG"}}}' \
  --actions '{
    "CPUStress": {
      "actionId": "aws:ec2:cpu-stress",
      "description": "Inject CPU stress",
      "parameters": {
        "DurationSeconds": "120",
        "CpuLoad": "90"
      },
      "targets": {"Instances": "Instances"}
    }
  }' \
  --role-arn "arn:aws:iam::ACCOUNT_ID:role/AWS-FIS-Default-Role" \
  --region eu-north-1 \
  --output json)

TEMPLATE_ID=$(echo $FIS_EXPERIMENT | jq -r '.ExperimentTemplate.id')
echo "FIS template created: $TEMPLATE_ID"

# Step 2: Start the experiment
EXPERIMENT=$(aws fis start-experiment \
  --experiment-template-id $TEMPLATE_ID \
  --region eu-north-1 \
  --output json)

EXPERIMENT_ID=$(echo $EXPERIMENT | jq -r '.Experiment.id')
echo "FIS experiment started: $EXPERIMENT_ID"
echo "Timestamp: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

# Step 3: Monitor experiment status
while true; do
  EXPERIMENT_STATUS=$(aws fis get-experiment \
    --id $EXPERIMENT_ID \
    --region eu-north-1 \
    --query 'Experiment.state.status' \
    --output text)
  
  echo "Experiment status: $EXPERIMENT_STATUS"
  
  if [ "$EXPERIMENT_STATUS" = "completed" ] || [ "$EXPERIMENT_STATUS" = "failed" ]; then
    break
  fi
  
  sleep 5
done
```

### Monitor Failure in Real-Time

Open these commands in separate terminal windows:

```bash
# Window 1: Watch CloudWatch CPU every 10 seconds
watch -n 10 'aws cloudwatch get-metric-statistics \
  --namespace AWS/EC2 \
  --metric-name CPUUtilization \
  --dimensions Name=AutoScalingGroupName,Value=D7078E-MINI-PROJECT-GROUP35-ASG \
  --start-time $(date -u -d "5 minutes ago" +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 60 \
  --statistics Average,Maximum \
  --region eu-north-1 \
  --output table'

# Window 2: Watch HealthyHostCount every 10 seconds
watch -n 10 'aws cloudwatch get-metric-statistics \
  --namespace AWS/ApplicationELB \
  --metric-name HealthyHostCount \
  --dimensions Name=LoadBalancer,Value=app/D7078E-MINI-PROJECT-GROUP35-LB/xxxxx \
  --start-time $(date -u -d "5 minutes ago" +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 60 \
  --statistics Average \
  --region eu-north-1 \
  --output table'

# Window 3: Watch ASG activity every 5 seconds
watch -n 5 'aws autoscaling describe-scaling-activities \
  --auto-scaling-group-name D7078E-MINI-PROJECT-GROUP35-ASG \
  --region eu-north-1 \
  --max-records 20 \
  --output table'

# Window 4: Watch EC2 instances every 5 seconds
watch -n 5 'aws ec2 describe-instances \
  --filters "Name=tag:aws:autoscaling:groupName,Values=D7078E-MINI-PROJECT-GROUP35-ASG" \
  --region eu-north-1 \
  --query "Reservations[*].Instances[*].[InstanceId,State.Name,LaunchTime]" \
  --output table'
```

---

## TASK 6: EVIDENCE COLLECTION COMMANDS

### 1. Export SSM Command Log

```bash
# Use values from Task 5
COMMAND_ID="abc12345-1234-1234-1234-123456789012"  # From Task 5 output
INSTANCE_ID="i-0abc1234def567890"  # Target instance
REGION="eu-north-1"

# Get command details
aws ssm get-command-invocation \
  --command-id $COMMAND_ID \
  --instance-id $INSTANCE_ID \
  --region $REGION \
  --output json > ssm_command_log.json

# Display for verification
cat ssm_command_log.json | jq '.'

# Extract key fields
echo "=== SSM Command Summary ==="
cat ssm_command_log.json | jq '{
  CommandId: .CommandId,
  InstanceId: .InstanceId,
  Status: .Status,
  ExecutionStartDateTime: .ExecutionStartDateTime,
  ExecutionEndDateTime: .ExecutionEndDateTime,
  StandardOutputContent: .StandardOutputContent[0:200]
}'
```

### 2. Export CloudTrail SSM Events

```bash
REGION="eu-north-1"

# Get SSM SendCommand events
aws cloudtrail lookup-events \
  --lookup-attributes AttributeKey=EventName,AttributeValue=SendCommand \
  --region $REGION \
  --max-results 20 \
  --output json > cloudtrail_ssm_logs.json

echo "=== CloudTrail SSM Events ===" 
cat cloudtrail_ssm_logs.json | jq '.Events[] | {
  EventId: .EventId,
  EventName: .EventName,
  EventTime: .EventTime,
  Username: .Username,
  Resources: .Resources
}'
```

### 3. Export CloudTrail ASG/Launch Events

```bash
REGION="eu-north-1"

# Get Auto Scaling events
aws cloudtrail lookup-events \
  --lookup-attributes AttributeKey=EventSource,AttributeValue=autoscaling.amazonaws.com \
  --region $REGION \
  --max-results 50 \
  --output json > cloudtrail_asg_logs.json

echo "=== CloudTrail ASG Events ===" 
cat cloudtrail_asg_logs.json | jq '.Events[] | {
  EventName: .EventName,
  EventTime: .EventTime,
  Username: .Username
}'
```

### 4. Export CloudTrail Launch Events

```bash
REGION="eu-north-1"

# Get EC2 Launch events
aws cloudtrail lookup-events \
  --lookup-attributes AttributeKey=EventName,AttributeValue=RunInstances \
  --region $REGION \
  --max-results 50 \
  --output json > cloudtrail_launch_logs.json

echo "=== CloudTrail Launch Events ===" 
cat cloudtrail_launch_logs.json | jq '.Events[] | {
  EventName: .EventName,
  EventTime: .EventTime,
  Username: .Username,
  SourceIPAddress: .CloudTrailEvent | fromjson | .sourceIPAddress
}'
```

### 5. Export ASG Activity History

```bash
ASG_NAME="D7078E-MINI-PROJECT-GROUP35-ASG"
REGION="eu-north-1"

# Get ASG activity
aws autoscaling describe-scaling-activities \
  --auto-scaling-group-name $ASG_NAME \
  --region $REGION \
  --max-records 50 \
  --output json > asg_activity_log.json

# Display key activities
echo "=== ASG Activity Summary ===" 
cat asg_activity_log.json | jq '.Activities[] | {
  StartTime: .StartTime,
  EndTime: .EndTime,
  Description: .Description,
  StatusCode: .StatusCode,
  Cause: .Cause
}'

# Export as CSV
echo "Time,Activity,Status,Cause" > asg_activity_log.csv
cat asg_activity_log.json | jq -r '.Activities[] | [.StartTime, .Description, .StatusCode, .Cause] | @csv' >> asg_activity_log.csv
```

### 6. Export CloudWatch CPU Metrics

```bash
REGION="eu-north-1"
START_TIME="2026-01-05T16:20:00Z"  # REPLACE with actual start time
END_TIME="2026-01-05T16:40:00Z"    # REPLACE with actual end time

# CPU metrics for ASG
aws cloudwatch get-metric-statistics \
  --namespace AWS/EC2 \
  --metric-name CPUUtilization \
  --dimensions Name=AutoScalingGroupName,Value=D7078E-MINI-PROJECT-GROUP35-ASG \
  --start-time $START_TIME \
  --end-time $END_TIME \
  --period 60 \
  --statistics Average,Maximum,Minimum \
  --region $REGION \
  --output json > cloudwatch_cpu.json

# Display
echo "=== CPU Metrics ===" 
cat cloudwatch_cpu.json | jq '.Datapoints | sort_by(.Timestamp)[] | {
  Timestamp: .Timestamp,
  Average: .Average,
  Maximum: .Maximum,
  Minimum: .Minimum
}'

# Export as CSV
echo "Timestamp,Average,Maximum,Minimum" > cloudwatch_cpu.csv
cat cloudwatch_cpu.json | jq -r '.Datapoints | sort_by(.Timestamp)[] | [.Timestamp, .Average, .Maximum, .Minimum] | @csv' >> cloudwatch_cpu.csv
```

### 7. Export CloudWatch HealthyHostCount Metrics

```bash
REGION="eu-north-1"
START_TIME="2026-01-05T16:20:00Z"  # REPLACE with actual start time
END_TIME="2026-01-05T16:40:00Z"    # REPLACE with actual end time

# Get Load Balancer name (adjust based on your setup)
LB_ARN=$(aws elbv2 describe-load-balancers \
  --region $REGION \
  --query 'LoadBalancers[?contains(LoadBalancerName, `D7078E`)].LoadBalancerArn' \
  --output text | head -1)

LB_NAME=$(echo $LB_ARN | cut -d':' -f6)

# Get Target Group ARN
TG_ARN=$(aws elbv2 describe-target-groups \
  --region $REGION \
  --query 'TargetGroups[?contains(TargetGroupName, `D7078E`)].TargetGroupArn' \
  --output text | head -1)

TG_NAME=$(echo $TG_ARN | cut -d':' -f6)

# Get metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/ApplicationELB \
  --metric-name HealthyHostCount \
  --dimensions Name=LoadBalancer,Value=$LB_NAME Name=TargetGroup,Value=$TG_NAME \
  --start-time $START_TIME \
  --end-time $END_TIME \
  --period 60 \
  --statistics Average,Maximum,Minimum \
  --region $REGION \
  --output json > cloudwatch_healthy_hosts.json

# Display
echo "=== HealthyHostCount Metrics ===" 
cat cloudwatch_healthy_hosts.json | jq '.Datapoints | sort_by(.Timestamp)[] | {
  Timestamp: .Timestamp,
  HealthyHosts: .Average
}'

# Export as CSV
echo "Timestamp,HealthyHosts" > cloudwatch_healthy_hosts.csv
cat cloudwatch_healthy_hosts.json | jq -r '.Datapoints | sort_by(.Timestamp)[] | [.Timestamp, .Average] | @csv' >> cloudwatch_healthy_hosts.csv
```

### 8. Verify All Evidence Files

```bash
echo "=== Evidence File Inventory ===" 
ls -lh | grep -E "(ssm|cloudtrail|asg|cloudwatch)"

echo ""
echo "=== File Count ===" 
echo "SSM logs: $(ls -1 ssm_*.json 2>/dev/null | wc -l)"
echo "CloudTrail logs: $(ls -1 cloudtrail_*.json 2>/dev/null | wc -l)"
echo "ASG logs: $(ls -1 asg_*.json asg_*.csv 2>/dev/null | wc -l)"
echo "CloudWatch metrics: $(ls -1 cloudwatch_*.json cloudwatch_*.csv 2>/dev/null | wc -l)"

echo ""
echo "=== File Sizes ===" 
du -h ssm_*.json cloudtrail_*.json asg_*.* cloudwatch_*.* 2>/dev/null

echo ""
echo "=== All Files Ready for Lab Report ===" 
find . -maxdepth 1 \( -name "*.json" -o -name "*.csv" \) -type f -exec ls -1 {} \; | sort
```

### 9. Create Evidence Summary Report

```bash
# Create markdown summary
cat > EVIDENCE_SUMMARY.md << 'EOF'
# Task 5 & 6 Evidence Summary

## Failure Simulation Details

**Timestamp:** [INSERT START TIME HERE]

**Instance Stressed:** [INSERT INSTANCE ID HERE]

**Stress Command:**
- Tool: stress-ng
- CPU Load: 90%
- Duration: 120 seconds

**Method:** SSM send-command

## Evidence Files

### SSM Command Log
- File: `ssm_command_log.json`
- Size: $(ls -lh ssm_command_log.json | awk '{print $5}')
- Command ID: [SEE FILE]
- Status: Success
- Execution time: [SEE FILE]

### CloudTrail Events
- SSM Events: `cloudtrail_ssm_logs.json`
- ASG Events: `cloudtrail_asg_logs.json`
- Launch Events: `cloudtrail_launch_logs.json`
- Total events: [COUNT FROM FILES]

### ASG Activity
- File: `asg_activity_log.json` and `.csv`
- Records: [COUNT]
- Key event: Health check failure → Instance launch

### CloudWatch Metrics
- CPU metrics: `cloudwatch_cpu.json` and `.csv`
- HealthyHostCount: `cloudwatch_healthy_hosts.json` and `.csv`
- Time range: [START TIME] to [END TIME]
- Data points: [COUNT]

## Key Timeline

- Stress started: [INSERT EXACT TIME]
- CPU spike detected: [INSERT EXACT TIME]
- Health check failed: [INSERT EXACT TIME]
- ASG launched replacement: [INSERT EXACT TIME]
- Instance became healthy: [INSERT EXACT TIME]
- Total recovery time: [CALCULATE]

## Summary

✓ Failure successfully simulated
✓ Health check detected failure
✓ ASG automatically launched replacement
✓ System recovered without manual intervention
✓ All actions logged to CloudTrail
✓ Complete audit trail maintained

EOF

cat EVIDENCE_SUMMARY.md
```

---

## Quick Copy-Paste Workflow

### For Task 5:

```bash
# 1. Get instance ID
INSTANCE_ID=$(aws ec2 describe-instances --filters "Name=tag:aws:autoscaling:groupName,Values=D7078E-MINI-PROJECT-GROUP35-ASG" --region eu-north-1 --query 'Reservations[0].Instances[0].InstanceId' --output text)

# 2. Send stress command
COMMAND_ID=$(aws ssm send-command --instance-ids $INSTANCE_ID --document-name "AWS-RunShellScript" --parameters 'commands=["sudo apt-get update -y || true","sudo apt-get install -y stress-ng || true","stress-ng -c 0 -l 90 -t 120s"]' --region eu-north-1 --query 'Command.CommandId' --output text)

echo "Instance: $INSTANCE_ID"
echo "Command: $COMMAND_ID"
echo "Started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

# 3. Open CloudWatch dashboard and monitor for 5 minutes
```

### For Task 6:

```bash
# 1. Export all evidence
COMMAND_ID="[INSERT FROM TASK 5]"
INSTANCE_ID="[INSERT FROM TASK 5]"

# SSM logs
aws ssm get-command-invocation --command-id $COMMAND_ID --instance-id $INSTANCE_ID --region eu-north-1 --output json > ssm_command_log.json

# CloudTrail
aws cloudtrail lookup-events --lookup-attributes AttributeKey=EventName,AttributeValue=SendCommand --region eu-north-1 --output json > cloudtrail_ssm_logs.json
aws cloudtrail lookup-events --lookup-attributes AttributeKey=EventSource,AttributeValue=autoscaling.amazonaws.com --region eu-north-1 --output json > cloudtrail_asg_logs.json

# ASG activity
aws autoscaling describe-scaling-activities --auto-scaling-group-name D7078E-MINI-PROJECT-GROUP35-ASG --region eu-north-1 --output json > asg_activity_log.json

# CloudWatch metrics (adjust time range!)
aws cloudwatch get-metric-statistics --namespace AWS/EC2 --metric-name CPUUtilization --dimensions Name=AutoScalingGroupName,Value=D7078E-MINI-PROJECT-GROUP35-ASG --start-time 2026-01-05T16:20:00Z --end-time 2026-01-05T16:40:00Z --period 60 --statistics Average,Maximum --region eu-north-1 --output json > cloudwatch_cpu.json

# 2. Verify files
ls -lh *.json

# 3. Create summary
echo "Evidence collection complete!"
```

---

## Notes

- Replace all `[BRACKETED]` values with actual data from your environment
- Replace time ranges with actual failure simulation times
- Replace instance IDs, command IDs, and ARNs with your actual values
- Use `echo` and `jq` to pretty-print JSON for readability
- Export CSVs for easier analysis in spreadsheets
- Keep all JSON files for complete audit trail

