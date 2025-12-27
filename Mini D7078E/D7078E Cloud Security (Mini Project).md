D7078E Cloud Security (Mini Project)
Controlled Load Simulation — Auto Scaling & Failure

                        +-------------------------------+
                        |        WEB SERVER FARM        |
                        |  [EC2]  [EC2]  [EC2]  [EC2]   |
                        +-------------------------------+
                                      ^
                                      |
                                      |
                                 +----+----+
                                 | TARGET  |
                                 | (WWW)   |
                                 +----+----+
                                      ^
                 ^                    ^                    ^
                 |                    |                    |
        +--------+---------+  +-------+--------+  +--------+--------+
        |    ATTACKER 1    |  |   ATTACKER 2   |  |   ATTACKER 3    |
        |  (bot / client)  |  | (bot / client) |  | (bot / client)  |
        +------------------+  +----------------+  +-----------------+

                 ^                                             ^
                 |                                             |
        +--------+---------+                          +--------+---------+
        |    ATTACKER 4    |                          |    ATTACKER 5    |
        |  (bot / client)  |                          |  (bot / client)  |
        +------------------+                          +------------------+



Objective: In this mini-project, students build a web service on EC2 behind an
Application Load Balancer (ALB) and an Auto Scaling Group (ASG). Students
will run internal load-agents (containers) that generate configurable HTTP
request load to the ALB. The ASG will scale out when CPU utilization ≥ 80%
(scale-up up to max = 3). When the ASG reaches three instances, students will
trigger a safe failure simulation (using AWS Fault Injection Simulator or a local
CPU stress on instances) to observe health checks, replacement behaviour, and
failover. The objective is observation, measurement and producing
recommendations but not to attack any external system.
Safety & legal (must be read & signed before any test)
• Only run the lab in an AWS account you own.
• Do not target any resource outside the lab account and lab VPC. All traffic
generators must run inside the same VPC or from instances in the same
account/region.
• Instructor must pre-approve test window and resource limits (max
instances, bandwidth) in writing.
• Define abort criteria and stop actions (examples below). If any abort
criteria are reached, stop all load generators immediately and notify
instructor.
• Do not exceed agreed instance types / counts. Keep instance sizes small
(t2.micro/t3.micro/t3.small) unless instructor authorises bigger.
• Prefer AWS Fault Injection Simulator (FIS) to create controlled CPU faults
rather than network flooding. FIS is safer and designed for chaos
experiments.
Abort criteria (example) — stop load if any of:
• CloudWatch: ALB 5xx rate > 20% for 5 minutes
• CPU on any instance > 95% for 5 minutes continuously
• Network egress > instructor limit (e.g., 50 Mbps)
• Unexpected billing alerts or instructor requests stop
Tools & components students will use
• Python 3 load generator (safe, rate-limited).
• Docker + docker-compose for running multiple agents.
• ALB, Target Group, Auto Scaling Group, Launch Template (AMI), EC2.
• CloudWatch Metrics & Alarms, SNS (optional) for notifications.
• (Recommended) AWS Fault Injection Simulator (FIS) or stress-ng
launched by instructor to simulate CPU faults after ASG reaches 3
instances.
• CloudTrail (to audit operations).
• All CLI operations can be done with aws cli or Console
Tasks
1. Prepare a web app AMI: launch EC2, install a lightweight web app that
serves an HTML page and a CPU-consuming endpoint (burn that does
some CPU work). Create an AMI from it for ASG launch template.
2. Create an Application Load Balancer + Target Group (health check),
create Launch Template, create Auto Scaling Group with min=1
desired=1 max=3. Configure IAM role for instances.
3. Configure CloudWatch alarm & scaling policy to add instances when
average CPU ≥ 80%. Attach scaling policy to ASG.
4. Implement and run internal load-agents (Python containers) with
configurable RPS/agents/duration. Run them from the management host
inside lab. Observe scaling as load increases.
5. Wait until ASG reaches 3 instances. When it does, run the safe failure
simulation: either trigger a FIS experiment to increase CPU or run a
controlled stress command on instances via SSM to mimic failure.
Observe ALB health checks, instance replacement, failed requests and
CloudWatch metrics.
6. Tear down: stop generators, restore environment, collect logs/metrics,
and produce analysis.

                      Attack Architecture
                      ===================

                           [ATTACKER]
                               |
                               v
               +-----------------------------------+
               |   My Bots (Internal Load Agents)  |
               | [BOT] [BOT] [BOT] [BOT] [BOT]     |
               +-----------------------------------+
                    \      \      |      /      /
                     \      \     |     /      /
                      \      \    |    /      /
                       \      \   |   /      /
                        v      v  v  v      v

                     [ Application Load Balancer ]
                                   |
                          ===============================================
                          |                  |                          |
                          v                  v                          v
  [ EC2 instance (contents) ]   [ EC2 instance (contents) ]   [ EC2 instance (contents) ]
                           


                           Auto Scaling Group
                           Min=1 | Desired=1 | Max=3

               Safe Failure Simulation (FIS or SSM Stress)

Safe Python load generator: (Example)
This generator is rate-limited and configurable. It is intended only for lab/intra-
VPC testing. Do not use against public systems.
Check Canvas (File->Labs) section for agent.py file.
*Notes:
workers × rps = per-container RPS. Control these values.
Start with low values (e.g., workers=2, rps=1) and ramp up slowly.
The script logs basic status; students must capture CloudWatch metrics to
correlate.
Docker + docker-compose to run many agents (Example)
Run multiple agent containers from the management host so traffic originates
from inside the lab.
dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY agent.py .
RUN pip install aiohttp
ENTRYPOINT ["python", "agent.py"]
Find docker-compose.yml file in Canvas (File->Labs) and run this bash script in
your terminal
docker compose up --scale agent1=3
Start with scale=1 then increase it gradually
Auto Scaling & CloudWatch configuration
Launch Template: reference the AMI prepared with web app.
Target Group: configure ALB target group health check path.
ASG: min=1 desired=1 max=3, attached to target group; cross-AZ enabled.
CloudWatch Alarm (scale-out):
Metric: Average CPUUtilization for ASG or Target group

Threshold: >= 80 percent for 2 datapoints of 1 minute
Action: Scale out by +1 (or use step scaling)
Scale-in:
Threshold: <= 30 percent for 5 datapoints of 1 minute
Action: Scale in by -1
Verify autoscaling action via CloudWatch/ASG activity history.
You can use the AWS Console or CLI aws autoscaling put-scaling-policy + aws
cloudwatch put-metric-alarm.
Safe failure simulation (how to simulate “crash after 3 servers”)
Do not perform network floods. Instead, use one of these safe instructorcontrolled
methods:
Method A (preferred): AWS Fault Injection Simulator (FIS)
• Create an FIS experiment that increases CPU load on one or more instances
(uses SSM and CloudWatch). FIS runs are auditable and can be limited by
IAM. Use FIS to simulate instance degradation or kernel panic.
• Advantage: auditable, reversible, safe if limited by IAM policies and
timeouts.
Method B: Controlled CPU stress via SSM
• Using SSM aws ssm send-command to run stress-ng or a CPU burn on
target instance(s) for limited time. This approach is safer than sending
external traffic to crash instances, since it runs inside instance and is
reversible.
• Example (run only after approval):
aws ssm send-command --instance-ids <INSTANCE_ID> --document-name "AWSRunShellScript"
\
--parameters commands=["sudo apt-get update -y || true","sudo apt-get install -y stressng
|| true","stress-ng -c 0 -l 90 -t 120s"]
Deliverables:
1. Lab report (PDF) and a recorded screen video explaining each tasks.
2. Python agent code and docker-compose file used (with config values).
3. CloudWatch graphs: CPU, ALB RequestCount, ALB.
4. ASG activity log screenshot showing scale-out to 3 and any replacements.
5. FIS / SSM command logs used to simulate failure and CloudTrail evidence.
6. A timeline of events (ramp up → scale out → failure → recovery).
7. Reflections: what worked, what failed, potential production risks,
mitigation steps.