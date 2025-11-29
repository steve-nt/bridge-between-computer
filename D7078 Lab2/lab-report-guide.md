# D7078E Lab 2: Lab Report Writing Guide

## Overview

This guide provides detailed instructions on what to include in each section of your lab report. Use this to understand the depth and type of analysis expected for each task.

---

## Section 1: Introduction (1-2 pages)

### What to Write

**Objective Statement:**
- Clearly state the purpose of Lab 2
- Mention both Task 2.1 and Task 2.2
- Explain how cloud security and storage are related

Example:
```
"This lab aims to develop practical skills in cloud security and storage services 
on Amazon Web Services. Task 2.1 focuses on implementing least-privilege access 
control through security groups and IAM roles, while Task 2.2 develops cloud 
storage applications using AWS SDK for Java."
```

**Motivation:**
- Why is cloud security important?
- Why is understanding S3 operations critical for cloud developers?
- Industry relevance

**Lab Structure:**
- Overview of two main tasks
- What you will learn and implement

### Key Points to Mention
- ✓ Principle of least privilege
- ✓ Defense in depth (security layers)
- ✓ Cloud scalability and accessibility

---

## Section 2: Task 2.1 - AWS CLI Security Configuration (3-4 pages)

### 2.1.1 Objective and Methodology

**What to Write:**
- Clear statement of Task 2.1 objectives
- Methodology: step-by-step approach
- Tools used: AWS CLI, bash scripting
- Architecture decisions made

Example:
```
"Task 2.1 implements AWS infrastructure with three security layers:
1. Network layer (Security Groups with inbound rules)
2. Identity layer (IAM roles and policies)
3. Instance layer (EC2 with attached role)

This multi-layered approach exemplifies the principle of defense in depth."
```

### 2.2.2 Implementation Details

#### Subsection A: Security Group Configuration

**What to Include:**

1. **Architecture Diagram**
   - Draw/insert diagram showing:
     * VPC and security group
     * Inbound rules (SSH restricted, HTTP open)
     * Outbound rules (if configured)
     * EC2 instance within security group

```
Example ASCII representation:
┌─────────────────────────────────────────┐
│           Default VPC                   │
│  ┌──────────────────────────────────┐   │
│  │   Security Group                 │   │
│  │                                  │   │
│  │  Inbound Rules:                  │   │
│  │  • SSH (22) from 203.0.113.25/32│   │
│  │  • HTTP (80) from 0.0.0.0/0      │   │
│  │  • ICMP (ping) from 0.0.0.0/0    │   │
│  │                                  │   │
│  │  ┌──────────────────────────┐    │   │
│  │  │   EC2 Instance           │    │   │
│  │  │   AMI: Amazon Linux 2    │    │   │
│  │  │   Role: Lab2S3AccessRole │    │   │
│  │  └──────────────────────────┘    │   │
│  └──────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

2. **Code/Commands Executed**
   - Show the actual commands used
   - Explain what each command does
   - Include output snippets

```bash
# Create security group
aws ec2 create-security-group \
    --group-name lab2-security-group \
    --description "Lab 2 Security Group" \
    --vpc-id vpc-xxxxxx

# Result: Created SG with ID: sg-0abc123xyz
```

3. **Analysis**
   - Why SSH restricted to single IP?
     * Reduces attack surface
     * Only authorized access
     * Follows least privilege principle
   
   - Why HTTP open to 0.0.0.0/0?
     * Web services need public accessibility
     * Can restrict further based on needs

#### Subsection B: IAM Policy and Role Configuration

**What to Include:**

1. **IAM Policy Structure**
   - Present the complete policy JSON
   - Explain each statement/action

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "S3FullAccess",
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",      // Read objects
                "s3:PutObject",      // Upload objects
                "s3:DeleteObject",   // Delete objects
                "s3:ListBucket"      // List contents
            ],
            "Resource": [
                "arn:aws:s3:::lab2-bucket-*",
                "arn:aws:s3:::lab2-bucket-*/*"
            ]
        }
    ]
}
```

2. **Least Privilege Justification**
   - What does this policy allow?
   - What does it deny?
   - Why is this appropriate?

```
Example Analysis:
"The policy grants S3 read/write/delete operations on lab2 buckets only.
It DOES NOT grant:
- S3 bucket deletion rights
- Cross-bucket access
- IAM modification permissions
- EC2 termination permissions

This follows least privilege by:
1. Limiting to specific resource (lab2 buckets)
2. Only necessary actions (CRUD operations)
3. No administrative privileges"
```

3. **Role Attachment**
   - Show how role was attached to instance profile
   - Trust relationship explanation

#### Subsection C: EC2 Instance Launch

**What to Include:**

1. **Instance Details**
```
Instance Configuration:
- Instance Type: t2.micro
- AMI: Amazon Linux 2 (ami-0c02fb55a...)
- IAM Role: Lab2S3AccessRole
- Security Group: lab2-security-group
- Key Pair: lab2-keypair
- Public IP: 203.0.113.100
- Availability Zone: us-east-1a
```

2. **Instance Profile Role**
   - How instance profile connects role to instance
   - How credentials are obtained by EC2 instance
   - Difference from hardcoded credentials

```
Architecture:
EC2 Instance 
    ↓ (requests credentials from)
Instance Metadata Service 
    ↓ (retrieves credentials for)
IAM Role (Lab2S3AccessRole) 
    ↓ (has attached)
IAM Policy (Lab2S3Policy)
    ↓ (grants permissions for)
S3 Resources (lab2-bucket-*)
```

### 2.1.3 Validation and Testing

**What to Write:**

1. **SSH Connection Test**
```bash
# Command executed:
ssh -i lab2-keypair.pem ec2-user@203.0.113.100

# Result: Successfully connected
# Indicates: Security group SSH rule and key pair are correct
```

2. **IAM Role Validation**
```bash
# Test 1: Verify role is available
curl http://169.254.169.254/latest/meta-data/iam/security-credentials/
# Result: Lab2S3AccessRole

# Test 2: Test S3 access (IAM role permissions)
aws s3 ls
# Result: Successfully listed buckets
# Indicates: EC2 instance has S3 access through IAM role
```

3. **S3 Operations Test**
```bash
# Upload test
echo "test content" > test.txt
aws s3 cp test.txt s3://lab2-bucket-xyz/
# Result: upload: ./test.txt to s3://lab2-bucket-xyz/test.txt

# Download test
aws s3 cp s3://lab2-bucket-xyz/test.txt downloaded-test.txt
# Result: download: s3://lab2-bucket-xyz/test.txt to ./downloaded-test.txt

# Verify content
cat downloaded-test.txt
# Result: test content
```

### 2.1.4 Reflection and Analysis Questions

**Question 1: Why is it more secure to use IAM roles instead of embedding AWS access keys?**

Write 2-3 paragraphs explaining:

```
Answer Template:

"IAM roles provide several security advantages over embedded credentials:

TEMPORARY CREDENTIALS:
IAM roles issue temporary security credentials that automatically expire.
Unlike long-term access keys embedded in code, these temporary credentials 
have a lifespan (typically 1 hour), reducing the window of exposure if 
compromised. AWS automatically rotates these credentials.

REDUCED EXPOSURE RISK:
When credentials are embedded in code (hardcoded, in config files, or in 
EC2 user data), they become part of the artifact. If source code is 
compromised, leaked, or backed up improperly, credentials are also exposed.
IAM roles eliminate this risk as credentials are never stored in the code.

NO CREDENTIAL MANAGEMENT:
Developers don't need to manage, rotate, or revoke individual credentials.
This is handled automatically by AWS. It reduces operational overhead and 
human error.

AUDIT AND COMPLIANCE:
All API calls made using an IAM role include the role identity, enabling
fine-grained audit logging through CloudTrail. This provides better 
security visibility and compliance reporting.

EXAMPLE:
In Task 2.1, the EC2 instance accessed S3 without any credentials 
provided at launch. The instance automatically obtained temporary 
credentials from the instance metadata service. If these were compromised,
they would expire within 1 hour."
```

**Question 2: What principle of cloud security is applied when restricting SSH access to a single IP?**

```
Answer:
This demonstrates the "Principle of Least Privilege" and "Defense in Depth."

LEAST PRIVILEGE:
Only grant the minimum permissions necessary. By restricting SSH to a 
single IP, we grant access only to the authorized user, not to the world.

REDUCED ATTACK SURFACE:
Out of 4 billion possible IPv4 addresses, only 1 can connect. This 
dramatically reduces the attack surface and potential entry points.

PRACTICAL IMPACT:
- Prevents unauthorized users from attempting SSH attacks
- Port scanners on the internet cannot connect
- Only your designated location has SSH access
- Even if someone discovers the instance IP, they cannot SSH to it

IMPLEMENTATION:
aws ec2 authorize-security-group-ingress \
    --group-id sg-xxx \
    --protocol tcp \
    --port 22 \
    --cidr 203.0.113.25/32  # /32 = single IP
```

**Question 3: What happens if you remove the S3 actions from the IAM policy and try the upload again?**

```
Answer:
If S3 actions (GetObject, PutObject, DeleteObject, ListBucket) are removed
from the policy, the following occurs:

ERROR RESPONSE:
$ aws s3 cp test.txt s3://lab2-bucket-xyz/
An error occurred (AccessDenied) when calling the PutObject operation: 
User: arn:aws:iam::123456789012:role/Lab2S3AccessRole is not authorized 
to perform: s3:PutObject on resource: arn:aws:s3:::lab2-bucket-xyz/test.txt

ROOT CAUSE:
The IAM role no longer has the s3:PutObject action, so AWS denies the 
request at the authorization layer, before it reaches the S3 service.

WHAT STILL WORKS:
- EC2 DescribeInstances (still in policy)
- SSH connection (handled by security group)
- Network access (security group permits it)

WHAT FAILS:
- Any S3 operation (GetObject, PutObject, DeleteObject, ListBucket)

DEBUGGING:
- Check CloudTrail logs for "AccessDenied" events
- Verify IAM policy attached to role
- Use AWS Policy Simulator to test permissions
```

**Question 4: How would you modify the Security Group to restrict HTTP access to an internal network only?**

```
Answer:
To restrict HTTP to internal network (private subnet), modify the ingress rule:

CURRENT CONFIGURATION:
HTTP (80) from 0.0.0.0/0  # Open to entire internet

RESTRICTED CONFIGURATION:
HTTP (80) from 10.0.0.0/8  # Open to private network only

COMMANDS:
# Step 1: Remove existing HTTP rule
aws ec2 revoke-security-group-ingress \
    --group-id sg-xxx \
    --protocol tcp \
    --port 80 \
    --cidr 0.0.0.0/0

# Step 2: Add new rule with internal CIDR
aws ec2 authorize-security-group-ingress \
    --group-id sg-xxx \
    --protocol tcp \
    --port 80 \
    --cidr 10.0.0.0/8

IMPLICATIONS:
- Public users cannot access HTTP service
- Only users within 10.0.0.0/8 network can access
- Useful for internal services or behind a load balancer
- Load balancer would be in public subnet, EC2 in private subnet
- Users connect to load balancer (public), which forwards to EC2 (private)
```

**Question 5: What challenges did you face when designing least-privilege policies?**

```
Answer:
Several challenges arise when implementing least privilege:

GRANULARITY CHALLENGE:
AWS actions are very granular. For example:
- s3:GetObject (read one object)
- s3:ListBucket (list objects)
- s3:GetBucketLocation (read bucket metadata)
Determining which actions are truly necessary requires deep understanding 
of the application.

RESOURCE ARN FORMAT VARIATIONS:
Different services have different ARN formats:
- S3 bucket: arn:aws:s3:::bucket-name
- S3 object: arn:aws:s3:::bucket-name/object-key
- EC2 instance: arn:aws:ec2:region:account:instance/instance-id
Mistakes in ARN syntax silently fail, denying access.

OVER-PERMISSIONING:
It's easier to grant broad permissions (S3FullAccess) than find the exact
set needed. Requires testing each API call:
1. Test with minimal policy (fails)
2. Add permissions incrementally
3. Verify all operations work
4. Document which actions are essential

WILDCARD MISUSE:
Using "s3:*" or "arn:aws:s3:::*" is too broad. Should use:
- Specific bucket: arn:aws:s3:::lab2-bucket-*
- Specific object path: arn:aws:s3:::lab2-bucket-*/uploads/*

EVOLUTION AND MAINTENANCE:
As application grows, required permissions change. Policies must be reviewed
and updated. Automation via Infrastructure as Code (IaC) helps.

SOLUTION APPROACH:
1. Start with nothing (empty policy)
2. Monitor CloudTrail for "AccessDenied" events
3. Add only the denied actions
4. Test thoroughly
5. Document and version policy
```

---

## Section 3: Task 2.2 - AWS SDK with Java (4-5 pages)

### 3.1 Environment Setup and SDK Architecture

**What to Write:**

1. **SDK Components and Dependencies**

Create a dependency diagram showing:
```
Application Code (S3BucketOperations.java)
        ↓
S3Client (Interface)
        ↓
S3ClientBuilder
├── Region Configuration
│   └── software.amazon.awssdk.regions.Region
├── Credentials Provider
│   ├── software.amazon.awssdk.auth.DefaultCredentialsProvider
│   ├── software.amazon.awssdk.auth.StaticCredentialsProvider
│   └── software.amazon.awssdk.auth.AwsCredentials
├── HTTP Transport Layer
│   └── Apache HttpClient (default)
├── Retry Strategy
│   └── software.amazon.awssdk.core.retry.RetryPolicy
└── Request/Response Models
    └── software.amazon.awssdk.services.s3.model.*
        ├── CreateBucketRequest
        ├── ListBucketsResponse
        ├── PutObjectRequest
        ├── GetObjectRequest
        └── ... (70+ model classes)
        ↓
AWS S3 API Endpoints (AWS Servers)
```

2. **How S3Client is Created**

```java
// Default creation with AWS CLI credentials
try (S3Client s3Client = S3Client.builder()
        .region(Region.US_EAST_1)
        .build()) {
    // S3Client automatically:
    // 1. Loads credentials from ~/.aws/credentials (DefaultCredentialsProvider)
    // 2. Creates HTTP client (Apache HttpClient)
    // 3. Configures retry policy (3 retries with exponential backoff)
    // 4. Sets endpoint to https://s3.us-east-1.amazonaws.com
}
```

Explain each step:
- Region selection determines endpoint URL
- Credentials provider chain (environment → config → instance role → STS)
- HTTP client handles low-level networking
- Retry strategy handles transient failures

3. **Package Details**

```
software.amazon.awssdk.services.s3:
- S3Client: Main service interface
- S3ClientBuilder: Fluent builder for configuration
- Request classes: Implement parameter passing
- Response classes: Encapsulate operation results
- Exception classes: Service-specific errors

software.amazon.awssdk.regions:
- Region: Enum of all AWS regions
- RegionMetadata: Region-specific information

software.amazon.awssdk.auth:
- Credentials providers: How credentials are obtained
- Signature calculators: AWS Signature v4 algorithm

software.amazon.awssdk.core:
- ResponseTransformer: Transforms response to desired format
- SdkClient: Base interface for all service clients
```

### 3.2 S3 Operations Implementation

**What to Write:**

1. **Task 1: Creating Buckets in Multiple Regions**

```
Implementation Details:

REGION SELECTION:
Chose three geographically distributed regions:
- us-east-1 (N. Virginia) - Default AWS region, lowest cost
- eu-west-1 (Ireland) - Serves European users
- ap-southeast-1 (Singapore) - Serves Asian-Pacific users

CODE PATTERN:
for (String regionName : REGIONS) {
    S3Client s3Client = S3Client.builder()
            .region(Region.of(regionName))
            .build();
    
    CreateBucketRequest request = CreateBucketRequest.builder()
            .bucket(bucketName)
            .build();
    
    s3Client.createBucket(request);
}

KEY POINTS:
- Each region requires separate S3Client instance
- CreateBucketConfiguration needed for non-us-east-1 regions
- Bucket names must be globally unique
- Regional isolation for compliance (data residency)
```

2. **Task 2-6: CRUD Operations**

For each operation (List, Upload, Download, Delete), write:

**Purpose:**
- Why is this operation important?
- Real-world use cases

**Implementation:**
- Show code snippet
- Explain parameters
- Error handling

**Example for Upload:**
```
PURPOSE:
PutObject uploads an object to S3. Essential for:
- Storing user files
- Backing up data
- Archival
- Data lakes

CODE PATTERN:
PutObjectRequest request = PutObjectRequest.builder()
        .bucket(bucketName)
        .key(objectKey)
        .build();

s3Client.putObject(request, 
    RequestBody.fromFile(new File(filepath)));

PARAMETERS:
- bucket: Destination bucket name
- key: Object path within bucket (e.g., "uploads/file.txt")
- RequestBody: Object content source (file, string, stream, bytes)

ERROR HANDLING:
try {
    s3Client.putObject(request, body);
} catch (S3Exception e) {
    System.err.println("S3 Error: " + 
        e.awsErrorDetails().errorMessage());
}
```

### 3.3 Latency Measurement and Analysis

**What to Write:**

1. **Methodology**

```
APPROACH:
1. Created 1 MB test files
2. Uploaded to 3 regions (3 iterations each)
3. Downloaded from 3 regions (3 iterations each)
4. Measured round-trip latency (start time to completion)
5. Calculated average latency per region

METRICS:
- Upload latency: Time from PutObject start to completion
- Download latency: Time from GetObject start to completion
- Latency variation: Standard deviation across iterations

SAMPLE SIZE:
- 3 iterations per operation per region
- Total: 18 measurements (9 upload + 9 download)
- Sufficient for trend analysis

LIMITATIONS:
- Does not account for object size variations
- AWS may cache objects (subsequent requests faster)
- Network conditions vary with time
- Single-threaded measurement (not representative of concurrent load)
```

2. **Results Table and Chart**

```
Results Summary:

┌────────────────┬──────────────────┬──────────────────┬────────────────┐
│ Region         │ Avg Upload (ms)  │ Avg Download(ms) │ Avg Total (ms) │
├────────────────┼──────────────────┼──────────────────┼────────────────┤
│ us-east-1      │       245        │       195        │       440      │
│ eu-west-1      │       456        │       412        │       868      │
│ ap-southeast-1 │       523        │       478        │      1001      │
└────────────────┴──────────────────┴──────────────────┴────────────────┘

OBSERVATIONS:
1. us-east-1 is fastest (N. Virginia has most AWS infrastructure)
2. eu-west-1 is 2x slower than us-east-1
3. ap-southeast-1 is 2.3x slower than us-east-1
4. Pattern: Latency increases with geographic distance

EXPLANATION:
- Geographic distance: More miles = longer transmission time
- Network path: More intermediate hops (routers, switches)
- AWS infrastructure density: us-east-1 has more capacity
- Internet backbone: Atlantic faster than routing to Asia
```

3. **Analysis and Insights**

```
KEY FINDINGS:

GEOGRAPHIC IMPACT:
- Latency correlates strongly with distance
- us-east-1 to ap-southeast-1: ~8,800 miles
- Network speed: ~3 × 10^8 m/s (fiber optic)
- Theoretical latency: 30ms just for light travel
- Actual latency 1000ms includes: routing, processing, queuing

IMPLICATIONS FOR APPLICATION DESIGN:

1. REGION SELECTION:
   - Choose region closest to primary users
   - Consider compliance/data residency requirements
   - Balance cost (us-east-1 cheaper) vs. latency

2. MULTI-REGION STRATEGY:
   - Mirror data to nearby regions for read latency
   - Accept higher latency for write consistency
   - Use S3 replication for automatic synchronization

3. CACHING LAYERS:
   - CloudFront (CDN) can reduce latency by 50-80%
   - Cache frequently accessed objects
   - Reduces S3 API calls

4. APPLICATION OPTIMIZATION:
   - Pre-fetch objects expected to be needed
   - Parallel uploads/downloads for throughput
   - Batch operations when possible
```

---

## Section 4: Comparison and Analysis (1-2 pages)

### 4.1 AWS CLI vs. AWS SDK

**What to Write:**

```
COMPARISON TABLE:

┌────────────────────┬──────────────────────┬──────────────────────┐
│ Feature            │ AWS CLI              │ AWS SDK (Java)       │
├────────────────────┼──────────────────────┼──────────────────────┤
│ Language           │ Bash/Shell           │ Compiled (Java)      │
│ Use Case           │ Ad-hoc operations    │ Embedded in apps     │
│ Performance        │ Slower (subprocess)  │ Faster (in-process)  │
│ Ease of Use        │ Simple commands      │ More code needed     │
│ Automation         │ Scripts              │ Applications         │
│ Error Handling     │ Exit codes/output    │ Exceptions/objects   │
│ Credentials        │ CLI config file      │ Provider chain       │
│ Learning Curve     │ Shallow              │ Steeper              │
└────────────────────┴──────────────────────┴──────────────────────┘

DETAILED ANALYSIS:

AWS CLI (Task 2.1):
✓ Advantages:
  - Quick to prototype infrastructure
  - Easy to learn and remember commands
  - Perfect for one-time setup tasks
  - Can be scheduled via cron jobs
  - No compilation needed
  
✗ Disadvantages:
  - Slower than SDK (spawns new process)
  - Limited error handling capabilities
  - Difficult for complex conditional logic
  - Not suitable for real-time applications
  - Requires shell knowledge

AWS SDK (Task 2.2):
✓ Advantages:
  - Fast in-process execution
  - Rich error handling and exceptions
  - Complex business logic easily implemented
  - Integrates into production applications
  - Type-safe method parameters
  
✗ Disadvantages:
  - More setup required
  - More code to write
  - Steeper learning curve
  - Requires understanding of SDK architecture

WHEN TO USE EACH:
- Use CLI: Infrastructure setup, ad-hoc operations, DevOps scripts
- Use SDK: Production applications, embedded services, APIs
```

### 4.2 Security Implications

```
SECURITY COMPARISON:

AWS CLI (Task 2.1):
- Credentials stored in ~/.aws/credentials (plaintext)
- Risk: If server compromised, attacker gets access keys
- Mitigation: Use IAM roles on EC2 instead of credentials file
- Authentication: Long-lived access keys
- Audit: CloudTrail logs CLI operations

AWS SDK in EC2 (Task 2.2):
- Credentials obtained from instance metadata service
- Instance has IAM role, not credentials
- Risk: Metadata service has IMDS token protection (optional)
- Mitigation: Enable IMDSv2 (requires token for metadata access)
- Authentication: Temporary credentials (rotate automatically)
- Audit: CloudTrail shows role identity, not specific credentials

BEST PRACTICES IMPLEMENTED:
1. Task 2.1: Used least-privilege IAM policy
2. Task 2.1: Restricted SSH to single IP
3. Task 2.1: Avoided embedding credentials
4. Task 2.2: Leveraged instance role in Java code
5. Task 2.2: No hardcoded credentials in source code
```

---

## Section 5: Challenges and Solutions (1 page)

### What to Write

Document any issues encountered and how you resolved them:

```
CHALLENGE 1: AWS Credentials Configuration
Problem: "Unable to locate credentials"
Root Cause: aws configure not run properly
Solution: 
1. Ran: aws configure
2. Verified: cat ~/.aws/credentials
3. Confirmed: aws sts get-caller-identity
4. Result: Credentials properly configured

CHALLENGE 2: Security Group Rules
Problem: Cannot SSH to EC2 instance
Root Cause: SSH rule allowed only /24 CIDR, but only /32 available
Solution:
1. Identified: IP might change
2. Modified: Used public IP with /32
3. Tested: Successfully SSHed
4. Learning: Security groups are evaluated at connection time

CHALLENGE 3: IAM Policy Testing
Problem: S3 upload fails with AccessDenied
Root Cause: Forgot to include s3:PutObject action
Solution:
1. Checked: CloudTrail logs
2. Found: Missing s3:PutObject
3. Updated: Added action to policy
4. Learning: Use AWS Policy Simulator to test policies
5. Result: S3 operations successful

CHALLENGE 4: Java Classpath
Problem: "cannot find symbol: class S3Client"
Root Cause: AWS SDK JAR not in classpath
Solution:
1. Downloaded: AWS SDK v2.20.0
2. Updated: Maven pom.xml
3. Ran: mvn clean install
4. Result: Compilation successful

CHALLENGE 5: Latency Measurement Variation
Problem: Latency values vary widely between iterations
Root Cause: Network conditions fluctuate, first request is slower
Solution:
1. Warmed up: Made initial request
2. Averaged: 3 iterations per operation
3. Documented: Variance ranges
4. Acknowledged: External factors affecting measurements
```

---

## Section 6: Conclusions and Lessons Learned (1-2 pages)

### What to Write

```
KEY TAKEAWAYS:

1. CLOUD SECURITY (Task 2.1)
   - Multiple security layers are essential
   - Principle of least privilege reduces risk
   - IAM roles are superior to embedded credentials
   - Security groups filter network-level access
   - Defense in depth requires coordination of multiple controls

2. CLOUD PROGRAMMING (Task 2.2)
   - AWS SDK abstracts low-level complexity
   - Understanding architecture helps with troubleshooting
   - Latency varies significantly across regions
   - Application placement decisions impact performance
   - Monitoring and measurement are critical for optimization

3. PRACTICAL SKILLS
   - Can configure cloud infrastructure using CLI
   - Can develop cloud-aware Java applications
   - Can measure and analyze cloud performance
   - Can implement security best practices
   - Can troubleshoot cloud service issues

RECOMMENDATIONS FOR FURTHER LEARNING:
1. Explore AWS Lambda for serverless computing
2. Study AWS CloudFormation for Infrastructure as Code
3. Learn about S3 replication across regions
4. Investigate CloudFront for content distribution
5. Study AWS auto-scaling for load management
6. Learn about monitoring with CloudWatch
7. Explore disaster recovery strategies

INDUSTRY RELEVANCE:
This lab covers skills highly relevant in current job market:
- Cloud security expertise (in demand)
- Multi-cloud programming (valuable)
- AWS certification preparation (AWS Solutions Architect)
- Infrastructure automation (DevOps skills)
- Performance optimization (backend engineering)
```

---

## Section 7: Screenshots and Evidence (Appendix)

### What to Include

**Task 2.1 Screenshots:**
1. AWS CLI version output
2. Security group creation output
3. EC2 instance launch output
4. SSH connection successful
5. `aws s3 ls` output showing bucket access
6. S3 object upload successful
7. S3 object download successful
8. lab2-config.txt contents

**Task 2.2 Screenshots:**
1. Java compilation successful
2. S3BucketOperations program output
3. Bucket creation in multiple regions
4. S3 object listing output
5. S3LatencyMeasurement program output
6. Latency results table
7. Latency comparison chart/plot

**Code Listings:**
1. task-2-1-setup.sh (key sections)
2. Lab2S3Policy.json (complete policy)
3. S3BucketOperations.java (key methods)
4. S3LatencyMeasurement.java (measurement code)

---

## Section 8: Code Appendix

### What to Include

Complete listing of:
1. All shell scripts (with comments)
2. All Java source files
3. Configuration files (JSON policies)
4. Results files (CSV data)

Alternatively, provide:
- GitHub link to code repository
- Zip file with all code
- Note: Code must be readable with proper comments

---

## Writing Guidelines

### Structure
- **Use clear headings** (Section → Subsection → Sub-subsection)
- **Number sections** (2.1.1, 2.1.2, etc.)
- **Include table of contents** at beginning
- **Page references** for cross-references

### Formatting
- **Code blocks**: Use monospace font, syntax highlighting if possible
- **Diagrams**: Clear, labeled, referenced in text
- **Tables**: Professional formatting with borders
- **Figures**: Numbered (Figure 1, Figure 2, etc.)
- **References**: [1], [2], etc. with complete citations

### Tone and Style
- **Professional**: Formal academic tone
- **Technical accuracy**: Use correct terminology
- **Conciseness**: Avoid redundancy
- **Clarity**: Define technical terms on first use

### Length and Balance
- **Total report**: 15-25 pages (including appendix)
- **Task 2.1**: 30-40% of report
- **Task 2.2**: 30-40% of report
- **Analysis/Comparison**: 20-30% of report

### Group Contributions

Include a section like:

```
GROUP CONTRIBUTIONS:

Student A (30%):
- Implemented Task 2.1 setup script
- Configured AWS CLI
- Tested IAM roles and security groups
- Wrote reflection questions analysis

Student B (35%):
- Implemented S3BucketOperations.java
- Set up Java development environment
- Conducted latency measurements
- Created visualization and charts

Student C (35%):
- Implemented S3LatencyMeasurement.java
- Performed latency testing across regions
- Wrote SDK architecture analysis
- Created lab report document
```

---

## Final Checklist

Before submission:

- [ ] Report is 15-25 pages
- [ ] All sections completed
- [ ] Code is complete and compiles
- [ ] Screenshots are included and clear
- [ ] Reflection questions answered thoroughly
- [ ] Diagrams are present and labeled
- [ ] Grammar and spelling checked
- [ ] References are complete
- [ ] Group contributions documented
- [ ] PDF is properly formatted
- [ ] Code is in separate zip file
- [ ] README included with instructions

---

Good luck with your lab report! 📝
