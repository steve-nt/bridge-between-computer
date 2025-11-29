# D7078E Lab 2: Cloud Services (Storage) - Complete Lab Guide

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Task 2.1: Security Groups, IAM Policies & IAM Roles](#task-21)
4. [Task 2.2: AWS SDK with Java](#task-22)
5. [Troubleshooting](#troubleshooting)
6. [Resource Cleanup](#cleanup)

---

## Overview <a name="overview"></a>

This lab is divided into two main tasks:

**Task 2.1**: Configure AWS infrastructure using AWS CLI
- Create Security Groups with inbound rules
- Design and attach IAM policies (least privilege)
- Create and attach IAM roles to EC2 instances
- Launch EC2 instance with proper security configuration

**Task 2.2**: Develop cloud services using AWS SDK for Java
- Create S3 buckets in multiple regions
- Perform CRUD operations on S3 objects
- Measure and analyze latency across regions

---

## Prerequisites <a name="prerequisites"></a>

### Required Software
```bash
# 1. AWS CLI (v2 recommended)
aws --version  # Should show version 2.x.x

# 2. Java Development Kit (JDK 11+)
java -version
javac -version

# 3. IDE (IntelliJ IDEA, Eclipse, or NetBeans)

# 4. Maven (for Java project management)
mvn --version
```

### AWS Account Setup
1. Valid AWS Account with console access
2. AWS Credentials configured:
   ```bash
   aws configure
   # Enter: AWS Access Key ID
   # Enter: AWS Secret Access Key
   # Enter: Default region (e.g., us-east-1)
   # Enter: Default output format (json)
   ```

3. Verify configuration:
   ```bash
   aws sts get-caller-identity
   ```

### Get Your IP Address
```bash
# For IPv4
curl https://checkip.amazonaws.com

# For IPv6
curl https://checkipv6.amazonaws.com
```

---

## Task 2.1: AWS CLI - Setup & Configuration <a name="task-21"></a>

### Step 1: Prepare the Script

```bash
# Make the setup script executable
chmod +x task-2-1-setup.sh

# Check the script for review
cat task-2-1-setup.sh
```

### Step 2: Run the Setup Script

```bash
# Replace YOUR_IP with your actual public IP address
./task-2-1-setup.sh YOUR_IP

# Example:
./task-2-1-setup.sh 192.168.1.100/32
```

**What the script does:**
1. ✓ Checks AWS CLI installation
2. ✓ Creates a security group with custom rules
3. ✓ Adds SSH inbound rule (restricted to your IP only)
4. ✓ Adds HTTP inbound rule (open to 0.0.0.0/0)
5. ✓ Creates an S3 bucket
6. ✓ Creates IAM policy with S3 and EC2 permissions
7. ✓ Creates IAM role and instance profile
8. ✓ Attaches policy to role
9. ✓ Creates EC2 key pair
10. ✓ Launches EC2 instance with role and security group
11. ✓ Saves configuration to `lab2-config.txt`

### Step 3: Verify Setup Completion

```bash
# Check the configuration file
cat lab2-config.txt

# Expected output:
# SECURITY_GROUP_ID=sg-xxxxxxxxxxxxxx
# INSTANCE_ID=i-xxxxxxxxxxxxxx
# S3_BUCKET_NAME=lab2-bucket-1234567890
# etc.
```

### Step 4: Connect to EC2 Instance and Validate IAM Role

```bash
# SSH into your instance (replace IP with PUBLIC_IP from lab2-config.txt)
ssh -i lab2-keypair.pem ec2-user@<PUBLIC_IP>

# Once logged in, test S3 access (IAM role should work automatically)
aws s3 ls

# Upload a file to test S3 access
echo "Test content" > test.txt
aws s3 cp test.txt s3://lab2-bucket-<your-bucket-name>/

# Verify upload
aws s3 ls s3://lab2-bucket-<your-bucket-name>/

# Download the file
aws s3 cp s3://lab2-bucket-<your-bucket-name>/test.txt downloaded-test.txt

# Check file content
cat downloaded-test.txt

# Exit the instance
exit
```

### Step 5: Answer Reflection Questions

**Question 1: Why is it more secure to use IAM roles instead of embedding AWS access keys?**
- ✓ IAM roles provide temporary security credentials
- ✓ Credentials are automatically rotated by AWS
- ✓ No need to embed long-term secrets in code
- ✓ Fine-grained access control through policies
- ✓ Can audit role access with CloudTrail

**Question 2: What principle of cloud security is applied when restricting SSH access to a single IP?**
- ✓ Principle of Least Privilege
- ✓ Limits attack surface
- ✓ Only authorized users can access the instance
- ✓ Defense in depth with security groups

**Question 3: What happens if you remove the S3 actions from the IAM policy and try the upload again?**
- Access Denied (AccessDenied) error
- The instance can still describe EC2 resources (based on policy)
- S3 operations will fail with permission denied
- Use AWS CloudTrail to audit the failed attempts

**Question 4: How would you modify the Security Group to restrict HTTP access to an internal network only?**
```bash
# Replace 0.0.0.0/0 with your internal CIDR
aws ec2 revoke-security-group-ingress \
    --group-id sg-xxxxxx \
    --protocol tcp \
    --port 80 \
    --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
    --group-id sg-xxxxxx \
    --protocol tcp \
    --port 80 \
    --cidr 10.0.0.0/8  # Internal network CIDR
```

**Question 5: What challenges did you face when designing least-privilege policies?**
- Balancing security with functionality
- Testing all required permissions
- Understanding AWS action granularity
- Different resource ARN formats
- Managing policy versioning

---

## Task 2.2: AWS SDK with Java <a name="task-22"></a>

### Part I: Environment Setup

#### Step 1: Install AWS SDK for Java

If using Maven, add to `pom.xml`:

```xml
<dependency>
    <groupId>software.amazon.awssdk</groupId>
    <artifactId>s3</artifactId>
    <version>2.20.0</version>
</dependency>
```

Or use Gradle:

```gradle
implementation 'software.amazon.awssdk:s3:2.20.0'
```

#### Step 2: Configure AWS Credentials for Java

Option A: Use AWS CLI credentials (recommended)
```bash
# Credentials are automatically picked up from ~/.aws/credentials
# No additional configuration needed!
```

Option B: Set environment variables
```bash
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key
export AWS_REGION=us-east-1
```

Option C: Hardcode in application (not recommended for production)
```java
StaticCredentialsProvider.create(AwsBasicCredentials.create(
    "access-key-id",
    "secret-access-key"
));
```

#### Step 3: IDE Configuration

**IntelliJ IDEA:**
1. Create new Maven Project
2. File → Project Structure → Project Settings → SDK → Choose JDK 11+
3. Add AWS SDK dependency to pom.xml
4. Right-click project → Maven → Reload Projects

**Eclipse:**
1. File → New → Maven Project
2. Select DynamoDB/S3 archetype
3. Configure project group ID and artifact ID
4. Add AWS SDK dependency

**NetBeans:**
1. File → New Project → Maven → Maven Project
2. Configure project properties
3. Right-click Libraries → Add JAR/Folder (AWS SDK)

### Part II: S3 Operations with Java

#### Step 1: Compile Java Programs

```bash
# Assuming AWS SDK is in your classpath

# Compile S3BucketOperations.java
javac -cp ".:path/to/aws-sdk/*" S3BucketOperations.java

# Compile S3LatencyMeasurement.java
javac -cp ".:path/to/aws-sdk/*" S3LatencyMeasurement.java
```

#### Step 2: Run S3 Bucket Operations

```bash
# Run basic S3 operations
java -cp ".:path/to/aws-sdk/*" S3BucketOperations

# Expected Output:
# === AWS S3 Bucket Operations Demo ===
# 
# >>> TASK 1: Creating S3 Buckets in Multiple Regions
# 
# Creating bucket: lab2-bucket-us-east-1-1701231456
# Region: us-east-1
# ✓ Bucket created successfully
# ... (similar for other regions)
```

**What the program does:**
1. Creates S3 buckets in 3 regions (us-east-1, eu-west-1, ap-southeast-1)
2. Lists all buckets in the account
3. Uploads test objects to buckets
4. Downloads objects and measures latency
5. Lists all objects in bucket
6. Deletes objects from bucket

#### Step 3: Run Latency Measurement

```bash
# Run latency measurement across regions
java -cp ".:path/to/aws-sdk/*" S3LatencyMeasurement

# Expected Output:
# === AWS S3 Latency Measurement Across Regions ===
# 
# Testing Region: us-east-1
# ==================================================
# Test file created: latency-test-us-east-1.bin (1024 KB)
# Bucket created: latency-test-us-east-1-1701231500
# 
# Upload Tests (3 iterations):
#   Iteration 1: 245ms
#   Iteration 2: 234ms
#   Iteration 3: 256ms
# 
# Download Tests (3 iterations):
#   Iteration 1: 189ms
#   Iteration 2: 201ms
#   Iteration 3: 195ms
# ... (similar for other regions)
#
# ================================================================================
# LATENCY MEASUREMENT RESULTS
# ================================================================================
# Region               Avg Upload (ms)      Avg Download (ms)    Avg Total (ms)
# --------------------------------------------------------------------------------
# us-east-1           245.00               195.00               440.00
# eu-west-1           456.00               412.00               868.00
# ap-southeast-1      523.00               478.00               1001.00
```

### Step 4: Analyze Results

Create a CSV file for plotting:

```bash
# Extract results into CSV for analysis
cat > latency-results.csv << EOF
Region,Upload_ms,Download_ms,Total_ms
us-east-1,245,195,440
eu-west-1,456,412,868
ap-southeast-1,523,478,1001
EOF
```

Use tools to plot:
- **Python + Matplotlib**:
```python
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('latency-results.csv')
data.plot(x='Region', y=['Upload_ms', 'Download_ms'], kind='bar')
plt.title('S3 Operation Latency by Region')
plt.ylabel('Latency (ms)')
plt.tight_layout()
plt.savefig('latency-comparison.png')
```

- **Or use online tools**: https://www.chartjs.org/

### Understanding S3 Clients in Java

#### Architecture & Dependencies

```
Application Code
    ↓
S3Client (Interface)
    ↓
S3ClientBuilder
    ├── Region Configuration
    ├── Credentials Provider
    ├── HTTP Client
    └── Retry Strategy
    ↓
Transport Layer (HTTP)
    ├── Apache HttpClient (default)
    ├── Netty
    └── URL Connection
    ↓
AWS S3 API Endpoints
```

#### Key Classes and Packages:

```
software.amazon.awssdk.services.s3
├── S3Client (main interface)
├── S3ClientBuilder (fluent builder pattern)
├── model/ (request/response classes)
│   ├── CreateBucketRequest
│   ├── PutObjectRequest
│   ├── GetObjectRequest
│   ├── ListBucketsResponse
│   └── ... (70+ model classes)
└── paginators/ (pagination support)

software.amazon.awssdk.regions
├── Region (enum of AWS regions)
└── RegionMetadata

software.amazon.awssdk.auth
├── AwsCredentials
├── DefaultCredentialsProvider
└── StaticCredentialsProvider

software.amazon.awssdk.core
├── ResponseTransformer (result handling)
├── SdkClient (base interface)
└── sync/async request bodies
```

#### Creating S3 Client - Example Code

```java
// Default client (uses AWS credentials from environment/config)
try (S3Client s3Client = S3Client.builder()
        .region(Region.US_EAST_1)
        .build()) {
    // Use client
}

// With custom region endpoint
try (S3Client s3Client = S3Client.builder()
        .region(Region.EU_WEST_1)
        .endpointOverride(URI.create("https://s3.eu-west-1.amazonaws.com"))
        .build()) {
    // Use client
}

// With custom credentials
try (S3Client s3Client = S3Client.builder()
        .region(Region.US_EAST_1)
        .credentialsProvider(
            StaticCredentialsProvider.create(
                AwsBasicCredentials.create("access-key", "secret-key")
            )
        )
        .build()) {
    // Use client
}
```

---

## Troubleshooting <a name="troubleshooting"></a>

### Task 2.1 Issues

**Problem: "Unable to locate credentials"**
```bash
Solution:
1. Run: aws configure
2. Verify: cat ~/.aws/credentials
3. Check: aws sts get-caller-identity
```

**Problem: "Invalid IP address format"**
```bash
Solution:
1. Get your IP: curl https://checkip.amazonaws.com
2. Use CIDR notation: IP/32 (e.g., 203.0.113.25/32)
3. Run: ./task-2-1-setup.sh 203.0.113.25/32
```

**Problem: "Security group cannot be deleted"**
```bash
Solution:
1. Terminate all instances first
2. Remove security group from network interfaces
3. Wait 30 seconds
4. Try deletion again
```

**Problem: "Cannot SSH to instance - Connection refused"**
```bash
Possible causes:
1. Security group SSH rule not allowing your IP
2. Instance not fully initialized (wait 2-3 minutes)
3. Wrong key pair or permissions (chmod 400 key.pem)
4. Check: aws ec2 describe-instances --instance-ids i-xxxxx
```

### Task 2.2 Issues

**Problem: "Cannot find aws-java-sdk classes"**
```bash
Solution:
1. Add to pom.xml (Maven):
   <dependency>
       <groupId>software.amazon.awssdk</groupId>
       <artifactId>s3</artifactId>
       <version>2.20.0</version>
   </dependency>

2. Run: mvn clean install
3. Or compile with: javac -cp ".:aws-sdk/*" S3BucketOperations.java
```

**Problem: "Access Denied to S3"**
```bash
Solution:
1. Check AWS credentials: aws sts get-caller-identity
2. Verify IAM permissions: aws iam list-attached-user-policies --user-name <your-user>
3. Check bucket policy: aws s3api get-bucket-policy --bucket <bucket-name>
```

**Problem: "Region not available or endpoint not found"**
```bash
Solution:
1. Verify region name: aws ec2 describe-regions
2. Check region format: "us-east-1" (not "US-EAST-1")
3. Ensure account has access to region
```

---

## Resource Cleanup <a name="cleanup"></a>

### Automatic Cleanup with Script

```bash
# Make cleanup script executable
chmod +x task-2-1-cleanup.sh

# Run cleanup (will prompt for confirmation)
./task-2-1-cleanup.sh

# You will be asked: "This will delete all Lab 2.1 resources. Are you sure? (yes/no):"
# Type: yes
```

**What cleanup does:**
1. Terminates EC2 instance
2. Deletes key pair
3. Deletes security group
4. Detaches and deletes IAM policy
5. Deletes IAM role and instance profile
6. Deletes S3 bucket and all objects
7. Removes local configuration files

### Manual Cleanup (if script fails)

```bash
# Load configuration
source lab2-config.txt

# Terminate instance
aws ec2 terminate-instances --instance-ids $INSTANCE_ID
aws ec2 wait instance-terminated --instance-ids $INSTANCE_ID

# Delete key pair
aws ec2 delete-key-pair --key-name $KEY_PAIR_NAME
rm -f ${KEY_PAIR_NAME}.pem

# Delete security group (wait 30 seconds after instance termination)
sleep 30
aws ec2 delete-security-group --group-id $SECURITY_GROUP_ID

# Delete IAM resources
aws iam detach-role-policy --role-name $IAM_ROLE_NAME --policy-arn $POLICY_ARN
aws iam delete-policy --policy-arn $POLICY_ARN
aws iam remove-role-from-instance-profile --instance-profile-name $IAM_ROLE_NAME --role-name $IAM_ROLE_NAME
aws iam delete-instance-profile --instance-profile-name $IAM_ROLE_NAME
aws iam delete-role --role-name $IAM_ROLE_NAME

# Delete S3 bucket
aws s3 rm s3://$S3_BUCKET_NAME --recursive
aws s3 rb s3://$S3_BUCKET_NAME
```

### Check AWS Costs

```bash
# View AWS Cost Explorer
aws ce get-cost-and-usage \
    --time-period Start=2025-01-01,End=2025-01-31 \
    --granularity MONTHLY \
    --metrics "UnblendedCost"

# Or use AWS Console:
# 1. Go to AWS Management Console
# 2. Search for "Billing"
# 3. Check "Cost Explorer"
```

---

## Quick Reference Commands

```bash
# AWS CLI - Security Groups
aws ec2 describe-security-groups --filters "Name=group-name,Values=lab2-*"
aws ec2 authorize-security-group-ingress --group-id sg-xxx --protocol tcp --port 22 --cidr YOUR_IP/32
aws ec2 delete-security-group --group-id sg-xxx

# AWS CLI - IAM
aws iam list-roles
aws iam get-role --role-name Lab2S3AccessRole
aws iam list-attached-role-policies --role-name Lab2S3AccessRole
aws iam delete-role --role-name Lab2S3AccessRole

# AWS CLI - S3
aws s3 ls                                           # List all buckets
aws s3 ls s3://bucket-name/                         # List objects
aws s3 cp file.txt s3://bucket-name/                # Upload
aws s3 cp s3://bucket-name/file.txt ./              # Download
aws s3 rm s3://bucket-name/file.txt                 # Delete object
aws s3 rb s3://bucket-name                          # Delete bucket

# AWS CLI - EC2
aws ec2 describe-instances
aws ec2 describe-instances --instance-ids i-xxxxx
aws ec2 run-instances --image-id ami-xxxxx --instance-type t2.micro
aws ec2 terminate-instances --instance-ids i-xxxxx
ssh -i key.pem ec2-user@PUBLIC_IP

# Java - Compile & Run
javac -cp ".:aws-sdk/*" FileName.java
java -cp ".:aws-sdk/*" ClassName
mvn clean compile exec:java -Dexec.mainClass="ClassName"
```

---

## Lab Report Structure

See `lab-report-guide.md` for detailed instructions on writing the lab report.

---

## Additional Resources

- [AWS CLI Documentation](https://docs.aws.amazon.com/cli/)
- [AWS SDK for Java Developer Guide](https://docs.aws.amazon.com/sdk-for-java/latest/developer-guide/)
- [S3 API Reference](https://docs.aws.amazon.com/AmazonS3/latest/API/)
- [IAM Best Practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)
- [AWS Security Groups](https://docs.aws.amazon.com/vpc/latest/userguide/VPC_SecurityGroups.html)

---

## Support

For issues or questions:
1. Check the Troubleshooting section above
2. Review AWS CloudTrail logs: `aws cloudtrail lookup-events`
3. Check AWS service health: https://status.aws.amazon.com/
4. Contact your course instructor via email or Canvas
