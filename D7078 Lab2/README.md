# D7078E Lab 2: Cloud Services (Storage) - Complete Solution Package

**Course**: D7078E: Cloud Security  
**Lab**: Lab 2 - Programming Cloud Services (Storage)  
**Deadline**: November 30, 2025  

---

## 📦 Package Contents

This repository contains everything needed to complete Lab 2:

### Documentation
- **`lab-guide.md`** - Step-by-step implementation guide for both tasks
- **`lab-report-guide.md`** - Detailed guide on writing the lab report with analysis tips
- **`README.md`** - This file

### Scripts
- **`task-2-1-setup.sh`** - Automated AWS infrastructure setup (Task 2.1)
- **`task-2-1-cleanup.sh`** - Automated AWS resource cleanup

### Java Source Code
- **`S3BucketOperations.java`** - CRUD operations on S3 buckets
- **`S3LatencyMeasurement.java`** - Performance testing across AWS regions

---

## 🚀 Quick Start

### Prerequisites
```bash
# Check Java installation (11+)
java -version

# Check AWS CLI (v2)
aws --version

# Configure AWS credentials
aws configure

# Verify credentials work
aws sts get-caller-identity
```

### Task 2.1: Security Configuration (30 minutes)

```bash
# Get your public IP
curl https://checkip.amazonaws.com

# Run setup script (replace IP)
./task-2-1-setup.sh 203.0.113.100/32

# SSH into instance and test S3 access
ssh -i lab2-keypair.pem ec2-user@<INSTANCE_IP>
  aws s3 ls
  exit

# Clean up when done
./task-2-1-cleanup.sh
```

### Task 2.2: Java Development (1-2 hours)

```bash
# Setup AWS SDK for Java
# Add to pom.xml or download JAR files

# Compile programs
javac -cp ".:aws-sdk/*" S3BucketOperations.java
javac -cp ".:aws-sdk/*" S3LatencyMeasurement.java

# Run operations
java -cp ".:aws-sdk/*" S3BucketOperations

# Run latency measurement (5-10 minutes)
java -cp ".:aws-sdk/*" S3LatencyMeasurement

# Analyze results and create charts
# (Results saved in latency-results.csv)
```

---

## 📚 Detailed Guides

### For Implementation: Read `lab-guide.md`
Contains:
- ✅ Prerequisites and setup instructions
- ✅ Step-by-step execution for Task 2.1
- ✅ Complete Task 2.2 Java development guide
- ✅ Troubleshooting section
- ✅ Resource cleanup procedures
- ✅ Quick reference commands

### For Lab Report: Read `lab-report-guide.md`
Contains:
- ✅ What to write in each section
- ✅ Analysis tips for reflection questions
- ✅ Diagram templates
- ✅ Example code explanations
- ✅ Results interpretation guide
- ✅ Writing structure and guidelines

---

## 🏗️ Architecture Overview

### Task 2.1: Infrastructure Setup
```
┌─────────────────────────────────────────────────────┐
│                   AWS Account                       │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │           Default VPC (us-east-1)            │   │
│  │                                              │   │
│  │  ┌─────────────────────────────────────┐    │   │
│  │  │    Security Group                   │    │   │
│  │  │                                     │    │   │
│  │  │  Inbound Rules:                     │    │   │
│  │  │  • SSH (22) from YOUR_IP/32        │    │   │
│  │  │  • HTTP (80) from 0.0.0.0/0        │    │   │
│  │  │                                     │    │   │
│  │  │  ┌──────────────────────────────┐  │    │   │
│  │  │  │     EC2 Instance             │  │    │   │
│  │  │  │     (Amazon Linux 2)         │  │    │   │
│  │  │  │     Role: Lab2S3AccessRole   │  │    │   │
│  │  │  └──────────────────────────────┘  │    │   │
│  │  └─────────────────────────────────────┘    │   │
│  └──────────────────────────────────────────────┘   │
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │  IAM Role: Lab2S3AccessRole                 │   │
│  │  ├─ Policy: Lab2S3Policy                    │   │
│  │  │  └─ S3 permissions (read/write/delete)   │   │
│  │  ├─ Trust: EC2.amazonaws.com                │   │
│  │  └─ Instance Profile: Lab2S3AccessRole      │   │
│  └──────────────────────────────────────────────┘   │
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │  S3 Bucket: lab2-bucket-[timestamp]         │   │
│  │  Region: us-east-1                          │   │
│  └──────────────────────────────────────────────┘   │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### Task 2.2: Application Architecture
```
Application Layer
└── Java S3 Programs
    ├── S3BucketOperations.java
    │   ├── createBucketsInRegions()
    │   ├── listBuckets()
    │   ├── uploadObjects()
    │   ├── downloadObjects()
    │   └── deleteObjects()
    │
    └── S3LatencyMeasurement.java
        ├── testRegion()
        ├── measureUpload()
        ├── measureDownload()
        └── generateResults()

SDK Layer (software.amazon.awssdk)
└── S3Client
    ├── CreateBucketRequest
    ├── ListBucketsResponse
    ├── PutObjectRequest
    ├── GetObjectRequest
    └── DeleteObjectRequest

Transport Layer
└── Apache HttpClient
    └── AWS Signature v4

AWS Services
├── S3 (us-east-1)
├── S3 (eu-west-1)
└── S3 (ap-southeast-1)
```

---

## 📋 Task Checklist

### Task 2.1: AWS CLI Configuration
- [ ] AWS CLI configured with credentials
- [ ] Security group created with SSH and HTTP rules
- [ ] S3 bucket created
- [ ] IAM policy created (Lab2S3Policy.json)
- [ ] IAM role created and policy attached
- [ ] Instance profile created
- [ ] EC2 key pair created
- [ ] EC2 instance launched with role and security group
- [ ] Verified SSH access
- [ ] Verified S3 access via IAM role
- [ ] Configuration saved to lab2-config.txt
- [ ] Cleanup script tested

### Task 2.2: Java Development
- [ ] AWS SDK for Java installed
- [ ] Java IDE configured
- [ ] S3BucketOperations.java compiled
- [ ] S3LatencyMeasurement.java compiled
- [ ] Buckets created in 3 regions
- [ ] List buckets operation works
- [ ] Upload operation works (at least 1 MB file)
- [ ] Download operation works
- [ ] Delete operation works
- [ ] Latency measurements collected (3 iterations per region)
- [ ] Results analyzed and documented
- [ ] Graphs/charts created

### Lab Report
- [ ] Introduction written (1-2 pages)
- [ ] Task 2.1 implementation documented (3-4 pages)
- [ ] Task 2.2 implementation documented (4-5 pages)
- [ ] All reflection questions answered (2-3 pages)
- [ ] Challenges and solutions documented
- [ ] Conclusions written
- [ ] Screenshots included
- [ ] Code appendix complete
- [ ] Group contributions documented
- [ ] Report formatted and proofread
- [ ] PDF exported
- [ ] Code zipped separately

---

## 🔐 Security Best Practices Implemented

### Task 2.1 Security Features
✅ **Principle of Least Privilege**
- IAM policy grants only S3 CRUD operations
- No administrative permissions
- Restricted to lab2-bucket-* resources

✅ **Defense in Depth**
- Network layer: Security groups restrict inbound traffic
- Identity layer: IAM roles provide authentication
- Instance layer: EC2 with specific role attachment

✅ **Restricted Access**
- SSH only from your specific IP (/32)
- HTTP open to internet (0.0.0.0/0) - can be restricted further
- No hardcoded credentials in EC2 user data

✅ **Temporary Credentials**
- IAM role provides temporary STS credentials
- Credentials auto-rotate every hour
- Not managed manually by developers

### Task 2.2 Security Features
✅ **No Hardcoded Credentials**
- Java code uses DefaultCredentialsProvider
- Credentials loaded from AWS CLI configuration
- No secrets in source code

✅ **Automatic Credential Rotation**
- SDK handles credential refresh automatically
- Invalid credentials trigger credential chain re-evaluation

---

## 📊 Expected Results

### Task 2.1 Output
```
Security Group ID: sg-0abc123xyz
IAM Role: Lab2S3AccessRole
S3 Bucket: lab2-bucket-1701231456
EC2 Instance ID: i-0def456xyz
Instance Public IP: 203.0.113.100
SSH Connection: SUCCESS
S3 Access via IAM Role: SUCCESS
```

### Task 2.2 Output (S3BucketOperations)
```
>>> TASK 1: Creating S3 Buckets in Multiple Regions

Creating bucket: lab2-bucket-us-east-1-1701231500
Region: us-east-1
✓ Bucket created successfully

>>> TASK 2: Listing All S3 Buckets

Total Buckets: 3
  - lab2-bucket-us-east-1-1701231500
  - lab2-bucket-eu-west-1-1701231505
  - lab2-bucket-ap-southeast-1-1701231510

>>> TASK 3-6: Upload, Download, List, Delete Operations

[All operations completed successfully with latency measurements]
```

### Task 2.2 Output (S3LatencyMeasurement)
```
================================================================================
LATENCY MEASUREMENT RESULTS
================================================================================
Region               Avg Upload (ms)      Avg Download (ms)    Avg Total (ms)
--------------------------------------------------------------------------------
us-east-1           245.00               195.00               440.00
eu-west-1           456.00               412.00               868.00
ap-southeast-1      523.00               478.00              1001.00
================================================================================

Fastest Region: us-east-1
```

---

## 🛠️ Troubleshooting Guide

### Common Issues and Solutions

**AWS CLI Issues**
```bash
# Cannot find credentials
aws configure  # Run setup
aws sts get-caller-identity  # Verify

# Invalid IP format
curl https://checkip.amazonaws.com  # Get correct IP
./task-2-1-setup.sh 203.0.113.25/32  # Use CIDR notation
```

**AWS Infrastructure Issues**
```bash
# Cannot SSH to instance
# 1. Check security group allows your IP
# 2. Wait 3 minutes for instance initialization
# 3. Verify key pair permissions: chmod 400 lab2-keypair.pem

# Instance role not working
# 1. Verify instance profile created: aws iam list-instance-profiles
# 2. Check role attached: aws iam get-instance-profile --instance-profile-name Lab2S3AccessRole
# 3. Wait 30 seconds for role to propagate
```

**Java Compilation Issues**
```bash
# Cannot find S3Client class
# 1. Add AWS SDK to pom.xml
# 2. Run: mvn clean install
# 3. Or manually add JAR files to classpath

# Classpath issues
# Use: javac -cp ".:lib/*:aws-sdk/*" FileName.java
```

**S3 Access Issues**
```bash
# Access Denied errors
aws iam list-attached-role-policies --role-name Lab2S3AccessRole
aws s3api get-bucket-policy --bucket lab2-bucket-xxx

# Region issues
aws ec2 describe-regions --query 'Regions[].RegionName'
```

For detailed troubleshooting, see `lab-guide.md` section "Troubleshooting".

---

## 📝 Lab Report Submission Requirements

### File Format
- ✅ PDF format (generated from Overleaf or similar)
- ✅ Use LTU lab report template
- ✅ 15-25 pages total (including appendices)

### Content Requirements
- ✅ Introduction and objectives
- ✅ Task 2.1: AWS CLI implementation (30-40% of report)
- ✅ Task 2.2: Java development (30-40% of report)
- ✅ Analysis and comparison (20-30% of report)
- ✅ Reflection questions with detailed answers
- ✅ Challenges and solutions
- ✅ Conclusions and lessons learned
- ✅ Screenshots of all major steps
- ✅ Code appendix (commented and readable)
- ✅ Diagrams/architecture illustrations
- ✅ Results tables and graphs
- ✅ Group member contributions documented

### File Organization
```
Submission/
├── Lab2Report.pdf              (Main report)
├── Lab2-Code.zip               (All source code)
│   ├── task-2-1-setup.sh
│   ├── task-2-1-cleanup.sh
│   ├── S3BucketOperations.java
│   ├── S3LatencyMeasurement.java
│   ├── pom.xml                 (if using Maven)
│   └── README.md               (Instructions to compile/run)
└── Screenshots/                (Supporting images)
    ├── setup-output.png
    ├── latency-results.png
    └── ...
```

---

## 🔗 Important Links

### AWS Documentation
- [AWS CLI User Guide](https://docs.aws.amazon.com/cli/latest/userguide/)
- [AWS SDK for Java Developer Guide](https://docs.aws.amazon.com/sdk-for-java/latest/developer-guide/)
- [Amazon S3 API Reference](https://docs.aws.amazon.com/AmazonS3/latest/API/)
- [IAM User Guide](https://docs.aws.amazon.com/iam/)
- [EC2 User Guide](https://docs.aws.amazon.com/ec2/)

### LTU Resources
- [LTU Lab Report Template (Overleaf)](https://www.overleaf.com/latex/templates/lulea-university-of-technology-english-report-template-tvm-department/tfhyswbgngsr)
- [LTU Plagiarism Policy](https://www.ltu.se/en/student-web/your-studies/students-rights-and-obligations/cheating-and-plagiarism)

### Tools and References
- [AWS Architecture Icons](https://aws.amazon.com/architecture/icons/)
- [AWS Pricing Calculator](https://calculator.aws/)
- [AWS Well-Architected Framework](https://aws.amazon.com/architecture/well-architected/)

---

## 📞 Support and Questions

### Before Asking for Help
1. Check `lab-guide.md` troubleshooting section
2. Review AWS CloudTrail logs: `aws cloudtrail lookup-events`
3. Check AWS service status: https://status.aws.amazon.com/
4. Verify AWS credentials: `aws sts get-caller-identity`

### Getting Help
- **AWS Documentation**: [docs.aws.amazon.com](https://docs.aws.amazon.com)
- **Stack Overflow**: Tag questions with `amazon-s3`, `aws-cli`, `java`
- **Course Instructor**: Email or Canvas message
- **Classmates**: Collaborate on understanding (not copying code)

---

## 📅 Important Dates

- **Lab Deadline**: November 30, 2025
- **Report Submission**: Canvas LAB 2 folder
- **Format**: PDF + Code ZIP

---

## 📝 Version History

- **v1.0** (November 29, 2025): Initial release
  - Complete setup scripts
  - Java programs for S3 operations
  - Comprehensive guides
  - Lab report writing guide

---

## ✨ Key Features of This Solution Package

✅ **Automated Setup**
- One command to create all infrastructure
- Automatic configuration saving
- Error handling and recovery

✅ **Complete Java Examples**
- Multiple regions testing
- CRUD operation demonstrations
- Latency measurement and analysis

✅ **Comprehensive Documentation**
- Step-by-step implementation guide
- Detailed lab report guide
- Architecture diagrams
- Troubleshooting section

✅ **Security Best Practices**
- Least privilege IAM policies
- Secure credential handling
- Defense in depth approach
- Temporary credentials via IAM roles

✅ **Production-Ready Code**
- Error handling and validation
- Clear comments and documentation
- Configurable parameters
- Scalable architecture

---

## 📄 License and Attribution

This solution package is provided for educational purposes as part of the D7078E: Cloud Security course at Luleå University of Technology.

---

## 🎓 Learning Outcomes

After completing this lab, you will be able to:

### Task 2.1 Outcomes
✅ Create and manage AWS Security Groups using CLI
✅ Design least-privilege IAM policies
✅ Create and attach IAM roles to EC2 instances
✅ Understand identity-based access control in AWS
✅ Apply security best practices in cloud infrastructure

### Task 2.2 Outcomes
✅ Set up AWS SDK for Java in development environment
✅ Create and configure S3 service clients
✅ Implement CRUD operations on S3 objects
✅ Measure and analyze cloud service latency
✅ Understand multi-region cloud architecture implications

### Overall Outcomes
✅ Integrate security practices into cloud development
✅ Analyze cloud service performance
✅ Make informed decisions about cloud resource placement
✅ Document cloud solutions professionally
✅ Apply cloud computing concepts in production scenarios

---

Good luck with your lab! 🚀

For questions or clarifications, refer to the detailed guides:
- **Implementation**: `lab-guide.md`
- **Report Writing**: `lab-report-guide.md`
