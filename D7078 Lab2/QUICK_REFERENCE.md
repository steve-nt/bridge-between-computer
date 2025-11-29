# D7078E Lab 2 - Quick Reference Card

## 🎯 30-Second Overview

**Task 2.1**: Use AWS CLI to create security infrastructure (45 min)  
**Task 2.2**: Use Java SDK to test S3 operations and measure latency (1 hour)  
**Report**: Write 15-25 page analysis (3-5 hours)

---

## 🚀 Quick Start Commands

```bash
# 1. GET YOUR IP
curl https://checkip.amazonaws.com

# 2. CONFIGURE AWS
aws configure
aws sts get-caller-identity

# 3. RUN TASK 2.1 (replace IP)
./task-2-1-setup.sh YOUR_IP/32

# 4. VERIFY AND TEST
ssh -i lab2-keypair.pem ec2-user@<PUBLIC_IP>
  aws s3 ls
  exit

# 5. COMPILE JAVA (assuming AWS SDK in aws-sdk/ folder)
javac -cp ".:aws-sdk/*" S3BucketOperations.java
javac -cp ".:aws-sdk/*" S3LatencyMeasurement.java

# 6. RUN JAVA PROGRAMS
java -cp ".:aws-sdk/*" S3BucketOperations
java -cp ".:aws-sdk/*" S3LatencyMeasurement

# 7. CLEANUP WHEN DONE
./task-2-1-cleanup.sh
# Type "yes" when prompted
```

---

## 📚 Which Guide to Read?

| Need | Read This | Location |
|------|-----------|----------|
| Step-by-step setup | lab-guide.md | Sections 2-4 |
| Java development | lab-guide.md | Section 4.2 |
| Write lab report | lab-report-guide.md | All sections |
| Quick help | This file + README.md | Quick Start |
| Troubleshooting | lab-guide.md | Section 5 |

---

## 🔑 Key Concepts

### Task 2.1: Three Security Layers
```
Network Layer:       Security Group (restricts traffic)
Identity Layer:      IAM Role + Policy (grants permissions)
Instance Layer:      EC2 with attached role (uses permissions)
```

### Task 2.2: SDK Architecture
```
Your Code
   ↓
S3Client (from AWS SDK)
   ↓
CreateBucketRequest / PutObjectRequest / GetObjectRequest
   ↓
HTTP Client (Apache)
   ↓
AWS S3 API
```

---

## 📋 Task 2.1: Checklist

- [ ] `./task-2-1-setup.sh YOUR_IP/32` runs without errors
- [ ] lab2-config.txt created with infrastructure details
- [ ] Can SSH: `ssh -i lab2-keypair.pem ec2-user@<IP>`
- [ ] S3 access works: `aws s3 ls` (from EC2)
- [ ] File upload works: `aws s3 cp test.txt s3://bucket/`
- [ ] File download works: `aws s3 cp s3://bucket/file .`

## 📋 Task 2.2: Checklist

- [ ] S3BucketOperations.java compiles
- [ ] S3LatencyMeasurement.java compiles
- [ ] Both programs run successfully
- [ ] Created buckets in 3 regions
- [ ] Measured latency (3 iterations)
- [ ] Analyzed results and identified fastest region

---

## 🐛 Most Common Issues

### "Unable to locate credentials"
```bash
aws configure
cat ~/.aws/credentials  # Should show access key and secret key
aws sts get-caller-identity  # Should show your account ID
```

### "Cannot SSH to instance"
```bash
# Check security group allows your IP
aws ec2 describe-security-groups --query 'SecurityGroups[?GroupName==`lab2-security-group`]'

# Wait 2-3 minutes for instance to initialize
aws ec2 describe-instances --query 'Reservations[0].Instances[0].State.Name'

# Verify key permissions
chmod 400 lab2-keypair.pem
```

### "S3BucketOperations compilation fails"
```bash
# Download AWS SDK for Java 2.20.0
mvn dependency:copy-dependencies

# Or use Maven pom.xml:
# <dependency>
#   <groupId>software.amazon.awssdk</groupId>
#   <artifactId>s3</artifactId>
#   <version>2.20.0</version>
# </dependency>
```

### "Access Denied to S3"
```bash
# Verify IAM policy attached
aws iam list-attached-role-policies --role-name Lab2S3AccessRole

# Check role has S3 permissions
aws iam get-role-policy --role-name Lab2S3AccessRole --policy-name Lab2S3Policy
```

---

## 🎓 Reflection Questions Quick Answers

### Q1: Why IAM roles vs embedded credentials?
- ✅ Temporary credentials (auto-rotate)
- ✅ No secrets in code/config files
- ✅ Better audit trails
- ✅ Easy to revoke access

### Q2: What principle for SSH IP restriction?
- ✅ **Principle of Least Privilege**
- ✅ Reduces attack surface
- ✅ Only authorized user can connect

### Q3: Remove S3 actions from policy?
- ✅ AccessDenied error
- ✅ S3 operations fail
- ✅ Check CloudTrail for audit
- ✅ EC2 operations still work

### Q4: Restrict HTTP to internal network?
```bash
# Remove: 0.0.0.0/0
# Add: 10.0.0.0/8 (internal network)
aws ec2 revoke-security-group-ingress --group-id sg-xxx \
    --protocol tcp --port 80 --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress --group-id sg-xxx \
    --protocol tcp --port 80 --cidr 10.0.0.0/8
```

### Q5: Challenges in least-privilege policies?
- ✅ Finding exact actions needed
- ✅ ARN format variations
- ✅ Testing each permission
- ✅ Wildcard over-permissioning
- ✅ Keeping policies updated

---

## 📊 Expected Latency Results

| Region | Upload (ms) | Download (ms) | Total (ms) |
|--------|------------|--------------|-----------|
| us-east-1 | 200-300 | 150-250 | 350-550 |
| eu-west-1 | 400-550 | 350-500 | 750-1050 |
| ap-southeast-1 | 450-650 | 400-550 | 850-1200 |

**Note**: Actual latency varies with network conditions!

---

## 📝 Lab Report Structure (15-25 pages)

```
1. Introduction (1-2 pages)
2. Task 2.1 Details (3-4 pages)
   - Architecture
   - Implementation
   - Validation
   - Reflection questions
3. Task 2.2 Details (4-5 pages)
   - SDK explanation
   - Code implementation
   - Latency analysis
   - Results interpretation
4. Comparison & Analysis (1-2 pages)
5. Challenges & Solutions (1 page)
6. Conclusions (1 page)
7. Appendices (Code, screenshots, diagrams)
```

---

## 🔐 Security Checklist

- [ ] SSH restricted to YOUR_IP/32 only
- [ ] HTTP open to 0.0.0.0/0 (can restrict further)
- [ ] IAM policy has only needed S3 actions
- [ ] No hardcoded credentials in code
- [ ] EC2 uses IAM role, not credentials file
- [ ] All resources cleaned up after testing

---

## 💰 Cost Control

```bash
# Check current AWS costs
aws ce get-cost-and-usage --time-period Start=2025-11-01,End=2025-12-01

# Estimate with Calculator
# https://calculator.aws/

# Always cleanup when done!
./task-2-1-cleanup.sh
```

**Estimated cost**: <$0.10 USD for entire lab

---

## 📞 Emergency Help

**Still stuck?** Do this in order:

1. **Reread** the relevant section in lab-guide.md
2. **Check CloudTrail**: `aws cloudtrail lookup-events`
3. **Verify AWS access**: `aws sts get-caller-identity`
4. **Google the error**: Copy exact error message
5. **Ask instructor**: Email or Canvas message
6. **Check AWS status**: https://status.aws.amazon.com/

---

## ⏱️ Time Estimate

| Task | Time |
|------|------|
| Task 2.1 Setup & Testing | 45 min |
| Task 2.2 Development | 60 min |
| Lab Report Writing | 3-5 hours |
| **Total** | **4-6 hours** |

---

## 📂 Files at a Glance

| File | Purpose | Size |
|------|---------|------|
| README.md | Overview & quick start | 18 KB |
| lab-guide.md | Implementation steps | 17 KB |
| lab-report-guide.md | Report writing guide | 31 KB |
| task-2-1-setup.sh | Auto setup (EXECUTABLE) | 11 KB |
| task-2-1-cleanup.sh | Auto cleanup (EXECUTABLE) | 6.2 KB |
| S3BucketOperations.java | CRUD examples | 11 KB |
| S3LatencyMeasurement.java | Performance testing | 11 KB |

**Total Package**: ~105 KB (source + docs)

---

## ✅ Submission Checklist

Before uploading to Canvas:

**Code**:
- [ ] task-2-1-setup.sh works end-to-end
- [ ] S3BucketOperations.java compiles and runs
- [ ] S3LatencyMeasurement.java compiles and runs
- [ ] All .java files have comments
- [ ] Code is properly formatted

**Report**:
- [ ] 15-25 pages total
- [ ] All 5 reflection questions answered (2-3 paragraphs each)
- [ ] Screenshots included
- [ ] Code appendix included
- [ ] Diagrams included
- [ ] Group contributions documented
- [ ] Proper formatting and layout
- [ ] PDF version created

**Files**:
- [ ] Lab2Report.pdf (single file)
- [ ] Lab2-Code.zip (scripts + Java + README)
- [ ] Compressed and ready to upload

---

## 🎓 Learning Check

After completing the lab, can you:

- [ ] Explain the 3 security layers (network, identity, instance)?
- [ ] Describe what IAM policies are?
- [ ] Explain why least privilege matters?
- [ ] Describe how S3Client is created?
- [ ] Explain what RequestBody.fromFile does?
- [ ] Interpret latency results across regions?
- [ ] Explain why some regions are slower?
- [ ] Describe how instance metadata service works?

If "No" to any: Reread relevant section and run that task again.

---

## 🔗 Useful Links (Bookmarks)

- **AWS CloudTrail**: https://console.aws.amazon.com/cloudtrail/
- **EC2 Console**: https://console.aws.amazon.com/ec2/
- **S3 Console**: https://s3.console.aws.amazon.com/
- **IAM Console**: https://console.aws.amazon.com/iam/
- **AWS CLI Docs**: https://docs.aws.amazon.com/cli/
- **AWS SDK Java**: https://docs.aws.amazon.com/sdk-for-java/
- **LTU Report Template**: https://www.overleaf.com/latex/templates/...

---

## 💡 Pro Tips

1. **Save configuration**: lab2-config.txt is created automatically
2. **Use variables**: Source lab2-config.txt to reuse values
3. **Test permissions**: Use AWS Policy Simulator before running code
4. **Monitor costs**: Set up AWS billing alerts
5. **Version control**: Commit code to Git/GitHub
6. **Parallel work**: Multiple team members can work on different tasks
7. **Screenshot early**: Take screenshots as you complete each step
8. **Document bugs**: Write down any issues for report section

---

Last updated: November 29, 2025  
For full details, see: lab-guide.md and lab-report-guide.md
