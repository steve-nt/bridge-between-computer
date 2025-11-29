# D7078E Lab 2 - Complete Package Index

## 📑 All Files and Their Purpose

### 🔴 START HERE
**START_HERE.txt** (12 KB)
- ⭐ Read this first if you're new
- What is this lab about?
- Quick checklist
- Next steps and timeline
- Safety warnings and important notes

### 📖 Documentation Files

**README.md** (18 KB)
- Comprehensive overview of the entire lab
- Architecture diagrams
- Task checklists
- Troubleshooting section
- Learning outcomes
- Submission requirements

**lab-guide.md** (17 KB) ⭐ MAIN IMPLEMENTATION GUIDE
- Prerequisites and setup
- Detailed Task 2.1 steps (AWS CLI)
- Detailed Task 2.2 steps (Java SDK)
- Validation procedures
- Troubleshooting guide
- Quick reference commands
- Expected outputs

**lab-report-guide.md** (31 KB) ⭐ HOW TO WRITE YOUR REPORT
- Section-by-section writing guide
- What analysis to include
- Code explanation templates
- Diagram examples
- Reflection question answers
- Lab report structure
- Writing guidelines
- Submission checklist

**QUICK_REFERENCE.md** (9.1 KB)
- Cheat sheet for common commands
- Quick links and tips
- Time estimates
- File descriptions at a glance
- Emergency help procedures
- Learning check questions

**DELIVERABLES.txt** (11 KB)
- Complete list of all files
- File sizes and purposes
- Features summary
- Completion time estimates
- Support resources
- Version information

### 🔧 Automation Scripts (Executable)

**task-2-1-setup.sh** (11 KB) ✅ EXECUTABLE
- Creates all AWS infrastructure
- Creates security groups with rules
- Creates S3 bucket
- Creates IAM policy and role
- Launches EC2 instance
- Saves configuration automatically
- Error handling and validation

Usage:
```bash
./task-2-1-setup.sh YOUR_IP/32
```

**task-2-1-cleanup.sh** (6.2 KB) ✅ EXECUTABLE
- Deletes all created resources
- Terminates EC2 instances
- Removes security groups
- Deletes IAM roles and policies
- Removes S3 buckets
- Safety confirmations

Usage:
```bash
./task-2-1-cleanup.sh
```

### ☕ Java Source Code

**S3BucketOperations.java** (11 KB)
- CRUD operations on S3 buckets
- Create buckets in multiple regions
- List buckets
- Upload objects
- Download objects
- List objects
- Delete objects
- Error handling
- Console output with progress

Compile:
```bash
javac -cp ".:aws-sdk/*" S3BucketOperations.java
```

Run:
```bash
java -cp ".:aws-sdk/*" S3BucketOperations
```

**S3LatencyMeasurement.java** (11 KB)
- Measures S3 operation latency
- Tests multiple regions
- 1 MB test file operations
- 3 iterations per operation
- Upload and download latency
- Results calculation and display
- Regional comparison

Compile:
```bash
javac -cp ".:aws-sdk/*" S3LatencyMeasurement.java
```

Run:
```bash
java -cp ".:aws-sdk/*" S3LatencyMeasurement
```

---

## 📂 File Organization by Task

### Task 2.1: AWS CLI Configuration
- **Main Guide**: lab-guide.md (Section 2.1)
- **Implementation Script**: task-2-1-setup.sh
- **Cleanup Script**: task-2-1-cleanup.sh
- **Configuration Output**: lab2-config.txt (generated)
- **Policy Definition**: Lab2S3Policy.json (generated)
- **Help Files**: README.md, QUICK_REFERENCE.md, START_HERE.txt

### Task 2.2: Java Development
- **Main Guide**: lab-guide.md (Section 4.2)
- **CRUD Code**: S3BucketOperations.java
- **Latency Testing**: S3LatencyMeasurement.java
- **Help Files**: README.md, QUICK_REFERENCE.md
- **Example Output**: Shown in lab-guide.md

### Lab Report
- **Main Guide**: lab-report-guide.md ⭐ READ THIS FIRST
- **Examples**: Throughout lab-report-guide.md
- **Code to Include**: S3BucketOperations.java, S3LatencyMeasurement.java
- **Diagrams**: Instructions in lab-report-guide.md
- **Screenshots**: Take during Task 2.1 and 2.2 execution

---

## 🎯 How to Use This Package

### For Complete Beginners
1. Read: **START_HERE.txt** (orientation)
2. Read: **README.md** (overview)
3. Read: **lab-guide.md** (implementation steps)
4. Execute: **task-2-1-setup.sh** (Task 2.1)
5. Execute: **S3BucketOperations.java** and **S3LatencyMeasurement.java** (Task 2.2)
6. Read: **lab-report-guide.md** (how to write report)
7. Write your lab report

### For Experienced Users
1. Skim: **QUICK_REFERENCE.md** (commands and overview)
2. Execute: **task-2-1-setup.sh YOUR_IP/32**
3. Compile and run Java programs
4. Use **lab-report-guide.md** for report structure
5. Submit

### For Troubleshooting
1. Check: **QUICK_REFERENCE.md** "Most Common Issues"
2. Check: **lab-guide.md** "Troubleshooting Section"
3. Check: Inline code comments
4. Check: AWS CloudTrail logs
5. Contact instructor if still stuck

### For Writing Lab Report
1. Read: **lab-report-guide.md** (all sections)
2. Follow the structure and templates provided
3. Use code snippets as examples
4. Include screenshots from your implementation
5. Answer reflection questions thoroughly
6. Check final checklist before submission

---

## 📊 File Statistics

```
Total Package: ~155 KB

Documentation:  ~98 KB (63%)
  - lab-guide.md: 17 KB
  - lab-report-guide.md: 31 KB
  - README.md: 18 KB
  - QUICK_REFERENCE.md: 9.1 KB
  - START_HERE.txt: 12 KB
  - DELIVERABLES.txt: 11 KB

Code:  ~39 KB (25%)
  - S3BucketOperations.java: 11 KB
  - S3LatencyMeasurement.java: 11 KB
  - task-2-1-setup.sh: 11 KB
  - task-2-1-cleanup.sh: 6.2 KB

Other: ~18 KB (12%)
  - Generated files (lab2-config.txt, Lab2S3Policy.json, etc.)
```

---

## 📋 Quick Navigation

### By Task
- **Task 2.1**: lab-guide.md Section 2.1 → task-2-1-setup.sh → task-2-1-cleanup.sh
- **Task 2.2**: lab-guide.md Section 4.2 → S3BucketOperations.java → S3LatencyMeasurement.java

### By Purpose
- **Learn**: START_HERE.txt → README.md → lab-guide.md
- **Do**: task-2-1-setup.sh → Java programs → lab-report-guide.md
- **Debug**: QUICK_REFERENCE.md → Troubleshooting sections
- **Write**: lab-report-guide.md → Write report → Submit

### By Topic
- **AWS CLI**: lab-guide.md Section 2.1, task-2-1-setup.sh
- **AWS SDK**: lab-guide.md Section 4.2, S3BucketOperations.java
- **Security**: lab-report-guide.md reflection questions
- **Latency**: S3LatencyMeasurement.java, lab-report-guide.md Section 3.3
- **Report**: lab-report-guide.md (entire file)

---

## 🔍 Finding Information

### Need to understand...
- **Cloud security** → lab-report-guide.md Q1-Q5
- **IAM roles** → lab-guide.md Task 2.1 section
- **Security groups** → lab-guide.md Task 2.1 section
- **S3 client creation** → lab-guide.md Section 4.2.1
- **Latency measurement** → S3LatencyMeasurement.java, lab-report-guide.md Section 3.3
- **Java compilation** → QUICK_REFERENCE.md, lab-guide.md
- **Troubleshooting** → QUICK_REFERENCE.md, lab-guide.md Section 5

### Need to find...
- **Specific AWS command** → QUICK_REFERENCE.md or lab-guide.md
- **Java code example** → S3BucketOperations.java or lab-guide.md
- **How to write report section** → lab-report-guide.md
- **What to include in report** → lab-report-guide.md
- **Expected output** → lab-guide.md or QUICK_REFERENCE.md
- **Error solution** → QUICK_REFERENCE.md, lab-guide.md Troubleshooting

---

## ⏱️ Estimated Reading Time

| File | Reading Time | Purpose |
|------|--------------|---------|
| START_HERE.txt | 10 min | Orientation |
| README.md | 15 min | Overview |
| QUICK_REFERENCE.md | 5 min | Quick help |
| lab-guide.md | 30 min | Implementation guide |
| lab-report-guide.md | 20 min | Report structure |
| Inline code comments | 10 min | Understanding code |

**Total**: ~90 minutes of reading (spread throughout the lab)

---

## ✅ Before You Start

**Check You Have:**
- [ ] All files present (use `ls -la`)
- [ ] Scripts are executable (`chmod +x *.sh`)
- [ ] Java installed (java -version)
- [ ] AWS CLI installed (aws --version)
- [ ] AWS credentials configured (aws configure)
- [ ] Valid AWS account

**Read First:**
- [ ] This INDEX.md file (you are here)
- [ ] START_HERE.txt (orientation)
- [ ] README.md (overview)

**Then Proceed:**
- [ ] Follow lab-guide.md Task 2.1
- [ ] Follow lab-guide.md Task 2.2
- [ ] Write report using lab-report-guide.md
- [ ] Submit to Canvas

---

## 📞 Help Resources

**In This Package:**
1. START_HERE.txt - Quick orientation
2. QUICK_REFERENCE.md - Common issues and commands
3. lab-guide.md - Detailed troubleshooting section
4. Inline code comments - Explanation of logic

**Online Resources:**
1. AWS Documentation - https://docs.aws.amazon.com/
2. AWS CLI Reference - https://docs.aws.amazon.com/cli/
3. AWS SDK for Java - https://docs.aws.amazon.com/sdk-for-java/
4. Stack Overflow - Search with [amazon-s3] [aws-cli] [java] tags

**From Instructor:**
1. Canvas discussion board
2. Course email
3. Office hours

---

## 🎓 Using This Package for Learning

This package is designed for:

1. **Self-paced learning**: Read guides at your own speed
2. **Hands-on practice**: Execute scripts and Java programs
3. **Reference material**: Look up specific topics
4. **Academic submission**: Complete lab report with provided templates
5. **Group work**: Multiple students can use the same scripts and guides

**Not suitable for:**
- Copying without understanding
- Submitting without reading
- Skipping reflection questions
- Ignoring security best practices

---

## 📝 File Modification Notes

**Do Not Modify:**
- ✅ Any .sh files (unless fixing your IP)
- ✅ Any .java files (unless adding comments)
- ✅ Any guide files (read-only)

**Will Be Generated:**
- lab2-config.txt (from setup script)
- Lab2S3Policy.json (from setup script)
- lab2-keypair.pem (from setup script)
- latency-test-*.bin (from Java program)
- latency-results.csv (from Java program)
- Downloaded test files

**You Create:**
- Lab2Report.pdf (your lab report)
- Lab2-Code.zip (for submission)
- Screenshots folder (supporting images)

---

## 🔄 Version and Updates

**Package Version**: 1.0  
**Release Date**: November 29, 2025  
**AWS SDK Version**: 2.20.0  
**Minimum Java**: JDK 11  
**AWS CLI**: v2 or later  

**Updates**: Check if newer versions available on course canvas

---

## 🎯 Success Criteria

**Have you successfully used this package if:**

✅ Task 2.1 Complete:
- [ ] Setup script runs without errors
- [ ] EC2 instance created and accessible
- [ ] S3 bucket created and accessible
- [ ] IAM role working correctly
- [ ] Cleanup script removes all resources

✅ Task 2.2 Complete:
- [ ] Java programs compile without errors
- [ ] All S3 operations (CRUD) work
- [ ] Latency measured in 3 regions
- [ ] Results analyzed and documented

✅ Report Complete:
- [ ] 15-25 pages
- [ ] All sections filled
- [ ] Screenshots included
- [ ] Code appendix included
- [ ] Reflection questions answered
- [ ] PDF properly formatted

✅ Submission Ready:
- [ ] Report as PDF
- [ ] Code in zip file
- [ ] README included
- [ ] All files compressed
- [ ] Ready for Canvas upload

---

## 🚀 Ready to Begin?

1. Confirm you've read this INDEX.md ✓
2. Read START_HERE.txt for orientation
3. Read README.md for complete overview
4. Open lab-guide.md and follow Task 2.1
5. Follow Task 2.2 in lab-guide.md
6. Use lab-report-guide.md to write report
7. Submit to Canvas before November 30, 2025

**Good luck with your lab! 🎓**

For questions, check the relevant guide file listed above.

---

**Package Contents Version**: 1.0  
**Last Updated**: November 29, 2025  
**Total Files**: 10 main files + generated files  
**Total Size**: ~155 KB source + documentation
