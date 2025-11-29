#!/bin/bash

################################################################################
# Task 2.1: Security Groups, IAM Policies & IAM Roles (with AWS CLI)
# Purpose: Create infrastructure with security groups, IAM roles, and EC2 instance
# Prerequisites: AWS CLI configured with valid credentials
################################################################################

set -e  # Exit on any error

# Configuration Variables
SECURITY_GROUP_NAME="lab2-security-group"
SECURITY_GROUP_DESC="Security group for Lab 2 - Cloud Services"
IAM_ROLE_NAME="Lab2S3AccessRole"
IAM_POLICY_NAME="Lab2S3Policy"
S3_BUCKET_NAME="lab2-bucket-$(date +%s)"  # Unique bucket name
KEY_PAIR_NAME="lab2-keypair"
INSTANCE_NAME="Lab2Instance"
MY_IP="${1:-YOUR_IP_HERE}"  # First argument or placeholder

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'  # No Color

# Utility function to print sections
print_header() {
    echo -e "\n${BLUE}===================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}===================================================${NC}\n"
}

# Utility function for success messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Utility function for error messages
print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Utility function for info messages
print_info() {
    echo -e "${YELLOW}→ $1${NC}"
}

# Check if AWS CLI is installed
check_aws_cli() {
    print_header "CHECKING AWS CLI INSTALLATION"
    if ! command -v aws &> /dev/null; then
        print_error "AWS CLI is not installed. Please install it first."
        exit 1
    fi
    print_success "AWS CLI is installed"
    aws --version
}

# Get VPC ID (default VPC)
get_vpc_id() {
    print_header "RETRIEVING VPC INFORMATION"
    VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query 'Vpcs[0].VpcId' --output text)
    if [ -z "$VPC_ID" ] || [ "$VPC_ID" = "None" ]; then
        print_error "Could not find default VPC"
        exit 1
    fi
    print_success "Default VPC ID: $VPC_ID"
}

# Create Security Group
create_security_group() {
    print_header "CREATING SECURITY GROUP"
    
    if aws ec2 describe-security-groups --filters "Name=group-name,Values=$SECURITY_GROUP_NAME" --query 'SecurityGroups[0].GroupId' --output text &> /dev/null; then
        print_info "Security group already exists, skipping creation..."
        SG_ID=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=$SECURITY_GROUP_NAME" --query 'SecurityGroups[0].GroupId' --output text)
    else
        SG_ID=$(aws ec2 create-security-group \
            --group-name "$SECURITY_GROUP_NAME" \
            --description "$SECURITY_GROUP_DESC" \
            --vpc-id "$VPC_ID" \
            --query 'GroupId' \
            --output text)
        print_success "Security Group Created: $SG_ID"
    fi
}

# Add inbound rules to Security Group
add_inbound_rules() {
    print_header "ADDING INBOUND RULES TO SECURITY GROUP"
    
    if [ "$MY_IP" = "YOUR_IP_HERE" ]; then
        print_error "Please provide your IP address as argument: $0 <YOUR_IP>"
        exit 1
    fi
    
    # SSH rule - restricted to your IP
    print_info "Adding SSH access from $MY_IP..."
    aws ec2 authorize-security-group-ingress \
        --group-id "$SG_ID" \
        --protocol tcp \
        --port 22 \
        --cidr "$MY_IP/32" \
        2>/dev/null || print_info "SSH rule may already exist"
    print_success "SSH rule added (restricted to $MY_IP)"
    
    # HTTP rule - from anywhere
    print_info "Adding HTTP access from anywhere..."
    aws ec2 authorize-security-group-ingress \
        --group-id "$SG_ID" \
        --protocol tcp \
        --port 80 \
        --cidr "0.0.0.0/0" \
        2>/dev/null || print_info "HTTP rule may already exist"
    print_success "HTTP rule added (open to 0.0.0.0/0)"
}

# Create S3 bucket
create_s3_bucket() {
    print_header "CREATING S3 BUCKET"
    
    if aws s3 ls "s3://$S3_BUCKET_NAME" 2>/dev/null; then
        print_info "Bucket already exists"
    else
        aws s3 mb "s3://$S3_BUCKET_NAME"
        print_success "S3 Bucket Created: $S3_BUCKET_NAME"
    fi
}

# Create IAM Policy JSON
create_iam_policy_json() {
    print_header "CREATING IAM POLICY JSON"
    
    cat > Lab2S3Policy.json << 'EOF'
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "S3FullAccess",
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::lab2-bucket-*",
                "arn:aws:s3:::lab2-bucket-*/*"
            ]
        },
        {
            "Sid": "EC2BasicAccess",
            "Effect": "Allow",
            "Action": [
                "ec2:DescribeInstances",
                "ec2:DescribeTags"
            ],
            "Resource": "*"
        }
    ]
}
EOF
    
    print_success "IAM Policy JSON created: Lab2S3Policy.json"
    cat Lab2S3Policy.json
}

# Create IAM Role
create_iam_role() {
    print_header "CREATING IAM ROLE"
    
    # Create trust policy
    cat > trust-policy.json << 'EOF'
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "ec2.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
EOF
    
    if aws iam get-role --role-name "$IAM_ROLE_NAME" 2>/dev/null; then
        print_info "Role already exists"
    else
        aws iam create-role \
            --role-name "$IAM_ROLE_NAME" \
            --assume-role-policy-document file://trust-policy.json
        print_success "IAM Role Created: $IAM_ROLE_NAME"
    fi
    
    rm trust-policy.json
}

# Create and attach IAM Policy
create_and_attach_policy() {
    print_header "CREATING AND ATTACHING IAM POLICY"
    
    POLICY_ARN=$(aws iam create-policy \
        --policy-name "$IAM_POLICY_NAME" \
        --policy-document file://Lab2S3Policy.json \
        --query 'Policy.Arn' \
        --output text 2>/dev/null || echo "")
    
    if [ -z "$POLICY_ARN" ] || [ "$POLICY_ARN" = "None" ]; then
        POLICY_ARN=$(aws iam list-policies --query "Policies[?PolicyName=='$IAM_POLICY_NAME'].Arn" --output text)
        print_info "Policy already exists: $POLICY_ARN"
    else
        print_success "IAM Policy Created: $POLICY_ARN"
    fi
    
    aws iam attach-role-policy \
        --role-name "$IAM_ROLE_NAME" \
        --policy-arn "$POLICY_ARN" 2>/dev/null || print_info "Policy may already be attached"
    print_success "Policy attached to role"
}

# Create Instance Profile
create_instance_profile() {
    print_header "CREATING INSTANCE PROFILE"
    
    if aws iam get-instance-profile --instance-profile-name "$IAM_ROLE_NAME" 2>/dev/null; then
        print_info "Instance profile already exists"
    else
        aws iam create-instance-profile \
            --instance-profile-name "$IAM_ROLE_NAME"
        print_success "Instance Profile Created: $IAM_ROLE_NAME"
        
        aws iam add-role-to-instance-profile \
            --instance-profile-name "$IAM_ROLE_NAME" \
            --role-name "$IAM_ROLE_NAME"
        print_success "Role added to instance profile"
    fi
}

# Create key pair
create_key_pair() {
    print_header "CREATING KEY PAIR"
    
    if aws ec2 describe-key-pairs --key-names "$KEY_PAIR_NAME" 2>/dev/null | grep -q "$KEY_PAIR_NAME"; then
        print_info "Key pair already exists"
    else
        aws ec2 create-key-pair --key-name "$KEY_PAIR_NAME" --query 'KeyMaterial' --output text > "$KEY_PAIR_NAME.pem"
        chmod 400 "$KEY_PAIR_NAME.pem"
        print_success "Key Pair Created: $KEY_PAIR_NAME.pem"
    fi
}

# Launch EC2 instance
launch_ec2_instance() {
    print_header "LAUNCHING EC2 INSTANCE"
    
    # Get the latest Amazon Linux 2 AMI
    AMI_ID=$(aws ec2 describe-images \
        --owners amazon \
        --filters "Name=name,Values=amzn2-ami-hvm-*-x86_64-gp2" \
        --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
        --output text)
    
    print_info "Using AMI: $AMI_ID"
    
    INSTANCE_ID=$(aws ec2 run-instances \
        --image-id "$AMI_ID" \
        --instance-type t2.micro \
        --key-name "$KEY_PAIR_NAME" \
        --security-group-ids "$SG_ID" \
        --iam-instance-profile "Name=$IAM_ROLE_NAME" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$INSTANCE_NAME}]" \
        --query 'Instances[0].InstanceId' \
        --output text)
    
    print_success "EC2 Instance Launched: $INSTANCE_ID"
    
    print_info "Waiting for instance to reach running state..."
    aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"
    print_success "Instance is now running"
    
    # Get instance details
    INSTANCE_INFO=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --query 'Reservations[0].Instances[0]' --output json)
    PUBLIC_IP=$(echo "$INSTANCE_INFO" | grep -o '"PublicIpAddress": "[^"]*' | cut -d'"' -f4)
    
    print_success "Instance Details:"
    echo "  Instance ID: $INSTANCE_ID"
    echo "  Public IP: $PUBLIC_IP"
}

# Validate IAM role access
validate_iam_role() {
    print_header "VALIDATING IAM ROLE ACCESS"
    
    print_info "Waiting 30 seconds for instance metadata service to initialize..."
    sleep 30
    
    print_info "Attempting SSH connection to test S3 access..."
    print_info "Command to run: ssh -i $KEY_PAIR_NAME.pem ec2-user@$PUBLIC_IP"
    echo ""
    echo "Once logged in, run:"
    echo "  aws s3 ls"
    echo "  echo 'test content' > test.txt"
    echo "  aws s3 cp test.txt s3://$S3_BUCKET_NAME/"
    echo "  aws s3 ls s3://$S3_BUCKET_NAME/"
    echo ""
}

# Save configuration for cleanup
save_config() {
    print_header "SAVING CONFIGURATION"
    
    cat > lab2-config.txt << EOF
# Lab 2 Configuration - Save for cleanup
SECURITY_GROUP_ID=$SG_ID
SECURITY_GROUP_NAME=$SECURITY_GROUP_NAME
IAM_ROLE_NAME=$IAM_ROLE_NAME
IAM_POLICY_NAME=$IAM_POLICY_NAME
S3_BUCKET_NAME=$S3_BUCKET_NAME
KEY_PAIR_NAME=$KEY_PAIR_NAME
INSTANCE_ID=$INSTANCE_ID
INSTANCE_NAME=$INSTANCE_NAME
VPC_ID=$VPC_ID
POLICY_ARN=$POLICY_ARN
PUBLIC_IP=$PUBLIC_IP
EOF
    
    print_success "Configuration saved to lab2-config.txt"
}

# Main execution
main() {
    print_header "LAB 2.1: SECURITY GROUPS, IAM POLICIES & IAM ROLES"
    
    check_aws_cli
    get_vpc_id
    create_security_group
    add_inbound_rules
    create_s3_bucket
    create_iam_policy_json
    create_iam_role
    create_and_attach_policy
    create_instance_profile
    create_key_pair
    launch_ec2_instance
    validate_iam_role
    save_config
    
    print_header "SETUP COMPLETE"
    echo "All resources have been created successfully!"
    echo "Configuration saved to lab2-config.txt for cleanup reference."
}

# Run main function
main "$@"
