#!/bin/bash

################################################################################
# Task 2.1 Cleanup: Delete all resources created during Lab 2.1
# Purpose: Remove infrastructure to avoid AWS charges
################################################################################

set -e

# Color codes
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "\n${BLUE}===================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}===================================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}→ $1${NC}"
}

# Load configuration from file
load_config() {
    print_header "LOADING CONFIGURATION"
    
    if [ ! -f "lab2-config.txt" ]; then
        print_error "Configuration file not found. Run setup script first."
        exit 1
    fi
    
    source lab2-config.txt
    print_success "Configuration loaded"
}

# Terminate EC2 instances
terminate_instances() {
    print_header "TERMINATING EC2 INSTANCES"
    
    if [ -z "$INSTANCE_ID" ]; then
        print_info "No instance ID found, skipping..."
        return
    fi
    
    STATUS=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null || echo "")
    
    if [ "$STATUS" = "running" ] || [ "$STATUS" = "stopped" ]; then
        print_info "Terminating instance: $INSTANCE_ID"
        aws ec2 terminate-instances --instance-ids "$INSTANCE_ID"
        
        print_info "Waiting for instance to terminate..."
        aws ec2 wait instance-terminated --instance-ids "$INSTANCE_ID"
        print_success "Instance terminated"
    else
        print_info "Instance already terminated or not found"
    fi
}

# Delete key pair
delete_key_pair() {
    print_header "DELETING KEY PAIR"
    
    if [ -z "$KEY_PAIR_NAME" ]; then
        print_info "No key pair name found, skipping..."
        return
    fi
    
    if aws ec2 describe-key-pairs --key-names "$KEY_PAIR_NAME" 2>/dev/null | grep -q "$KEY_PAIR_NAME"; then
        print_info "Deleting key pair: $KEY_PAIR_NAME"
        aws ec2 delete-key-pair --key-name "$KEY_PAIR_NAME"
        
        if [ -f "${KEY_PAIR_NAME}.pem" ]; then
            rm "${KEY_PAIR_NAME}.pem"
            print_success "Key pair deleted and PEM file removed"
        else
            print_success "Key pair deleted"
        fi
    else
        print_info "Key pair not found"
    fi
}

# Delete security group
delete_security_group() {
    print_header "DELETING SECURITY GROUP"
    
    if [ -z "$SECURITY_GROUP_ID" ]; then
        print_info "No security group ID found, skipping..."
        return
    fi
    
    print_info "Waiting 10 seconds for instance termination to fully complete..."
    sleep 10
    
    if aws ec2 describe-security-groups --group-ids "$SECURITY_GROUP_ID" 2>/dev/null; then
        print_info "Deleting security group: $SECURITY_GROUP_ID"
        aws ec2 delete-security-group --group-id "$SECURITY_GROUP_ID"
        print_success "Security group deleted"
    else
        print_info "Security group not found"
    fi
}

# Delete IAM policy
delete_iam_policy() {
    print_header "DELETING IAM POLICY"
    
    if [ -z "$POLICY_ARN" ]; then
        print_info "No policy ARN found, skipping..."
        return
    fi
    
    print_info "Detaching policy from role: $IAM_ROLE_NAME"
    aws iam detach-role-policy \
        --role-name "$IAM_ROLE_NAME" \
        --policy-arn "$POLICY_ARN" 2>/dev/null || print_info "Policy may not be attached"
    
    print_info "Deleting policy: $POLICY_ARN"
    aws iam delete-policy --policy-arn "$POLICY_ARN"
    print_success "Policy deleted"
}

# Delete IAM role and instance profile
delete_iam_role() {
    print_header "DELETING IAM ROLE AND INSTANCE PROFILE"
    
    if [ -z "$IAM_ROLE_NAME" ]; then
        print_info "No role name found, skipping..."
        return
    fi
    
    # Remove role from instance profile
    print_info "Removing role from instance profile..."
    aws iam remove-role-from-instance-profile \
        --instance-profile-name "$IAM_ROLE_NAME" \
        --role-name "$IAM_ROLE_NAME" 2>/dev/null || print_info "Role may not be in instance profile"
    
    # Delete instance profile
    print_info "Deleting instance profile: $IAM_ROLE_NAME"
    aws iam delete-instance-profile \
        --instance-profile-name "$IAM_ROLE_NAME"
    print_success "Instance profile deleted"
    
    # Delete role
    print_info "Deleting IAM role: $IAM_ROLE_NAME"
    aws iam delete-role --role-name "$IAM_ROLE_NAME"
    print_success "IAM role deleted"
}

# Delete S3 bucket
delete_s3_bucket() {
    print_header "DELETING S3 BUCKET"
    
    if [ -z "$S3_BUCKET_NAME" ]; then
        print_info "No bucket name found, skipping..."
        return
    fi
    
    if aws s3 ls "s3://$S3_BUCKET_NAME" 2>/dev/null; then
        print_info "Deleting all objects in bucket: $S3_BUCKET_NAME"
        aws s3 rm "s3://$S3_BUCKET_NAME" --recursive
        
        print_info "Deleting bucket: $S3_BUCKET_NAME"
        aws s3 rb "s3://$S3_BUCKET_NAME"
        print_success "S3 bucket deleted"
    else
        print_info "S3 bucket not found"
    fi
}

# Delete local configuration files
cleanup_local_files() {
    print_header "CLEANING UP LOCAL FILES"
    
    if [ -f "Lab2S3Policy.json" ]; then
        rm Lab2S3Policy.json
        print_success "Removed Lab2S3Policy.json"
    fi
    
    if [ -f "lab2-config.txt" ]; then
        rm lab2-config.txt
        print_success "Removed lab2-config.txt"
    fi
}

# Main execution
main() {
    print_header "LAB 2.1 CLEANUP: REMOVING ALL RESOURCES"
    
    load_config
    terminate_instances
    delete_key_pair
    delete_security_group
    delete_iam_policy
    delete_iam_role
    delete_s3_bucket
    cleanup_local_files
    
    print_header "CLEANUP COMPLETE"
    echo "All resources have been deleted successfully!"
}

# Confirmation prompt
read -p "This will delete all Lab 2.1 resources. Are you sure? (yes/no): " confirmation
if [ "$confirmation" != "yes" ]; then
    print_error "Cleanup cancelled"
    exit 0
fi

main
