package com.group35.d7078;

import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.model.CreateBucketRequest;
import software.amazon.awssdk.services.s3.model.CreateBucketConfiguration;
import software.amazon.awssdk.services.s3.model.CreateBucketResponse;
import software.amazon.awssdk.services.s3.model.S3Exception;

public class CreateS3Buckets {
    
    public static void main(String[] args) {
        System.out.println("==================================================");
        System.out.println("AWS S3 Bucket Creation - Multiple Regions");
        System.out.println("==================================================\n");
        
        // Define the three regions
        String[] regions = {"eu-north-1", "eu-west-1", "eu-central-1"};
        String[] regionNames = {"Stockholm", "Ireland", "Frankfurt"};
        
        // Create buckets in each region
        for (int i = 0; i < regions.length; i++) {
            String region = regions[i];
            String regionName = regionNames[i];
            
            System.out.println("Creating bucket in: " + regionName + " (" + region + ")");
            createBucketInRegion(region, regionName);
            System.out.println();
        }
        
        System.out.println("==================================================");
        System.out.println("All buckets created successfully!");
        System.out.println("==================================================");
    }
    
    public static void createBucketInRegion(String regionString, String regionName) {
        // Convert string region to Region enum
        Region region = Region.of(regionString);
        
        // Create bucket name (must be globally unique)
        String bucketName = "group35-d7078-bucket-" + regionString.toLowerCase();
        
        try {
            // Create S3 client for the specific region
            S3Client s3Client = S3Client.builder()
                .region(region)
                .build();
            
            System.out.println("S3Client created for region: " + region);
            System.out.println("Bucket Name: " + bucketName);
            
            // Create the CreateBucketRequest
            CreateBucketRequest.Builder requestBuilder = CreateBucketRequest.builder()
                .bucket(bucketName);
            
            // For regions other than us-east-1, we need to specify CreateBucketConfiguration
            if (!regionString.equals("us-east-1")) {
                CreateBucketConfiguration bucketConfig = CreateBucketConfiguration.builder()
                    .locationConstraint(region.id())
                    .build();
                requestBuilder.createBucketConfiguration(bucketConfig);
            }
            
            CreateBucketRequest createBucketRequest = requestBuilder.build();
            
            // Execute the create bucket request
            CreateBucketResponse response = s3Client.createBucket(createBucketRequest);
            
            System.out.println("✓ Bucket created successfully!");
            System.out.println("  Location: " + response.location());
            System.out.println("  Response Code: " + response.sdkHttpResponse().statusCode());
            
            // Close the S3 client
            s3Client.close();
            System.out.println("  S3Client closed.");
            
        } catch (S3Exception e) {
            // Handle bucket already exists error
            if (e.awsErrorDetails().errorCode().equals("BucketAlreadyOwnedByYou")) {
                System.out.println("✗ Bucket already exists (owned by you): " + bucketName);
            } else if (e.awsErrorDetails().errorCode().equals("BucketAlreadyExists")) {
                System.out.println("✗ Bucket name already taken by another user: " + bucketName);
            } else {
                System.out.println("✗ Error creating bucket: " + e.getMessage());
                System.out.println("  Error Code: " + e.awsErrorDetails().errorCode());
            }
        } catch (Exception e) {
            System.out.println("✗ Unexpected error: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
