import software.amazon.awssdk.core.sync.ResponseTransformer;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.model.*;
import software.amazon.awssdk.regions.Region;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class S3BucketOperations {
    
    private static final String[] REGIONS = {
        "us-east-1",
        "eu-west-1",
        "ap-southeast-1"
    };
    
    public static void main(String[] args) {
        System.out.println("=== AWS S3 Bucket Operations Demo ===\n");
        
        createBucketsInRegions();
        listBuckets();
        uploadObjects();
        downloadObjects();
        listObjectsInBucket();
        deleteObjects();
    }
    
    /**
     * Create S3 buckets in three different regions
     * Demonstrates: S3Client creation, region configuration, bucket creation
     */
    public static void createBucketsInRegions() {
        System.out.println(">>> TASK 1: Creating S3 Buckets in Multiple Regions\n");
        
        for (String regionName : REGIONS) {
            try (S3Client s3Client = S3Client.builder()
                    .region(Region.of(regionName))
                    .build()) {
                
                String bucketName = "lab2-bucket-" + regionName + "-" + System.currentTimeMillis();
                
                System.out.println("Creating bucket: " + bucketName);
                System.out.println("Region: " + regionName);
                
                CreateBucketRequest createBucketRequest = CreateBucketRequest.builder()
                        .bucket(bucketName)
                        .build();
                
                s3Client.createBucket(createBucketRequest);
                System.out.println("✓ Bucket created successfully\n");
                
            } catch (S3Exception e) {
                System.out.println("✗ Error creating bucket: " + e.awsErrorDetails().errorMessage() + "\n");
            }
        }
    }
    
    /**
     * List all S3 buckets in the account
     * Demonstrates: ListBuckets operation, bucket enumeration
     */
    public static void listBuckets() {
        System.out.println(">>> TASK 2: Listing All S3 Buckets\n");
        
        try (S3Client s3Client = S3Client.builder()
                .region(Region.US_EAST_1)
                .build()) {
            
            ListBucketsResponse response = s3Client.listBuckets();
            List<Bucket> buckets = response.buckets();
            
            System.out.println("Total Buckets: " + buckets.size());
            for (Bucket bucket : buckets) {
                System.out.println("  - " + bucket.name() + 
                                 " (Created: " + bucket.creationDate() + ")");
            }
            System.out.println();
            
        } catch (S3Exception e) {
            System.out.println("✗ Error listing buckets: " + e.awsErrorDetails().errorMessage() + "\n");
        }
    }
    
    /**
     * Upload objects to S3 buckets
     * Demonstrates: PutObject operation, file upload, metadata handling
     */
    public static void uploadObjects() {
        System.out.println(">>> TASK 3: Uploading Objects to S3\n");
        
        try (S3Client s3Client = S3Client.builder()
                .region(Region.US_EAST_1)
                .build()) {
            
            String bucketName = getFirstLabBucket(s3Client);
            if (bucketName == null) {
                System.out.println("No suitable bucket found for upload\n");
                return;
            }
            
            // Upload text file
            String objectKey = "test-upload-" + System.currentTimeMillis() + ".txt";
            String fileContent = "This is a test file for Lab 2.2\nIt demonstrates S3 object upload operations.";
            
            System.out.println("Uploading object: " + objectKey);
            System.out.println("To bucket: " + bucketName);
            
            PutObjectRequest putObjectRequest = PutObjectRequest.builder()
                    .bucket(bucketName)
                    .key(objectKey)
                    .build();
            
            s3Client.putObject(putObjectRequest, 
                    software.amazon.awssdk.core.sync.RequestBody.fromString(fileContent));
            
            System.out.println("✓ Object uploaded successfully\n");
            
        } catch (S3Exception e) {
            System.out.println("✗ Error uploading object: " + e.awsErrorDetails().errorMessage() + "\n");
        }
    }
    
    /**
     * Download objects from S3 buckets
     * Demonstrates: GetObject operation, file download, latency measurement
     */
    public static void downloadObjects() {
        System.out.println(">>> TASK 4: Downloading Objects from S3\n");
        
        try (S3Client s3Client = S3Client.builder()
                .region(Region.US_EAST_1)
                .build()) {
            
            String bucketName = getFirstLabBucket(s3Client);
            if (bucketName == null) {
                System.out.println("No suitable bucket found for download\n");
                return;
            }
            
            // List objects to find one to download
            ListObjectsV2Request listRequest = ListObjectsV2Request.builder()
                    .bucket(bucketName)
                    .build();
            
            ListObjectsV2Response listResponse = s3Client.listObjectsV2(listRequest);
            
            if (listResponse.contents().isEmpty()) {
                System.out.println("No objects found in bucket for download\n");
                return;
            }
            
            String objectKey = listResponse.contents().get(0).key();
            
            System.out.println("Downloading object: " + objectKey);
            System.out.println("From bucket: " + bucketName);
            
            long startTime = System.currentTimeMillis();
            
            GetObjectRequest getObjectRequest = GetObjectRequest.builder()
                    .bucket(bucketName)
                    .key(objectKey)
                    .build();
            
            String downloadFile = "downloaded-" + objectKey;
            s3Client.getObject(getObjectRequest, 
                    ResponseTransformer.toFile(new File(downloadFile)));
            
            long endTime = System.currentTimeMillis();
            long latency = endTime - startTime;
            
            System.out.println("✓ Object downloaded successfully");
            System.out.println("Latency: " + latency + "ms");
            System.out.println("Saved to: " + downloadFile + "\n");
            
        } catch (S3Exception e) {
            System.out.println("✗ Error downloading object: " + e.awsErrorDetails().errorMessage() + "\n");
        }
    }
    
    /**
     * List all objects in a bucket
     * Demonstrates: ListObjectsV2 operation, pagination
     */
    public static void listObjectsInBucket() {
        System.out.println(">>> TASK 5: Listing Objects in Bucket\n");
        
        try (S3Client s3Client = S3Client.builder()
                .region(Region.US_EAST_1)
                .build()) {
            
            String bucketName = getFirstLabBucket(s3Client);
            if (bucketName == null) {
                System.out.println("No suitable bucket found\n");
                return;
            }
            
            System.out.println("Listing objects in bucket: " + bucketName + "\n");
            
            ListObjectsV2Request listRequest = ListObjectsV2Request.builder()
                    .bucket(bucketName)
                    .build();
            
            ListObjectsV2Response listResponse;
            int objectCount = 0;
            
            do {
                listResponse = s3Client.listObjectsV2(listRequest);
                
                for (S3Object object : listResponse.contents()) {
                    System.out.println("  - " + object.key() + 
                                     " (Size: " + object.size() + " bytes, " +
                                     "Modified: " + object.lastModified() + ")");
                    objectCount++;
                }
                
                listRequest = ListObjectsV2Request.builder()
                        .bucket(bucketName)
                        .continuationToken(listResponse.nextContinuationToken())
                        .build();
                        
            } while (listResponse.isTruncated());
            
            System.out.println("\nTotal objects: " + objectCount + "\n");
            
        } catch (S3Exception e) {
            System.out.println("✗ Error listing objects: " + e.awsErrorDetails().errorMessage() + "\n");
        }
    }
    
    /**
     * Delete objects from S3 buckets
     * Demonstrates: DeleteObject operation
     */
    public static void deleteObjects() {
        System.out.println(">>> TASK 6: Deleting Objects from S3\n");
        
        try (S3Client s3Client = S3Client.builder()
                .region(Region.US_EAST_1)
                .build()) {
            
            String bucketName = getFirstLabBucket(s3Client);
            if (bucketName == null) {
                System.out.println("No suitable bucket found for deletion\n");
                return;
            }
            
            ListObjectsV2Request listRequest = ListObjectsV2Request.builder()
                    .bucket(bucketName)
                    .build();
            
            ListObjectsV2Response listResponse = s3Client.listObjectsV2(listRequest);
            
            if (listResponse.contents().isEmpty()) {
                System.out.println("No objects to delete\n");
                return;
            }
            
            String objectKey = listResponse.contents().get(0).key();
            
            System.out.println("Deleting object: " + objectKey);
            System.out.println("From bucket: " + bucketName);
            
            DeleteObjectRequest deleteRequest = DeleteObjectRequest.builder()
                    .bucket(bucketName)
                    .key(objectKey)
                    .build();
            
            s3Client.deleteObject(deleteRequest);
            System.out.println("✓ Object deleted successfully\n");
            
        } catch (S3Exception e) {
            System.out.println("✗ Error deleting object: " + e.awsErrorDetails().errorMessage() + "\n");
        }
    }
    
    /**
     * Helper method to get the first lab2 bucket
     */
    private static String getFirstLabBucket(S3Client s3Client) {
        try {
            ListBucketsResponse response = s3Client.listBuckets();
            return response.buckets().stream()
                    .map(Bucket::name)
                    .filter(name -> name.startsWith("lab2-bucket-"))
                    .findFirst()
                    .orElse(null);
        } catch (S3Exception e) {
            return null;
        }
    }
}
