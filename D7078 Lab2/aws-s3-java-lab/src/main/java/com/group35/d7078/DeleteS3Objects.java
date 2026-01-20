package com.group35.d7078;

import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.model.DeleteObjectRequest;
import software.amazon.awssdk.services.s3.model.DeleteObjectResponse;
import software.amazon.awssdk.services.s3.model.ListObjectsV2Request;
import software.amazon.awssdk.services.s3.model.ListObjectsV2Response;
import software.amazon.awssdk.services.s3.model.S3Object;
import software.amazon.awssdk.services.s3.model.S3Exception;
import java.util.List;

public class DeleteS3Objects {
    
    public static void main(String[] args) {
        System.out.println("==================================================");
        System.out.println("AWS S3 - Delete Objects");
        System.out.println("==================================================\n");
        
        // S3 bucket and region
        String bucketName = "group35-d7078-bucket-eu-north-1";
        Region region = Region.EU_NORTH_1;
        
        // Objects to delete
        String[] objectsToDelete = {
            "test.txt",
            "sample-document.txt"
        };
        
        // Create S3 client
        S3Client s3Client = S3Client.builder()
            .region(region)
            .build();
        
        try {
            System.out.println("Deleting objects from bucket: " + bucketName);
            System.out.println("Region: " + region);
            System.out.println();
            
            // List objects before deletion
            System.out.println("Objects in bucket BEFORE deletion:");
            listObjects(s3Client, bucketName);
            
            System.out.println();
            System.out.println("==================================================");
            System.out.println("Deleting specific objects...");
            System.out.println("==================================================\n");
            
            // Delete each object
            for (String objectKey : objectsToDelete) {
                deleteObjectFromS3(s3Client, bucketName, objectKey);
            }
            
            System.out.println();
            System.out.println("==================================================");
            System.out.println("Objects in bucket AFTER deletion:");
            System.out.println("==================================================");
            listObjects(s3Client, bucketName);
            
            System.out.println();
            System.out.println("==================================================");
            System.out.println("✓ All deletions completed!");
            System.out.println("==================================================");
            
        } catch (S3Exception e) {
            System.out.println("✗ S3 Error during deletion: " + e.getMessage());
            System.out.println("  Error Code: " + e.awsErrorDetails().errorCode());
        } catch (Exception e) {
            System.out.println("✗ Unexpected error: " + e.getMessage());
            e.printStackTrace();
        } finally {
            s3Client.close();
            System.out.println("\n✓ S3Client closed.");
        }
    }
    
    public static void deleteObjectFromS3(S3Client s3Client, String bucketName, String objectKey) {
        try {
            System.out.println("Deleting: " + objectKey);
            
            // Create DeleteObjectRequest
            DeleteObjectRequest deleteObjectRequest = DeleteObjectRequest.builder()
                .bucket(bucketName)
                .key(objectKey)
                .build();
            
            // Delete the object
            DeleteObjectResponse response = s3Client.deleteObject(deleteObjectRequest);
            
            System.out.println("  ✓ Deletion successful!");
            System.out.println("    Response Code: " + response.sdkHttpResponse().statusCode());
            System.out.println("    Delete Marker: " + response.deleteMarker());
            System.out.println();
            
        } catch (S3Exception e) {
            System.out.println("  ✗ S3 Error: " + e.getMessage());
            System.out.println("    Error Code: " + e.awsErrorDetails().errorCode());
            System.out.println();
        } catch (Exception e) {
            System.out.println("  ✗ Error deleting object: " + e.getMessage());
            System.out.println();
        }
    }
    
    public static void listObjects(S3Client s3Client, String bucketName) {
        try {
            // Create ListObjectsV2Request
            ListObjectsV2Request listObjectsRequest = ListObjectsV2Request.builder()
                .bucket(bucketName)
                .build();
            
            // List objects
            ListObjectsV2Response listObjectsResponse = s3Client.listObjectsV2(listObjectsRequest);
            
            // Get list of objects
            List<S3Object> objects = listObjectsResponse.contents();
            
            if (objects == null || objects.isEmpty()) {
                System.out.println("No objects found in bucket: " + bucketName);
                return;
            }
            
            System.out.println("Total objects in bucket: " + objects.size());
            System.out.println();
            
            // Display each object
            for (int i = 0; i < objects.size(); i++) {
                S3Object s3Object = objects.get(i);
                
                System.out.println((i + 1) + ". Object Key: " + s3Object.key());
                System.out.println("   Size: " + formatFileSize(s3Object.size()));
                System.out.println("   Last Modified: " + s3Object.lastModified());
                System.out.println("   ETag: " + s3Object.eTag());
                System.out.println();
            }
            
        } catch (S3Exception e) {
            System.out.println("✗ S3 Error listing objects: " + e.getMessage());
            throw e;
        }
    }
    
    private static String formatFileSize(long bytes) {
        if (bytes <= 0) return "0 B";
        final String[] units = new String[]{"B", "KB", "MB", "GB", "TB"};
        int digitGroups = (int) (Math.log10(bytes) / Math.log10(1024));
        return String.format("%.2f %s", bytes / Math.pow(1024, digitGroups), units[digitGroups]);
    }
}
