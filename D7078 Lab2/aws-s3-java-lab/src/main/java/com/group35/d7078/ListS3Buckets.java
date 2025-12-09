     package com.group35.d7078;
     
     import software.amazon.awssdk.regions.Region;
     import software.amazon.awssdk.services.s3.S3Client;
     import software.amazon.awssdk.services.s3.model.ListBucketsRequest;
     import software.amazon.awssdk.services.s3.model.ListBucketsResponse;
     import software.amazon.awssdk.services.s3.model.Bucket;
     import software.amazon.awssdk.services.s3.model.S3Exception;
     import java.time.format.DateTimeFormatter;
     import java.util.List;
     
     public class ListS3Buckets {
         
         public static void main(String[] args) {
             System.out.println("==================================================");
             System.out.println("AWS S3 - List All Buckets");
             System.out.println("==================================================\n");
             
             // Create S3 client for eu-north-1 region
             Region region = Region.EU_NORTH_1;
             S3Client s3Client = S3Client.builder()
                 .region(region)
                 .build();
             
             try {
                 System.out.println("Listing buckets in region: " + region);
                 System.out.println();
                 
                 listBuckets(s3Client);
                 
             } catch (S3Exception e) {
                 System.out.println("✗ Error listing buckets: " + e.getMessage());
                 System.out.println("  Error Code: " + e.awsErrorDetails().errorCode());
             } catch (Exception e) {
                 System.out.println("✗ Unexpected error: " + e.getMessage());
                 e.printStackTrace();
             } finally {
                 s3Client.close();
                 System.out.println("\n✓ S3Client closed.");
             }
         }
         
         public static void listBuckets(S3Client s3Client) {
             try {
                 // Create list buckets request
                 ListBucketsRequest listBucketsRequest = ListBucketsRequest.builder().build();
                 
                 // Execute list buckets request
                 ListBucketsResponse listBucketsResponse = s3Client.listBuckets(listBucketsRequest);
                 
                 // Get list of buckets
                 List<Bucket> buckets = listBucketsResponse.buckets();
                 
                 if (buckets.isEmpty()) {
                     System.out.println("No buckets found in this account.");
                     return;
                 }
                 
                 System.out.println("Total buckets found: " + buckets.size());
                 System.out.println();
                 System.out.println("Bucket Details:");
                 System.out.println("==================================================");
                 
                 // Display each bucket
                 for (int i = 0; i < buckets.size(); i++) {
                     Bucket bucket = buckets.get(i);
                     
                     System.out.println((i + 1) + ". Bucket Name: " + bucket.name());
                     System.out.println("   Created: " + formatDate(bucket.creationDate()));
                     System.out.println();
                 }
                 
                 System.out.println("==================================================");
                 System.out.println("✓ Successfully listed all buckets!");
                 
             } catch (S3Exception e) {
                 System.out.println("✗ S3 Error: " + e.getMessage());
                 throw e;
             }
         }
         
         private static String formatDate(java.time.Instant instant) {
             DateTimeFormatter formatter = DateTimeFormatter.ISO_INSTANT;
             return formatter.format(instant);
         }
     }