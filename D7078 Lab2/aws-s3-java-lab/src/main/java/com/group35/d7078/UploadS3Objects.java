package com.group35.d7078;

import software.amazon.awssdk.core.sync.RequestBody;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.model.PutObjectRequest;
import software.amazon.awssdk.services.s3.model.PutObjectResponse;
import software.amazon.awssdk.services.s3.model.S3Exception;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class UploadS3Objects {
    
    public static void main(String[] args) {
        System.out.println("==================================================");
        System.out.println("AWS S3 - Upload Objects");
        System.out.println("==================================================\n");
        
        // S3 bucket and region
        String bucketName = "group35-d7078-bucket-eu-north-1";
        Region region = Region.EU_NORTH_1;
        
        // Files to upload
        String[] filesToUpload = {
            "/tmp/test.txt",
            "/tmp/sample-document.txt",
            "/tmp/data-file.txt"
        };
        
        // Create S3 client
        S3Client s3Client = S3Client.builder()
            .region(region)
            .build();
        
        try {
            System.out.println("Uploading objects to bucket: " + bucketName);
            System.out.println("Region: " + region);
            System.out.println();
            
            // Create sample files first
            createSampleFiles();
            
            // Upload each file
            for (String filePath : filesToUpload) {
                uploadFileToS3(s3Client, bucketName, filePath);
            }
            
            System.out.println("==================================================");
            System.out.println("✓ All files uploaded successfully!");
            System.out.println("==================================================");
            
        } catch (S3Exception e) {
            System.out.println("✗ S3 Error during upload: " + e.getMessage());
            System.out.println("  Error Code: " + e.awsErrorDetails().errorCode());
        } catch (Exception e) {
            System.out.println("✗ Unexpected error: " + e.getMessage());
            e.printStackTrace();
        } finally {
            s3Client.close();
            System.out.println("\n✓ S3Client closed.");
        }
    }
    
    public static void uploadFileToS3(S3Client s3Client, String bucketName, String filePath) {
        try {
            Path path = Paths.get(filePath);
            File file = new File(filePath);
            
            // Check if file exists
            if (!file.exists()) {
                System.out.println("✗ File not found: " + filePath);
                return;
            }
            
            String fileName = path.getFileName().toString();
            long fileSize = Files.size(path);
            
            System.out.println("Uploading: " + fileName);
            System.out.println("  File size: " + formatFileSize(fileSize));
            System.out.println("  File path: " + filePath);
            
            // Create PutObjectRequest
            PutObjectRequest putObjectRequest = PutObjectRequest.builder()
                .bucket(bucketName)
                .key(fileName)
                .build();
            
            // Upload the file
            PutObjectResponse response = s3Client.putObject(putObjectRequest, 
                RequestBody.fromFile(file));
            
            System.out.println("  ✓ Upload successful!");
            System.out.println("    ETag: " + response.eTag());
            System.out.println("    Response Code: " + response.sdkHttpResponse().statusCode());
            System.out.println();
            
        } catch (S3Exception e) {
            System.out.println("  ✗ S3 Error: " + e.getMessage());
            System.out.println("    Error Code: " + e.awsErrorDetails().errorCode());
            System.out.println();
        } catch (Exception e) {
            System.out.println("  ✗ Error uploading file: " + e.getMessage());
            System.out.println();
        }
    }
    
    private static void createSampleFiles() {
        try {
            // Create test.txt if it doesn't exist
            Path testFile = Paths.get("/tmp/test.txt");
            if (!Files.exists(testFile)) {
                Files.write(testFile, "This is a test file for S3 upload.\n".getBytes());
            }
            
            // Create sample-document.txt
            Path docFile = Paths.get("/tmp/sample-document.txt");
            if (!Files.exists(docFile)) {
                Files.write(docFile, "This is a sample document.\nIt contains multiple lines.\nGreat for testing uploads!".getBytes());
            }
            
            // Create data-file.txt
            Path dataFile = Paths.get("/tmp/data-file.txt");
            if (!Files.exists(dataFile)) {
                StringBuilder sb = new StringBuilder();
                for (int i = 1; i <= 100; i++) {
                    sb.append("Line ").append(i).append(": Sample data for S3 upload test.\n");
                }
                Files.write(dataFile, sb.toString().getBytes());
            }
            
            System.out.println("Sample files created/verified in /tmp/");
            System.out.println();
            
        } catch (Exception e) {
            System.out.println("Warning: Could not create sample files: " + e.getMessage());
        }
    }
    
    private static String formatFileSize(long bytes) {
        if (bytes <= 0) return "0 B";
        final String[] units = new String[]{"B", "KB", "MB", "GB", "TB"};
        int digitGroups = (int) (Math.log10(bytes) / Math.log10(1024));
        return String.format("%.2f %s", bytes / Math.pow(1024, digitGroups), units[digitGroups]);
    }
}
