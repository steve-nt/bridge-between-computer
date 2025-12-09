package com.group35.d7078;

import software.amazon.awssdk.core.sync.RequestBody;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.model.GetObjectRequest;
import software.amazon.awssdk.services.s3.model.PutObjectRequest;
import software.amazon.awssdk.services.s3.model.PutObjectResponse;
import software.amazon.awssdk.services.s3.model.S3Exception;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

public class LatencyMeasurementS3 {
    
    // Store latency results
    static class LatencyResult {
        String region;
        long uploadTime;
        long downloadTime;
        long totalTime;
        double fileSize;
        
        public LatencyResult(String region) {
            this.region = region;
        }
        
        @Override
        public String toString() {
            return String.format(
                "Region: %-20s | Upload: %6d ms | Download: %6d ms | Total: %6d ms",
                region, uploadTime, downloadTime, totalTime
            );
        }
    }
    
    public static void main(String[] args) {
        System.out.println("==================================================");
        System.out.println("AWS S3 - Upload/Download Latency Measurement");
        System.out.println("==================================================\n");
        
        // Define regions and corresponding buckets
        Map<String, String> regionBuckets = new HashMap<>();
        regionBuckets.put("eu-north-1", "group35-d7078-bucket-eu-north-1");
        regionBuckets.put("eu-west-1", "group35-d7078-bucket-eu-west-1");
        regionBuckets.put("eu-central-1", "group35-d7078-bucket-eu-central-1");
        
        // Create large test file (1 MB)
        String testFile = "/tmp/latency-test-1mb.txt";
        createLargeTestFile(testFile, 1024 * 1024); // 1 MB
        
        // Store results for comparison
        Map<String, LatencyResult> results = new HashMap<>();
        
        try {
            System.out.println("Test file: " + testFile);
            System.out.println("File size: " + formatFileSize(Files.size(Paths.get(testFile))));
            System.out.println("\nTesting across 3 regions...");
            System.out.println("==================================================\n");
            
            // Test each region
            for (Map.Entry<String, String> entry : regionBuckets.entrySet()) {
                String regionString = entry.getKey();
                String bucketName = entry.getValue();
                
                System.out.println("Testing region: " + regionString + " (Bucket: " + bucketName + ")");
                
                Region region = Region.of(regionString);
                S3Client s3Client = S3Client.builder()
                    .region(region)
                    .build();
                
                try {
                    LatencyResult result = new LatencyResult(regionString);
                    
                    // Upload test
                    result.uploadTime = uploadFileToS3(s3Client, bucketName, testFile, regionString);
                    System.out.println("  Upload time: " + result.uploadTime + " ms");
                    
                    // Download test
                    result.downloadTime = downloadFileFromS3(s3Client, bucketName, testFile, regionString);
                    System.out.println("  Download time: " + result.downloadTime + " ms");
                    
                    result.totalTime = result.uploadTime + result.downloadTime;
                    result.fileSize = Files.size(Paths.get(testFile));
                    
                    results.put(regionString, result);
                    System.out.println();
                    
                } finally {
                    s3Client.close();
                }
            }
            
            // Print comparison table
            System.out.println("==================================================");
            System.out.println("LATENCY COMPARISON RESULTS");
            System.out.println("==================================================\n");
            
            System.out.println("Detailed Results:");
            for (LatencyResult result : results.values()) {
                System.out.println(result);
            }
            
            // Find fastest and slowest
            LatencyResult fastest = results.values().stream()
                .min((a, b) -> Long.compare(a.totalTime, b.totalTime))
                .orElse(null);
            
            LatencyResult slowest = results.values().stream()
                .max((a, b) -> Long.compare(a.totalTime, b.totalTime))
                .orElse(null);
            
            System.out.println("\n==================================================");
            System.out.println("ANALYSIS:");
            System.out.println("==================================================");
            System.out.println("Fastest region: " + fastest.region + " (" + fastest.totalTime + " ms)");
            System.out.println("Slowest region: " + slowest.region + " (" + slowest.totalTime + " ms)");
            
            long difference = slowest.totalTime - fastest.totalTime;
            double percentage = (difference * 100.0) / fastest.totalTime;
            System.out.println("Difference: " + difference + " ms (" + String.format("%.2f%%", percentage) + ")");
            
            // Calculate throughput
            System.out.println("\nThroughput Analysis:");
            for (LatencyResult result : results.values()) {
                double uploadThroughput = (result.fileSize / 1024.0) / (result.uploadTime / 1000.0); // KB/s
                double downloadThroughput = (result.fileSize / 1024.0) / (result.downloadTime / 1000.0); // KB/s
                System.out.printf("%s: Upload: %.2f KB/s | Download: %.2f KB/s\n",
                    result.region, uploadThroughput, downloadThroughput);
            }
            
        } catch (Exception e) {
            System.out.println("✗ Error during latency measurement: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    public static long uploadFileToS3(S3Client s3Client, String bucketName, String filePath, String region) {
        try {
            File file = new File(filePath);
            String fileName = "latency-test-1mb-" + region + ".txt";
            
            // Measure time
            long startTime = System.currentTimeMillis();
            
            PutObjectRequest putObjectRequest = PutObjectRequest.builder()
                .bucket(bucketName)
                .key(fileName)
                .build();
            
            s3Client.putObject(putObjectRequest, RequestBody.fromFile(file));
            
            long endTime = System.currentTimeMillis();
            return endTime - startTime;
            
        } catch (S3Exception e) {
            System.out.println("  ✗ S3 Error uploading: " + e.getMessage());
            return -1;
        } catch (Exception e) {
            System.out.println("  ✗ Error uploading file: " + e.getMessage());
            return -1;
        }
    }
    
    public static long downloadFileFromS3(S3Client s3Client, String bucketName, String filePath, String region) {
        try {
            String fileName = "latency-test-1mb-" + region + ".txt";
            String downloadPath = "/tmp/downloaded-" + region + ".txt";
            
            // Measure time
            long startTime = System.currentTimeMillis();
            
            GetObjectRequest getObjectRequest = GetObjectRequest.builder()
                .bucket(bucketName)
                .key(fileName)
                .build();
            
            InputStream s3ObjectInputStream = s3Client.getObject(getObjectRequest);
            
            // Write to file
            try (FileOutputStream fileOutputStream = new FileOutputStream(downloadPath)) {
                byte[] readBuffer = new byte[1024];
                int readLen = 0;
                while ((readLen = s3ObjectInputStream.read(readBuffer)) > 0) {
                    fileOutputStream.write(readBuffer, 0, readLen);
                }
            }
            s3ObjectInputStream.close();
            
            long endTime = System.currentTimeMillis();
            return endTime - startTime;
            
        } catch (S3Exception e) {
            System.out.println("  ✗ S3 Error downloading: " + e.getMessage());
            return -1;
        } catch (Exception e) {
            System.out.println("  ✗ Error downloading file: " + e.getMessage());
            return -1;
        }
    }
    
    private static void createLargeTestFile(String filePath, long sizeInBytes) {
        try {
            Path path = Paths.get(filePath);
            
            // Check if file exists and has correct size
            if (Files.exists(path) && Files.size(path) == sizeInBytes) {
                System.out.println("Large test file already exists: " + filePath);
                return;
            }
            
            System.out.println("Creating large test file (" + formatFileSize(sizeInBytes) + ")...");
            
            // Create file with random data
            byte[] buffer = new byte[1024 * 100]; // 100 KB chunks
            for (int i = 0; i < buffer.length; i++) {
                buffer[i] = (byte) ((i % 256) - 128); // Random-ish data
            }
            
            try (FileOutputStream fos = new FileOutputStream(filePath)) {
                long written = 0;
                while (written < sizeInBytes) {
                    long toWrite = Math.min(buffer.length, sizeInBytes - written);
                    fos.write(buffer, 0, (int) toWrite);
                    written += toWrite;
                }
            }
            
            System.out.println("✓ Test file created successfully\n");
            
        } catch (Exception e) {
            System.out.println("✗ Error creating test file: " + e.getMessage());
        }
    }
    
    private static String formatFileSize(long bytes) {
        if (bytes <= 0) return "0 B";
        final String[] units = new String[]{"B", "KB", "MB", "GB", "TB"};
        int digitGroups = (int) (Math.log10(bytes) / Math.log10(1024));
        return String.format("%.2f %s", bytes / Math.pow(1024, digitGroups), units[digitGroups]);
    }
}
