import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.model.*;
import software.amazon.awssdk.regions.Region;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

public class S3LatencyMeasurement {
    
    private static final String[] REGIONS = {
        "us-east-1",      // N. Virginia
        "eu-west-1",      // Ireland
        "ap-southeast-1"  // Singapore
    };
    
    private static final int TEST_FILE_SIZE = 1024 * 1024;  // 1 MB
    private static final int ITERATIONS = 3;
    
    public static void main(String[] args) {
        System.out.println("=== AWS S3 Latency Measurement Across Regions ===\n");
        
        Map<String, RegionMetrics> metrics = new HashMap<>();
        
        // Test each region
        for (String region : REGIONS) {
            metrics.put(region, testRegion(region));
        }
        
        // Print results
        printResults(metrics);
    }
    
    /**
     * Test latency for a specific region
     */
    private static RegionMetrics testRegion(String regionName) {
        System.out.println("Testing Region: " + regionName);
        System.out.println("=".repeat(50));
        
        RegionMetrics metrics = new RegionMetrics(regionName);
        
        try (S3Client s3Client = S3Client.builder()
                .region(Region.of(regionName))
                .build()) {
            
            String bucketName = createTestBucket(s3Client, regionName);
            String testFile = "latency-test-" + regionName + ".bin";
            
            // Create test file
            createTestFile(testFile, TEST_FILE_SIZE);
            System.out.println("Test file created: " + testFile + " (" + TEST_FILE_SIZE / 1024 + " KB)");
            
            // Run upload tests
            System.out.println("\nUpload Tests (" + ITERATIONS + " iterations):");
            for (int i = 0; i < ITERATIONS; i++) {
                long latency = measureUpload(s3Client, bucketName, testFile, i);
                metrics.addUploadLatency(latency);
                System.out.println("  Iteration " + (i + 1) + ": " + latency + "ms");
            }
            
            // Run download tests
            System.out.println("\nDownload Tests (" + ITERATIONS + " iterations):");
            for (int i = 0; i < ITERATIONS; i++) {
                long latency = measureDownload(s3Client, bucketName, "latency-test-0", i);
                metrics.addDownloadLatency(latency);
                System.out.println("  Iteration " + (i + 1) + ": " + latency + "ms");
            }
            
            // Cleanup
            cleanupBucket(s3Client, bucketName);
            new File(testFile).delete();
            
        } catch (Exception e) {
            System.out.println("✗ Error testing region: " + e.getMessage());
        }
        
        System.out.println();
        return metrics;
    }
    
    /**
     * Create a test bucket in the specified region
     */
    private static String createTestBucket(S3Client s3Client, String region) {
        String bucketName = "latency-test-" + region + "-" + System.currentTimeMillis();
        
        try {
            CreateBucketRequest.Builder builder = CreateBucketRequest.builder()
                    .bucket(bucketName);
            
            // Add LocationConstraint for non-us-east-1 regions
            if (!region.equals("us-east-1")) {
                builder.createBucketConfiguration(CreateBucketConfiguration.builder()
                        .locationConstraint(region)
                        .build());
            }
            
            s3Client.createBucket(builder.build());
            System.out.println("Bucket created: " + bucketName);
            
        } catch (S3Exception e) {
            System.out.println("✗ Error creating bucket: " + e.awsErrorDetails().errorMessage());
        }
        
        return bucketName;
    }
    
    /**
     * Create a test file of specified size
     */
    private static void createTestFile(String filename, int size) throws IOException {
        try (FileOutputStream fos = new FileOutputStream(filename)) {
            byte[] buffer = new byte[1024];
            int remaining = size;
            
            while (remaining > 0) {
                int toWrite = Math.min(buffer.length, remaining);
                fos.write(buffer, 0, toWrite);
                remaining -= toWrite;
            }
        }
    }
    
    /**
     * Measure upload latency
     */
    private static long measureUpload(S3Client s3Client, String bucketName, 
                                      String filePath, int iteration) {
        try {
            String objectKey = new File(filePath).getName().replace(".bin", "") + "-" + iteration;
            
            long startTime = System.currentTimeMillis();
            
            PutObjectRequest putRequest = PutObjectRequest.builder()
                    .bucket(bucketName)
                    .key(objectKey)
                    .build();
            
            s3Client.putObject(putRequest, 
                    software.amazon.awssdk.core.sync.RequestBody.fromFile(new File(filePath)));
            
            long endTime = System.currentTimeMillis();
            return endTime - startTime;
            
        } catch (S3Exception e) {
            System.out.println("✗ Error uploading: " + e.awsErrorDetails().errorMessage());
            return -1;
        }
    }
    
    /**
     * Measure download latency
     */
    private static long measureDownload(S3Client s3Client, String bucketName, 
                                        String objectKey, int iteration) {
        try {
            String downloadPath = "temp-download-" + iteration + ".bin";
            
            long startTime = System.currentTimeMillis();
            
            GetObjectRequest getRequest = GetObjectRequest.builder()
                    .bucket(bucketName)
                    .key(objectKey)
                    .build();
            
            s3Client.getObject(getRequest, 
                    software.amazon.awssdk.core.sync.ResponseTransformer.toFile(
                            new File(downloadPath)));
            
            long endTime = System.currentTimeMillis();
            
            // Cleanup downloaded file
            new File(downloadPath).delete();
            
            return endTime - startTime;
            
        } catch (S3Exception e) {
            System.out.println("✗ Error downloading: " + e.awsErrorDetails().errorMessage());
            return -1;
        }
    }
    
    /**
     * Delete all objects and bucket
     */
    private static void cleanupBucket(S3Client s3Client, String bucketName) {
        try {
            // Delete all objects
            ListObjectsV2Request listRequest = ListObjectsV2Request.builder()
                    .bucket(bucketName)
                    .build();
            
            ListObjectsV2Response listResponse = s3Client.listObjectsV2(listRequest);
            
            for (S3Object obj : listResponse.contents()) {
                DeleteObjectRequest deleteRequest = DeleteObjectRequest.builder()
                        .bucket(bucketName)
                        .key(obj.key())
                        .build();
                s3Client.deleteObject(deleteRequest);
            }
            
            // Delete bucket
            DeleteBucketRequest deleteBucketRequest = DeleteBucketRequest.builder()
                    .bucket(bucketName)
                    .build();
            s3Client.deleteBucket(deleteBucketRequest);
            
        } catch (S3Exception e) {
            System.out.println("✗ Error cleaning up: " + e.awsErrorDetails().errorMessage());
        }
    }
    
    /**
     * Print results in a formatted table
     */
    private static void printResults(Map<String, RegionMetrics> metrics) {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("LATENCY MEASUREMENT RESULTS");
        System.out.println("=".repeat(80));
        
        System.out.printf("%-20s %-20s %-20s %-20s%n", 
                "Region", "Avg Upload (ms)", "Avg Download (ms)", "Avg Total (ms)");
        System.out.println("-".repeat(80));
        
        for (String region : REGIONS) {
            RegionMetrics m = metrics.get(region);
            System.out.printf("%-20s %-20.2f %-20.2f %-20.2f%n",
                    region,
                    m.getAverageUploadLatency(),
                    m.getAverageDownloadLatency(),
                    m.getAverageUploadLatency() + m.getAverageDownloadLatency());
        }
        
        System.out.println("=".repeat(80));
        System.out.println("\nAnalysis:");
        
        // Find fastest and slowest
        String fastest = metrics.entrySet().stream()
                .min((e1, e2) -> Double.compare(
                        e1.getValue().getAverageUploadLatency() + 
                        e1.getValue().getAverageDownloadLatency(),
                        e2.getValue().getAverageUploadLatency() + 
                        e2.getValue().getAverageDownloadLatency()))
                .map(Map.Entry::getKey)
                .orElse("Unknown");
        
        System.out.println("Fastest Region: " + fastest);
        System.out.println("\nNote: Latency varies based on network conditions and AWS resources.");
    }
    
    /**
     * Inner class to store region metrics
     */
    static class RegionMetrics {
        private String region;
        private long[] uploadLatencies = new long[ITERATIONS];
        private long[] downloadLatencies = new long[ITERATIONS];
        private int uploadIndex = 0;
        private int downloadIndex = 0;
        
        RegionMetrics(String region) {
            this.region = region;
        }
        
        void addUploadLatency(long latency) {
            if (uploadIndex < ITERATIONS) {
                uploadLatencies[uploadIndex++] = latency;
            }
        }
        
        void addDownloadLatency(long latency) {
            if (downloadIndex < ITERATIONS) {
                downloadLatencies[downloadIndex++] = latency;
            }
        }
        
        double getAverageUploadLatency() {
            long sum = 0;
            for (long latency : uploadLatencies) {
                sum += latency;
            }
            return sum / (double) ITERATIONS;
        }
        
        double getAverageDownloadLatency() {
            long sum = 0;
            for (long latency : downloadLatencies) {
                sum += latency;
            }
            return sum / (double) ITERATIONS;
        }
    }
}
