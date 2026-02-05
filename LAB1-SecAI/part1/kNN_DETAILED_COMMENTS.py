# ==================================================================================
# k-Nearest Neighbors (k-NN) MNIST Classification - DETAILED COMMENTS
# ==================================================================================
# This notebook implements the k-Nearest Neighbor algorithm for MNIST digit 
# classification with L1 and L2 distance metrics, and extends it to k-NN voting.
# ==================================================================================

import numpy as np
from matplotlib import pyplot as plt
from collections import Counter

# Load the MNIST dataset from Keras
from keras.datasets import mnist

# Set the matplotlib style to dark background for better visualization
plt.style.use('dark_background')

# ==================================================================================
# STEP 1: LOAD AND PREPARE DATA
# ==================================================================================

# Load MNIST dataset
# Returns: training data (60,000 images of 28x28), training labels (0-9)
#          test data (10,000 images of 28x28), test labels (0-9)
(Xtr, Ltr), (X_test, L_test) = mnist.load_data()

print(f"Training data shape: {Xtr.shape}")  # (60000, 28, 28)
print(f"Test data shape: {X_test.shape}")    # (10000, 28, 28)

# ==================================================================================
# STEP 2: VISUALIZE A SAMPLE IMAGE
# ==================================================================================

# Extract first test image and its label
ImageTest = X_test[0, :, :]  # Single 28x28 image
Label = L_test[0]             # Label for that image

# Display the image
plt.title(f'Label is {Label}')
plt.imshow(ImageTest)
plt.show()
plt.close()

# ==================================================================================
# STEP 3: DATA PREPROCESSING FUNCTIONS
# ==================================================================================

# PREPROCESSING FUNCTION 1: CENTER FUNCTION (Mean Normalization)
# Purpose: Subtract the mean from each pixel to center the data around zero
# Why: Removes overall brightness bias, makes data more suitable for distance calculation
def center(X):
    # X.shape = (N, 28, 28) or (N, 784) after flattening
    # np.mean(X, axis=0) computes mean across all images for each pixel
    # This returns shape (28, 28) or (784,) containing the average value per pixel
    newX = X - np.mean(X, axis=0)  # Subtract pixel-wise mean from all images
    return newX

# PREPROCESSING FUNCTION 2: STANDARDIZE FUNCTION
# Purpose: Scale centered data by dividing by standard deviation
# Why: Ensures all features have similar ranges; prevents large pixel values from dominating
def standardize(X):
    # center(X) subtracts mean first
    # np.std(X, axis=0) calculates standard deviation for each pixel
    # Division by std scales each pixel to unit variance
    newX = center(X) / np.std(X, axis=0)  # Divide by pixel-wise standard deviation
    return newX

# Apply preprocessing and visualize
X_testCentered = center(X_test)

# Display original vs centered image
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 15))
ax0 = plt.subplot(1, 2, 1)
ax1 = plt.subplot(1, 2, 2)

ImageTestCentered = X_testCentered[0, :, :]

ax0.set_title(f'Original Image (Label: {Label})')
ax0.imshow(ImageTest)
ax1.set_title('After Mean Normalization')
ax1.imshow(ImageTestCentered)
plt.show()
plt.close()

# Apply standardization and visualize
X_testStandardized = standardize(X_test)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 15))
ax0 = plt.subplot(1, 2, 1)
ax1 = plt.subplot(1, 2, 2)

ImageTestStandardized = X_testStandardized[0, :, :]

ax0.set_title(f'Original Image (Label: {Label})')
ax0.imshow(ImageTest)
ax1.set_title('After Standardization')
ax1.imshow(ImageTestStandardized)
plt.show()
plt.close()

# ==================================================================================
# STEP 4: PREPARE DATA FOR CLASSIFICATION
# ==================================================================================

# Use only first 500 training samples (to speed up computation)
num_sample = 500
Tr_set = Xtr[:num_sample, :, :]  # Shape: (500, 28, 28)
Ltr_set = Ltr[:num_sample]        # Shape: (500,) - labels for those images

# Flatten images from (500, 28, 28) to (500, 784)
# This converts each 28x28 image into a single 784-dimensional vector
Tr_set = Tr_set.reshape(num_sample, Tr_set.shape[1] * Tr_set.shape[2])
print(f"Training set shape after flattening: {Tr_set.shape}")  # (500, 784)

# ==================================================================================
# STEP 5: IMPLEMENT 1-NN CLASSIFIER WITH L1 DISTANCE (Manhattan Distance)
# ==================================================================================

def predict(X):
    """
    1-Nearest Neighbor classifier using L1 (Manhattan) distance
    
    Algorithm:
    1. For each test image: Calculate L1 distance to ALL training images
    2. Find the training image with smallest distance (closest neighbor)
    3. Predict the label of that closest neighbor
    
    L1 Distance Formula:
    distance = sum(|x_test[i] - x_train[i]|) for all pixels i
    This is the sum of absolute differences between pixels
    
    Args:
        X: Test images, shape (N_test, 784) where N_test = number of test images
    
    Returns:
        Lpred: Predicted labels, shape (N_test,)
    """
    num_test = X.shape[0]  # Number of test images
    Lpred = np.zeros(num_test, dtype=Ltr_set.dtype)  # Array to store predictions
    
    # For each test image
    for i in range(num_test):
        # Calculate L1 distance from test image i to ALL training images
        # Tr_set - X[i,:] broadcasts to (500, 784), creating difference matrix
        # np.abs() takes absolute values
        # np.sum(..., axis=1) sums across pixels (axis=1), giving (500,) distance vector
        distances = np.sum(np.abs(Tr_set - X[i, :]), axis=1)
        
        # Find index of training image with smallest distance
        min_index = np.argmin(distances)
        
        # Predict the label of the closest training image
        Lpred[i] = Ltr_set[min_index]
    
    return Lpred

# ==================================================================================
# STEP 6: IMPLEMENT 1-NN CLASSIFIER WITH L2 DISTANCE (Euclidean Distance)
# ==================================================================================

def predictL2(X):
    """
    1-Nearest Neighbor classifier using L2 (Euclidean) distance
    
    Algorithm:
    Same as L1, but uses different distance metric
    
    L2 Distance Formula:
    distance = sqrt(sum((x_test[i] - x_train[i])^2) for all pixels i)
    This is the Euclidean distance (straight-line distance in high dimensions)
    
    Args:
        X: Test images, shape (N_test, 784)
    
    Returns:
        Lpred: Predicted labels, shape (N_test,)
    """
    num_test = X.shape[0]
    Lpred = np.zeros(num_test, dtype=Ltr_set.dtype)
    
    for i in range(num_test):
        # Calculate L2 distance (Euclidean distance)
        # np.square() squares each element: (x_test - x_train)^2
        # np.sum(..., axis=1) sums the squared differences for each training image
        # np.sqrt() takes square root to get final Euclidean distance
        distances = np.sqrt(np.sum(np.square(Tr_set - X[i, :]), axis=1))
        
        min_index = np.argmin(distances)
        Lpred[i] = Ltr_set[min_index]
    
    return Lpred

# ==================================================================================
# STEP 7: TEST CLASSIFIERS WITHOUT NORMALIZATION (THE BUG!)
# ==================================================================================

# Flatten test images to (10000, 784)
Test_images = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])

# Run predictions using raw pixel values (0-255) - NO NORMALIZATION
Labels_predicted = predict(Test_images)
Labels_predictedL2 = predictL2(Test_images)

# Calculate accuracy (fraction of correct predictions)
accuracy_L1 = np.mean(Labels_predicted == L_test)
accuracy_L2 = np.mean(Labels_predictedL2 == L_test)

print(f"Accuracy L1 (unnormalized): {accuracy_L1:.4f}")  # ~0.2649 (26.49%) - VERY LOW!
print(f"Accuracy L2 (unnormalized): {accuracy_L2:.4f}")  # ~0.1900 (19.00%) - EVEN WORSE!

# WHY SO LOW?
# With raw pixels [0-255]: distances are HUGE and dominated by brightness
# Small structural differences are overwhelmed by overall intensity differences
# The algorithm can't distinguish between similar shapes with different brightnesses

# ==================================================================================
# STEP 8: FIX THE BUG - NORMALIZE DATA BY DIVIDING BY 255
# ==================================================================================

# THIS IS THE KEY FIX!
# Normalize training data to [0, 1] range by dividing by max pixel value (255)
Tr_set = Tr_set / 255.0

# Run predictions again with normalized test data
Labels_predicted_normL1 = predict(Test_images / 255.0)  # Normalize test data too
Labels_predicted_normL2 = predictL2(Test_images / 255.0)

# Calculate accuracy with normalized data
accuracy_norm_L1 = np.mean(Labels_predicted_normL1 == L_test)
accuracy_norm_L2 = np.mean(Labels_predicted_normL2 == L_test)

print(f"\nAccuracy L1 (normalized):   {accuracy_norm_L1:.4f}")  # ~0.8110 (81.10%) - MUCH BETTER!
print(f"Accuracy L2 (normalized):   {accuracy_norm_L2:.4f}")  # ~0.8294 (82.94%) - EVEN BETTER!

# WHY THE DRAMATIC IMPROVEMENT?
# With normalized pixels [0-1]: distances are comparable across all pixels
# Each pixel contributes equally to distance calculation
# Network can now see structural similarities instead of being blinded by brightness

# ==================================================================================
# STEP 9: IMPLEMENT k-NN CLASSIFIER (Extension of 1-NN)
# ==================================================================================

def predictL2KNN(X, k):
    """
    k-Nearest Neighbors classifier using L2 (Euclidean) distance
    
    Algorithm:
    1. For each test image: Calculate L2 distance to ALL training images
    2. Find the k training images with smallest distances (k closest neighbors)
    3. Count votes from those k neighbors
    4. Predict the class with the most votes (majority voting)
    
    Why k-NN instead of 1-NN?
    - More robust to noise: single outlier can't fool it
    - However: if k is too large, different classes' data mixes and accuracy drops
    
    Args:
        X: Test images, shape (N_test, 784)
        k: Number of neighbors to consider (k=1 is standard 1-NN)
    
    Returns:
        Lpred: Predicted labels using majority voting
    """
    num_test = X.shape[0]
    Lpred = np.zeros(num_test, dtype=Ltr_set.dtype)
    
    for i in range(num_test):
        # Calculate L2 distance to all training images
        distances = np.sqrt(np.sum(np.square(Tr_set - X[i, :]), axis=1))
        
        # Find indices of k smallest distances
        # np.argsort() returns indices that would sort the array
        # [:k] takes the first k indices (smallest distances)
        knn_indices = np.argsort(distances)[:k]
        
        # Get the labels of those k nearest neighbors
        knn_labels = Ltr_set[knn_indices]
        
        # Majority voting: find the most common label among k neighbors
        # Counter(knn_labels).most_common(1) returns the most frequent label
        vote = Counter(knn_labels).most_common(1)  # Returns [(label, count)]
        Lpred[i] = vote[0][0]  # Extract just the label from the result
    
    return Lpred

# ==================================================================================
# STEP 10: TEST k-NN WITH DIFFERENT k VALUES
# ==================================================================================

# Run k-NN with k=3 and k=5
Labels_predicted_3NNL2 = predictL2KNN(Test_images / 255.0, k=3)
Labels_predicted_5NNL2 = predictL2KNN(Test_images / 255.0, k=5)

# Calculate accuracies
accuracy_1NN = np.mean(Labels_predicted_normL2 == L_test)  # k=1
accuracy_3NN = np.mean(Labels_predicted_3NNL2 == L_test)   # k=3
accuracy_5NN = np.mean(Labels_predicted_5NNL2 == L_test)   # k=5

print(f"\nAccuracy L2 1-NN: {accuracy_1NN:.4f}")  # ~0.8294 (82.94%)
print(f"Accuracy L2 3-NN: {accuracy_3NN:.4f}")  # ~0.8189 (81.89%) - LOWER!
print(f"Accuracy L2 5-NN: {accuracy_5NN:.4f}")  # ~0.8092 (80.92%) - EVEN LOWER!

# ==================================================================================
# INTERPRETATION OF RESULTS
# ==================================================================================

"""
KEY FINDINGS:

1. DATA NORMALIZATION IS CRITICAL (Task 1.4 - The Bug)
   - Without normalization (raw [0-255]): 26.49% L1, 19% L2
   - With normalization ([0-1]):         81.10% L1, 82.94% L2
   - Improvement: ~3x increase in accuracy!
   
   Why? Raw pixel values are dominated by overall brightness. Normalization 
   makes each pixel contribute equally to distance calculation.

2. L1 vs L2 DISTANCE (Task 1.3)
   - L1 (Manhattan) unnormalized: 26.49%
   - L2 (Euclidean) unnormalized: 19%
   - L2 (Euclidean) normalized:   82.94% (slightly better than L1)
   
   L1 is more robust to outliers, but L2 works better here because 
   handwritten digits have smooth intensity changes.

3. 1-NN vs k-NN (Task 1.5)
   - 1-NN:  82.94%
   - 3-NN:  81.89%  ↓ 1.05 percentage points
   - 5-NN:  80.92%  ↓ 2.02 percentage points
   
   Why does accuracy DECREASE with larger k?
   - k-NN is supposed to be more robust, but:
   - MNIST dataset is clean (well-separated classes)
   - Single closest neighbor is already an excellent match
   - Adding more neighbors introduces "distractor" images from different classes
   - For this dataset, keeping k=1 is optimal

4. COMPUTATIONAL COMPLEXITY
   - Each test image: computed distance to 500 training images
   - 10,000 test images × 500 training distances = 5,000,000 comparisons
   - This is feasible for k-NN but doesn't scale to larger datasets

5. WHEN TO USE k-NN
   - Good for: Clean, well-separated classes (like MNIST)
   - Problem: Scales poorly (O(N) per prediction)
   - Better alternatives: Neural networks, SVM (see Part II)
"""

# ==================================================================================
# SUMMARY FOR YOUR PROFESSOR
# ==================================================================================

"""
WHAT EACH CODE LINE DOES:

Preprocessing:
- center(X): Subtracts pixel-wise mean → centers data around 0
- standardize(X): Divides by std dev → scales to unit variance

Distance Metrics:
- L1: sum of |differences| (Manhattan distance)
- L2: sqrt(sum of squared differences) (Euclidean distance)

1-NN Algorithm:
1. For each test image, calculate distance to all training images
2. Find the training image with minimum distance
3. Predict that training image's label

k-NN Algorithm:
1. For each test image, calculate distances to all training images
2. Find k training images with smallest distances
3. Count votes from those k neighbors
4. Predict the class with most votes

CRITICAL BUG FIX:
- Normalization by 255 increased accuracy from 26% to 81%
- Without it, brightness dominates the distance calculation

RESULTS:
- Best: L2 1-NN on normalized data: 82.94% accuracy
- Shows that for MNIST, closest neighbor is usually correct
- k-NN works worse because it introduces confusion from distant neighbors
"""
