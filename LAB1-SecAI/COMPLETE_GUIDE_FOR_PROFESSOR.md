# Complete Lab 1 Guide - Explaining Every Line of Code

## Introduction
This guide explains every line of code in Lab 1 (Part I and II), organized by task with detailed explanations and results.

---

# PART I: FUNDAMENTALS OF MACHINE LEARNING

## Task 1.1: Data Preprocessing and Normalization

### What is happening?
We prepare MNIST data by applying mathematical transformations to make it suitable for machine learning algorithms.

### Code Line-by-Line:

```python
# Step 1: Import and load data
import numpy as np
from matplotlib import pyplot as plt
from keras.datasets import mnist

# Load MNIST dataset
(Xtr, Ltr), (X_test, L_test) = mnist.load_data()
```
**Explanation:**
- `Xtr` (training images): 60,000 images, each 28×28 pixels → shape (60000, 28, 28)
- `Ltr` (training labels): 60,000 labels (digits 0-9)
- `X_test`, `L_test`: 10,000 test images and labels

```python
# Visualize first test image
ImageTest = X_test[0, :, :]  # Extract first image (28×28)
Label = L_test[0]             # Get its label
plt.imshow(ImageTest)         # Display the image
```

### Preprocessing Functions:

```python
def center(X):
    """MEAN NORMALIZATION"""
    # Subtract pixel-wise mean from all images
    # For each pixel position: subtract average value across all images
    newX = X - np.mean(X, axis=0)
    return newX
```
**What it does:**
- Computes mean pixel value across all images: `np.mean(X, axis=0)` shape (28, 28)
- Subtracts this mean from every image
- **Result:** Pixel values are centered around 0 instead of [0-255]
- **Why:** Removes overall brightness bias, focuses on structure

```python
def standardize(X):
    """STANDARDIZATION"""
    # First center the data
    centered = center(X)
    # Then divide by standard deviation
    newX = centered / np.std(X, axis=0)
    return newX
```
**What it does:**
- Centers data (mean = 0)
- Divides by standard deviation (std = 1)
- **Result:** All features have similar scale
- **Problem:** Zero standard deviation pixels cause division by zero
- **Why:** Makes large and small pixel variations equally important

### Visualization:
```python
X_testCentered = center(X_test)
fig, ax = plt.subplots(1, 2, figsize=(15,15))
# Display original vs centered image side-by-side
```

---

## Task 1.2: Testing 1-NN with L1 Distance

### Code:

```python
# Prepare training data
num_sample = 500
Tr_set = Xtr[:num_sample, :, :]      # Take first 500 training images
Ltr_set = Ltr[:num_sample]             # Take first 500 labels

# Flatten images from (500, 28, 28) to (500, 784)
Tr_set = Tr_set.reshape(num_sample, 28*28)
```
**Why reshape?**
- k-NN works with vectors, not 2D images
- 28×28 = 784 pixels becomes single vector

```python
def predict(X):
    """1-NN CLASSIFIER WITH L1 DISTANCE (Manhattan)"""
    num_test = X.shape[0]                           # Number of test images
    Lpred = np.zeros(num_test, dtype=Ltr_set.dtype) # Array for predictions
    
    for i in range(num_test):
        # Calculate L1 distance to ALL training images
        # Formula: distance = sum(|x_test[i] - x_train[j]|) for all pixels
        distances = np.sum(np.abs(Tr_set - X[i, :]), axis=1)
        
        # Find training image with smallest distance (closest neighbor)
        min_index = np.argmin(distances)
        
        # Predict the label of that closest neighbor
        Lpred[i] = Ltr_set[min_index]
    
    return Lpred
```

**Step-by-step example:**
- For test image i (shape 784,)
- `Tr_set - X[i, :]` broadcasts to (500, 784) - differences
- `np.abs()` takes absolute values
- `np.sum(..., axis=1)` sums across 784 pixels for each training image → (500,)
- Result: 500 distances, find minimum

```python
# Test without normalization
Test_images = X_test.reshape(10000, 28*28)
Labels_predicted = predict(Test_images)

# Calculate accuracy
accuracy = np.mean(Labels_predicted == L_test)
print(f"Accuracy L1: {accuracy}")  # Output: 0.2649 (26.49%) - VERY LOW!
```

---

## Task 1.3: L2 Distance Modification

```python
def predictL2(X):
    """1-NN WITH L2 DISTANCE (Euclidean)"""
    num_test = X.shape[0]
    Lpred = np.zeros(num_test, dtype=Ltr_set.dtype)
    
    for i in range(num_test):
        # L2 Distance formula: sqrt(sum((x_test - x_train)^2))
        distances = np.sqrt(np.sum(np.square(Tr_set - X[i, :]), axis=1))
        
        min_index = np.argmin(distances)
        Lpred[i] = Ltr_set[min_index]
    
    return Lpred

# Test L2
Labels_predictedL2 = predictL2(Test_images)
accuracy_L2 = np.mean(Labels_predictedL2 == L_test)
print(f"Accuracy L2: {accuracy_L2}")  # Output: 0.19 (19%) - EVEN WORSE!
```

**Why L2 worse than L1?**
- L2 squares differences → large differences matter more
- Raw pixels [0-255] have large outliers (dark vs bright backgrounds)
- L2 is more sensitive to these outliers

**Comparison:**
| Metric | Accuracy | Why |
|--------|----------|-----|
| L1 (unnormalized) | 26.49% | Dominated by brightness |
| L2 (unnormalized) | 19% | Even more sensitive to brightness |

---

## Task 1.4: THE BUG FIX - Normalization

### The Problem:
Raw pixel values [0-255] are huge. Distance between images is dominated by overall brightness, not structure.

### The Solution:
```python
# THIS IS THE KEY FIX!
Tr_set = Tr_set / 255.0  # Scale pixels to [0, 1]

# Now test again with normalized data
Labels_predicted_normL1 = predict(Test_images / 255.0)
Labels_predicted_normL2 = predictL2(Test_images / 255.0)

# Calculate accuracies
accuracy_norm_L1 = np.mean(Labels_predicted_normL1 == L_test)
accuracy_norm_L2 = np.mean(Labels_predicted_normL2 == L_test)

print(f"Accuracy L1 (normalized):   {accuracy_norm_L1:.4f}")  # 0.8110 (81.10%)
print(f"Accuracy L2 (normalized):   {accuracy_norm_L2:.4f}")  # 0.8294 (82.94%)
```

### Impact:
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| L1 | 26.49% | 81.10% | +54.61 points (3.06x) |
| L2 | 19.00% | 82.94% | +63.94 points (4.37x) |

### Why This Works:
- **Before:** Pixel value differences [0, 255] dominated by brightness
- **After:** Pixel value differences [0, 1] scale all pixels equally
- Each pixel contributes equally to distance calculation
- Algorithm can now focus on shape structure instead of brightness

---

## Task 1.5: k-NN Implementation

```python
def predictL2KNN(X, k):
    """K-NEAREST NEIGHBORS CLASSIFIER"""
    num_test = X.shape[0]
    Lpred = np.zeros(num_test, dtype=Ltr_set.dtype)
    
    for i in range(num_test):
        # Calculate L2 distance to all training images
        distances = np.sqrt(np.sum(np.square(Tr_set - X[i, :]), axis=1))
        
        # Find k closest neighbors
        # np.argsort(distances) returns indices that would sort the array
        # [:k] takes the first k indices (smallest distances)
        knn_indices = np.argsort(distances)[:k]
        
        # Get labels of k nearest neighbors
        knn_labels = Ltr_set[knn_indices]
        
        # Majority voting: find most common label
        from collections import Counter
        vote = Counter(knn_labels).most_common(1)  # Returns [(label, count)]
        Lpred[i] = vote[0][0]  # Extract just the label
    
    return Lpred

# Test different k values
Labels_predicted_1NN = predictL2KNN(Test_images / 255.0, k=1)
Labels_predicted_3NN = predictL2KNN(Test_images / 255.0, k=3)
Labels_predicted_5NN = predictL2KNN(Test_images / 255.0, k=5)

accuracy_1NN = np.mean(Labels_predicted_1NN == L_test)
accuracy_3NN = np.mean(Labels_predicted_3NN == L_test)
accuracy_5NN = np.mean(Labels_predicted_5NN == L_test)

print(f"Accuracy 1-NN: {accuracy_1NN:.4f}")  # 0.8294
print(f"Accuracy 3-NN: {accuracy_3NN:.4f}")  # 0.8189 (↓ 1.05%)
print(f"Accuracy 5-NN: {accuracy_5NN:.4f}")  # 0.8092 (↓ 2.02%)
```

### Results:
| k | Accuracy | Change |
|---|----------|--------|
| 1 | 82.94% | - |
| 3 | 81.89% | -1.05% |
| 5 | 80.92% | -2.02% |

### Interpretation:
**Why does accuracy decrease with larger k?**
- MNIST is a clean dataset
- Single closest neighbor is usually correct
- Adding k=3 or k=5 includes "distractor" images from wrong classes
- Voting gets confused instead of improved
- **Lesson:** More neighbors doesn't always help; depends on data distribution

---

# PART II: ARTIFICIAL NEURAL NETWORKS AND BACKPROPAGATION

## Task 2: Neural Network Implementation

### Overview
Build a multi-layer neural network that learns to classify MNIST through gradient descent.

### Network Architecture:
```
Input Layer    (784 neurons - pixels)
    ↓
Hidden Layer 1 (100 neurons, sigmoid activation)
    ↓
Hidden Layer 2 (100 neurons, sigmoid activation)
    ↓
Output Layer   (10 neurons, softmax activation)
```

### Section 2.1: Activation Functions

#### Sigmoid Function:
```python
def f_sigmoid(X, deriv=False):
    """
    SIGMOID ACTIVATION FUNCTION
    f(x) = 1 / (1 + e^(-x))
    """
    if not deriv:
        return 1 / (1 + np.exp(-X))
    else:
        # Derivative: f'(x) = f(x) * (1 - f(x))
        return f_sigmoid(X) * (1 - f_sigmoid(X))
```

**What it does:**
- Maps any value to [0, 1] range
- Non-linear function (crucial!)
- Derivative needed for backpropagation

**Why sigmoid?**
- Smooth curve (differentiable everywhere)
- Squashes large values
- Without it: stacking layers is just linear regression

#### Softmax Function:
```python
def f_softmax(X):
    """
    SOFTMAX ACTIVATION (output layer)
    Converts outputs to probability distribution
    """
    # Numerical stability: subtract max before exp
    X_stable = X - np.max(X, axis=1, keepdims=True)
    exp_X = np.exp(X_stable)
    Z = np.sum(exp_X, axis=1).reshape(-1, 1)
    return exp_X / Z
```

**What it does:**
- Converts 10 outputs to probabilities that sum to 1
- softmax(x_i) = e^(x_i) / sum(e^(x_j))
- Result[i] = probability of class i

---

### Section 2.2: Layer Class

```python
class Layer:
    def __init__(self, size, batch_size, is_input=False, is_output=False, 
                 activation=f_sigmoid):
        """Initialize a single neural network layer"""
        
        # Z: Activation values from this layer
        self.Z = np.zeros((batch_size, size[0]))
        
        # W: Weight matrix (connects to next layer)
        # shape: (input_size, output_size)
        self.W = None
        if not is_output:
            self.W = np.random.normal(size=size, scale=1E-4)
        
        # S: Pre-activation values (before applying activation function)
        self.S = np.zeros((batch_size, size[0])) if not is_input else None
        
        # D: Delta (gradient) for this layer during backprop
        self.D = np.zeros((batch_size, size[0]))
        
        # Fp: Derivative of activation function (for backprop)
        self.Fp = np.zeros((size[0], batch_size)) if not (is_input or is_output) else None
```

**Key Variables:**
- `Z`: Output of this layer
- `S`: Input to activation function (pre-activation)
- `W`: Weights going to next layer
- `D`: Gradient for this layer
- `Fp`: Derivative of activation

```python
def forward_propagate(self):
    """Compute output of this layer"""
    
    if self.is_input:
        # Input: just matrix multiplication, no activation
        return self.Z.dot(self.W)
    
    # Apply activation function
    self.Z = self.activation(self.S)
    
    if self.is_output:
        # Output layer: return softmax directly
        return self.Z
    else:
        # Hidden layer: add bias term
        self.Z = np.append(self.Z, np.ones((self.Z.shape[0], 1)), axis=1)
        # Compute activation derivative for backprop
        self.Fp = self.activation(self.S, deriv=True).T
        return self.Z.dot(self.W)
```

---

### Section 2.3: MultiLayerPerceptron Class

```python
class MultiLayerPerceptron:
    def __init__(self, layer_config, batch_size=100):
        """
        Create neural network
        layer_config: [784, 100, 100, 10]
        Creates layers and initializes weights
        """
        self.layers = []
        self.num_layers = len(layer_config)
        
        for i in range(self.num_layers - 1):
            if i == 0:
                # Input layer (784 + 1 bias → 100)
                self.layers.append(Layer([layer_config[i]+1, layer_config[i+1]],
                                        batch_size, is_input=True))
            else:
                # Hidden layer (neurons + 1 bias → next_size)
                self.layers.append(Layer([layer_config[i]+1, layer_config[i+1]],
                                        batch_size, activation=f_sigmoid))
        
        # Output layer (10 neurons, no next layer)
        self.layers.append(Layer([layer_config[-1], None], batch_size,
                                is_output=True, activation=f_softmax))
```

#### Forward Propagation:
```python
def forward_propagate(self, data):
    """
    FORWARD PASS: Compute predictions
    Input → Layer1 → Layer2 → ... → Output
    """
    # Add bias to input
    self.layers[0].Z = np.append(data, np.ones((data.shape[0], 1)), axis=1)
    
    # Forward through each layer
    for i in range(self.num_layers - 1):
        self.layers[i+1].S = self.layers[i].forward_propagate()
    
    # Return final predictions (softmax probabilities)
    return self.layers[-1].forward_propagate()
```

#### Backpropagation:
```python
def backpropagate(self, yhat, labels):
    """
    BACKPROPAGATION: Compute gradients for all layers
    
    This is THE algorithm that makes neural networks learn!
    
    Process:
    1. Compute error at output: delta = yhat - labels
    2. Propagate back: delta_i = (W^T @ delta_next) * f'(S_i)
    3. These deltas tell us how to update weights
    """
    
    # Compute error at output layer
    self.layers[-1].D = (yhat - labels).T
    
    # Backpropagate through hidden layers
    for i in range(self.num_layers - 2, 0, -1):
        # Remove bias column from weight matrix
        W_nobias = self.layers[i].W[0:-1, :]
        
        # Compute delta using chain rule:
        # delta_i = (W^T @ delta_next) * f'(S_i)
        self.layers[i].D = W_nobias.dot(self.layers[i+1].D) * self.layers[i].Fp
```

**Step-by-step explanation:**
- Output delta: `yhat - labels` (difference between prediction and truth)
- Hidden delta: `(W^T @ delta_next) * activation'(S)`
  - `W^T @ delta_next`: How much error from next layer affects this layer
  - `* activation'(S)`: Scale by sensitivity of activation function

#### Weight Update:
```python
def update_weights(self, eta):
    """
    GRADIENT DESCENT: Update all weights
    
    W := W - eta * gradient
    
    eta: Learning rate (e.g., 0.05)
    - Larger eta: faster learning, but might diverge
    - Smaller eta: slower learning, but more stable
    """
    for i in range(0, self.num_layers - 1):
        # Compute gradient: delta @ activation
        W_grad = -eta * (self.layers[i+1].D.dot(self.layers[i].Z)).T
        
        # Update weights
        self.layers[i].W += W_grad
```

#### Training Loop:
```python
def evaluate(self, train_data, train_labels, test_data, test_labels,
             num_epochs=70, eta=0.05, eval_train=False, eval_test=True):
    """
    Main training procedure
    
    For num_epochs times:
        For each mini-batch:
            1. Forward propagate
            2. Backpropagate  
            3. Update weights
        Report accuracy
    """
    
    for t in range(0, num_epochs):
        # Process each mini-batch
        for b_data, b_labels in zip(train_data, train_labels):
            # Forward pass
            output = self.forward_propagate(b_data)
            
            # Backward pass (compute gradients)
            self.backpropagate(output, b_labels)
            
            # Update weights (gradient descent step)
            self.update_weights(eta=eta)
        
        # Evaluate accuracy on test set
        if eval_test:
            errs = 0
            for b_data, b_labels in zip(test_data, test_labels):
                output = self.forward_propagate(b_data)
                yhat = np.argmax(output, axis=1)
                errs += np.sum(1 - b_labels[np.arange(len(b_labels)), yhat])
            
            error_rate = float(errs) / N_test
            print(f"[{t:4d}] Test error: {error_rate:.5f}")
```

---

### Section 2.4: Data Preparation

```python
def label_to_bit_vector(labels, nbits):
    """Convert class labels to one-hot encoding"""
    # Example: label=3, nbits=10
    # Output: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    bit_vector = np.zeros((labels.shape[0], nbits))
    for i in range(labels.shape[0]):
        bit_vector[i, labels[i]] = 1.0
    return bit_vector

def create_batches(data, labels, batch_size, create_bit_vector=False):
    """Split data into mini-batches"""
    # Processes data in chunks instead of all at once
    # Speeds up learning and reduces memory usage
    chunked_data = []
    chunked_labels = []
    
    for idx in range(0, len(data), batch_size):
        chunked_data.append(data[idx:idx+batch_size, :])
        if create_bit_vector:
            labels_batch = label_to_bit_vector(labels[idx:idx+batch_size], 10)
            chunked_labels.append(labels_batch)
    
    return chunked_data, chunked_labels
```

---

### Section 2.5: Execution

```python
# Load MNIST
(Xtr, Ltr), (X_test, L_test) = mnist.load_data()

# Flatten and normalize
Xtr = Xtr.reshape(60000, 784).astype('float32') / 255
X_test = X_test.reshape(10000, 784).astype('float32') / 255

# Create mini-batches
train_data, train_labels = create_batches(Xtr, Ltr, batch_size=100, 
                                         create_bit_vector=True)
test_data, test_labels = create_batches(X_test, L_test, batch_size=100,
                                       create_bit_vector=True)

# Create network: 784 → 100 → 100 → 10
mlp = MultiLayerPerceptron([784, 100, 100, 10], batch_size=100)

# Train for 70 epochs with learning rate 0.05
mlp.evaluate(train_data, train_labels, test_data, test_labels,
             num_epochs=70, eta=0.05, eval_train=True, eval_test=True)
```

---

## Training Results and Analysis

### Expected Output:
```
[   0]  Training error: 0.46320 Test error: 0.46380
[   1]  Training error: 0.07495 Test error: 0.07610
[   2]  Training error: 0.04900 Test error: 0.05370
...
[  69]  Training error: 0.00000 Test error: 0.02560
```

### Interpretation:

| Epoch | Training Error | Test Error | Observation |
|-------|----------------|-----------|------------|
| 0 | 46.32% | 46.38% | Random initialization |
| 1 | 7.50% | 7.61% | Rapid initial learning |
| 10 | 1.91% | 3.50% | Good progress |
| 40 | 0.014% | 0.0277% | Nearly perfect on train |
| 69 | 0.000% | 0.0256% | 97.44% accuracy! |

### Key Insights:

1. **Rapid Initial Learning (Epochs 0-5)**
   - Error drops from 46% to ~7%
   - Network learns basic digit patterns quickly
   - Steep gradient descent slope

2. **Fine-tuning (Epochs 5-50)**
   - Error decreases gradually
   - Network refines learned features
   - Gentler gradient descent

3. **Convergence (Epochs 50+)**
   - Training error approaches 0%
   - Test error plateaus ~2.5%
   - Overfitting begins: memorizing training data

### Overfitting Analysis:
```
Training Accuracy: 100% (error = 0%)
Test Accuracy: 97.44% (error = 2.56%)
Overfitting Gap: 2.56 percentage points
```

The network perfectly memorized training data but doesn't generalize perfectly. This is normal and expected!

---

## Comparison: k-NN vs Neural Networks

| Property | k-NN | Neural Network |
|----------|------|----------------|
| Best Accuracy | 82.94% | 97.44% |
| Prediction Speed | O(N) per sample | O(1) per sample |
| Training Time | O(1) - no training | O(N*E) - E epochs |
| Interpretability | Clear | Black box |
| Scalability | Fails on big data | Scales well |
| Learning | Memorization | True learning |

---

## Learning Rate Effect (Task 2.3)

### Testing Different Learning Rates:

```python
# eta = 0.005 (too small)
mlp.evaluate(..., eta=0.005, num_epochs=70)
# Result: Very slow learning, needs 100+ epochs

# eta = 0.05 (good)
mlp.evaluate(..., eta=0.05, num_epochs=70)
# Result: Converges nicely in ~50 epochs ✓

# eta = 0.5 (too large)
mlp.evaluate(..., eta=0.5, num_epochs=70)
# Result: Diverges, loss increases, might oscillate
```

**Rule of Thumb:**
- Small eta (0.005): Slow but stable
- Medium eta (0.05): Good balance
- Large eta (0.5): Fast but unstable

---

## Extension: ReLU Activation (Task 2.4)

To replace sigmoid with ReLU (Rectified Linear Unit):

```python
def f_relu(X, deriv=False):
    """ReLU Activation: f(x) = max(0, x)"""
    if not deriv:
        return np.maximum(0, X)
    else:
        # Derivative: 1 if x > 0, else 0
        return (X > 0).astype(float)

# Use in hidden layers instead of sigmoid
self.layers.append(Layer(..., activation=f_relu))
```

---

## Key Takeaways for Your Professor

### Part I: k-Nearest Neighbors
1. **Data Preprocessing is Critical**
   - Normalization increased accuracy 3x (26% → 81%)
   - Raw features can be dominated by scale

2. **Distance Metrics Matter**
   - L1 better than L2 on unnormalized data
   - L2 slightly better after normalization

3. **Hyperparameter Tuning**
   - k-NN: k=1 best, k>1 adds confusion
   - Context-dependent (data distribution matters)

4. **Limitations**
   - O(N) prediction complexity (slow for large data)
   - Purely memorization-based
   - Requires lots of memory

### Part II: Neural Networks & Backpropagation
1. **Backpropagation is the Core Algorithm**
   - Efficiently computes gradients using chain rule
   - Enables learning in multi-layer networks

2. **Architecture Matters**
   - Multiple layers learn hierarchical features
   - Activation functions essential for non-linearity
   - Output layer needs different activation (softmax)

3. **Learning Process**
   - Gradient descent moves weights to minimize error
   - Learning rate controls convergence speed
   - Mini-batch updates accelerate learning

4. **Superior Performance**
   - 97.44% accuracy vs 82.94% for k-NN
   - Fast prediction (independent of training size)
   - True learning (generalizes beyond training data)

5. **Overfitting Phenomenon**
   - Network memorizes training data perfectly
   - Test accuracy shows real generalization
   - Gap between train/test indicates overfitting

---

## For Presentation to Professor

**Emphasize:**
1. How each line implements the algorithm
2. Why preprocessing matters (Part I)
3. How backpropagation works (Part II)
4. Trade-offs between approaches
5. Results showing neural networks' superiority
