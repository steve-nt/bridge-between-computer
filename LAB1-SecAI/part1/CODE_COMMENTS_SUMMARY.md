# Lab 1 Part I - Code Comments Summary

## Overview
I have created two heavily annotated Python files with detailed explanations of every line of code in the Jupyter notebooks.

---

## File 1: kNN_DETAILED_COMMENTS.py
**Based on:** D7079E_code_kNN_task.ipynb

### What This Code Does
Implements k-Nearest Neighbor classifier for MNIST digit recognition with three main tasks:

### Structure

#### Task 1.1: Data Preprocessing
- **center(X)**: Subtracts mean from data to center around 0
  - Removes brightness bias
  - Enables better distance calculation
  
- **standardize(X)**: Divides by standard deviation
  - Scales all features to similar ranges
  - Problem: Division by zero for "dead pixels"

#### Task 1.2-1.3: Distance Metrics
- **L1 (Manhattan Distance)**: sum(|differences|)
  - Unnormalized accuracy: 26.49% (VERY LOW!)
  - More robust to outliers
  
- **L2 (Euclidean Distance)**: sqrt(sum(differences^2))
  - Unnormalized accuracy: 19% (EVEN WORSE!)
  - More sensitive to large differences

#### Task 1.4: THE BUG FIX
**Critical Discovery:** Raw pixel values (0-255) dominate distance calculation
- **Solution:** Normalize by dividing by 255
- **Result:** Accuracy jumps from 26% → 81% (3x improvement!)
- **Why:** Normalized pixels [0-1] allow each pixel to contribute equally

#### Task 1.5: k-NN Implementation
**Algorithm:**
1. For each test image, find k closest training images
2. Count votes from those k neighbors
3. Predict the class with most votes

**Results:**
- 1-NN: 82.94% accuracy (BEST)
- 3-NN: 81.89% accuracy (1% drop)
- 5-NN: 80.92% accuracy (2% drop)

**Insight:** For MNIST, single closest neighbor is usually correct. Adding more neighbors introduces confusion from distant classes.

---

## File 2: ANN_BACKPROP_DETAILED_COMMENTS.py
**Based on:** D7079E_code-ANN_backprop_task.ipynb

### What This Code Does
Implements a Multi-Layer Perceptron with backpropagation learning algorithm

### Structure

#### Section 1: Activation Functions
- **Sigmoid σ(x) = 1/(1 + e^(-x))**
  - Used in hidden layers
  - Non-linear (essential for multi-layer networks)
  - Derivative: σ'(x) = σ(x)(1-σ(x))
  
- **Softmax**
  - Used at output layer
  - Converts outputs to probabilities (sum to 1)
  - Formula: softmax(x_i) = e^(x_i) / sum(e^(x_j))

#### Section 2: Layer Class
Represents a single neural network layer with:
- **Z**: Activations (output of layer)
- **S**: Pre-activation values (before applying activation function)
- **W**: Weight matrix (connects to next layer)
- **D**: Delta (gradient used during backpropagation)
- **Fp**: Derivative of activation function

**Key Method - forward_propagate():**
- Input layer: Z @ W (just linear transformation)
- Hidden layers: activation(S), add bias term, then @ W
- Output layer: softmax(S) (probability distribution)

#### Section 3: Multi-Layer Perceptron
**Network Architecture:**
```
Input (784) → Hidden1 (100, sigmoid) → Hidden2 (100, sigmoid) → Output (10, softmax)
```

**Three Main Methods:**

1. **forward_propagate(data)**
   - Passes data through all layers
   - Computes predictions
   - Returns softmax probabilities

2. **backpropagate(yhat, labels)**
   - THE CORE LEARNING ALGORITHM
   - Computes gradients using chain rule
   - Works backwards: output → hidden1 → hidden2 → input
   - **Key equation:** delta_i = (W_next^T @ delta_next) * activation'(S_i)

3. **update_weights(eta)**
   - Applies gradient descent
   - **Update rule:** W := W - eta * gradient
   - eta (learning rate) controls step size

#### Section 4: Training Loop (evaluate method)
**For each epoch:**
1. For each mini-batch:
   - Forward propagate: compute predictions
   - Backpropagate: compute gradients
   - Update weights: gradient descent step
2. Evaluate on test data
3. Report error rates

### Results
**Final Accuracy: 97.44%** (2.56% error after 70 epochs)

**Learning Progression:**
- Epoch 0: 46% error (random initialization)
- Epoch 1: 7.6% error (fast learning)
- Epoch 50: 0.001% training error (memorization)
- Epoch 69: 0.003% training error, 2.56% test error (overfitting)

**Key Insight:** Network achieves near-perfect training accuracy but 2.56% test error shows it's memorizing data rather than learning generalizable features.

---

## Understanding the Code

### Line-by-Line Structure

#### Part I (k-NN) Code Structure:
```python
# 1. Load MNIST data
(Xtr, Ltr), (X_test, L_test) = mnist.load_data()

# 2. Flatten images from (N, 28, 28) to (N, 784)
Tr_set = Tr_set.reshape(num_sample, 28*28)

# 3. Define distance metrics
def predict(X):           # L1 distance
def predictL2(X):         # L2 distance  
def predictL2KNN(X, k):   # k-NN voting

# 4. Test without normalization (THE BUG)
accuracy_L1 = np.mean(Labels_predicted == L_test)  # 26% ❌

# 5. Fix: Normalize by 255
Tr_set = Tr_set / 255.0
accuracy_L1 = np.mean(Labels_predicted_normL1 == L_test)  # 81% ✓

# 6. Test k-NN
for k in [1, 3, 5]:
    accuracy = np.mean(Labels_predicted_KNN == L_test)
```

#### Part II (ANN) Code Structure:
```python
# 1. Define activation functions
def f_sigmoid(X, deriv=False):     # Hidden layer activation
def f_softmax(X):                  # Output layer activation

# 2. Create Layer class
class Layer:
    def forward_propagate(self):   # Compute layer output
    # Variables: Z (activation), S (pre-activation), W (weights)

# 3. Create MultiLayerPerceptron class
class MultiLayerPerceptron:
    def forward_propagate(data):   # Forward pass
    def backpropagate(yhat, labels):  # Backward pass (LEARNING!)
    def update_weights(eta):       # Gradient descent
    def evaluate(...):             # Main training loop

# 4. Load and prepare MNIST data
Xtr, X_test = normalized_and_flattened
batched_data = create_batches(X_train, batch_size=100)

# 5. Create network and train
mlp = MultiLayerPerceptron([784, 100, 100, 10])
mlp.evaluate(train_data, train_labels, test_data, test_labels, 
             num_epochs=70, eta=0.05)
```

---

## Key Concepts Explained

### Task 1: k-NN
**What it does:**
- Simple: store all training data
- For each test image: find closest training image
- Predict that training image's label

**Why it works:**
- Similar images (by pixel distance) usually have same digit

**Why preprocessing matters:**
- Raw pixels [0-255]: brightness dominates
- Normalized [0-1]: structure matters equally

**Limitation:**
- O(N) per prediction (slow for large datasets)
- All hyperparameter tuning happens at prediction time

### Task 2: Artificial Neural Networks
**What it does:**
- Learn patterns automatically through backpropagation
- Multi-layer network learns hierarchical features
- Sigmoid provides non-linearity (essential!)

**How learning works:**
1. Make prediction (forward pass)
2. Compute error vs true label
3. Backpropagate error to compute gradients
4. Update weights with gradient descent

**Why it's better than k-NN:**
- Fast at prediction time (independent of training set size)
- True learning (generalizes to unseen data)
- Can learn complex patterns

**Learning rate (eta) effect:**
- eta=0.05: Good, converges in ~50 epochs
- eta=0.005: Too small, very slow
- eta=0.5: Too large, might oscillate/diverge

---

## Common Questions

### Q: Why does k-NN accuracy decrease with larger k?
**A:** MNIST is a clean dataset with well-separated classes. The single closest neighbor is usually correct. Adding k=3 or k=5 includes "distractor" images from different classes that hurt voting.

### Q: Why divide by 255 for normalization?
**A:** Pixel values range [0, 255]. Dividing by 255 scales to [0, 1]. This prevents brightness from dominating the distance calculation.

### Q: What is backpropagation?
**A:** An algorithm to efficiently compute gradients using the chain rule. Instead of computing gradients for each layer independently, we propagate error backwards through layers, using previously computed deltas.

### Q: Why use mini-batches instead of full batch?
**A:** 
1. Faster learning (update weights multiple times per epoch)
2. Less memory required
3. Noise in gradients helps escape local minima

### Q: Why is test error higher than training error?
**A:** Overfitting. The network learns training data patterns (including noise) that don't generalize to test data. Training error ≈ 0% but test error ≈ 2.56%.

### Q: What do the activation functions do?
**A:** Introduce non-linearity. Without them, stacking linear layers is just one big linear transformation (no more powerful than single layer). Sigmoid in hidden layers creates non-linear decision boundaries. Softmax at output produces probability distribution.

---

## For Your Professor

When presenting these codes, emphasize:

1. **Part I (k-NN):**
   - Shows importance of data preprocessing
   - Demonstrates how distance metric matters
   - Explains why simple algorithms can fail with raw data
   - Extends 1-NN to k-NN with voting mechanism

2. **Part II (ANN + Backprop):**
   - Implements core neural network learning algorithm
   - Shows how networks learn through gradient descent
   - Demonstrates backpropagation (chain rule in action)
   - Achieves much higher accuracy than k-NN (97% vs 82%)
   - Shows overfitting phenomenon

3. **Key Insight:**
   - Machine learning = good algorithms + good data
   - Part I shows importance of data preparation
   - Part II shows power of learning algorithms
   - Together: understanding both aspects is crucial
