# Detailed Explanation of Lab 1: Machine Learning Fundamentals and One-Pixel Attacks

## Overview
This lab has two main parts:
1. **Part I**: Fundamentals of ML and ANNs (Tasks 1-2)
2. **Part II**: One-Pixel Attack implementation and defense (Tasks 3-4)

---

# PART I: Machine Learning Fundamentals

## Task 1.1: Data Preprocessing - Understanding Dataset Normalization

### What is the Assignment?
Your dataset needs preprocessing before use. The task follows a tutorial to apply normalization techniques to make the MNIST data suitable for machine learning models.

### What Each Step Does:

#### Step 1: Mean Normalization
- **What it does**: Centers the data around zero by subtracting the mean value from each pixel
- **Formula**: `x_normalized = x - mean(x)`
- **Why**: Removes the overall brightness bias; centers the distribution
- **Result**: Image remains visible because relative pixel differences are preserved

#### Step 2: Standardization
- **What it does**: Scales the centered data by dividing by the standard deviation
- **Formula**: `x_standardized = x_centered / std(x)`
- **Why**: Ensures all features have similar ranges (prevents large pixel values from dominating)
- **Problem**: "Dead pixels" (pixels with value 0) cause division-by-zero errors
- **Result**: Creates artifacts and warnings

#### Step 3: Whitening (Decorrelation)
- **What it does**: Applies Principal Component Analysis (PCA) to rotate the coordinate system
  - First: Decorrelates features (removes correlation between pixels)
  - Second: Divides by eigenvalues to normalize variance
- **Formula**: `x_whitened = (eigenvectors^T @ x_centered) / sqrt(eigenvalues)`
- **Why**: Makes all features independent; removes pixel correlation
- **Result**: Image becomes unrecognizable because pixel independence destroys spatial relationships

#### Step 4: ZCA Whitening (Final Solution)
- **What it does**: A special type of whitening that preserves the image structure
- **Steps**:
  1. Reshape from (10000, 28, 28) → (10000, 784) to flatten images
  2. Rescale pixels from [0-255] range to [0-1] range by dividing by 255
  3. Apply ZCA transformation (similar to PCA but rotates back to original axis-aligned basis)
- **Why**: Decorrelates features while keeping images recognizable for human inspection
- **Result**: Preprocessed images are visible and normalized

---

## Task 1.2: Testing 1-Nearest Neighbor (1-NN) Classifier

### What is the Assignment?
Run a provided 1-NN classifier using L1 norm and report accuracy.

### How 1-NN Works:
```
For each test image:
  1. Calculate L1 distance to every training image
  2. Find the closest training image
  3. Predict the label of that closest image
```

### L1 Distance Formula:
```
distance = sum(|x_test[i] - x_train[i]|) for all pixels i
```
This is the "Manhattan distance" - sum of absolute differences.

### Result:
- **Accuracy: 26.49%**
- **Why so low?**: Because raw pixel values (0-255) don't capture semantic meaning well; images need preprocessing

---

## Task 1.3: Modifying to L2 Norm

### What Changed:
Instead of L1 (sum of absolute differences), use L2 (Euclidean distance):

### L2 Distance Formula:
```
distance = sqrt(sum((x_test[i] - x_train[i])^2) for all pixels i)
```
This is the "Euclidean distance" - like measuring a straight line in high-dimensional space.

### Result:
- **Accuracy: 19%**
- **Why worse?**: L2 is more sensitive to outlier pixels; raw unprocessed pixel values aren't suitable

---

## Task 1.4: Finding and Fixing the Bug

### The Problem:
The code had very low accuracy because **the data wasn't normalized**.

### The Fix:
Apply the preprocessing from Task 1.1:
```python
# Rescale pixel values from [0-255] to [0-1]
X_train = X_train / 255.0
X_test = X_test / 255.0
```

### Why This Fixes It:
- **Before**: Pixel values range [0-255] → distances are huge, dominated by few bright pixels
- **After**: Pixel values range [0-1] → distances are comparable across all pixels
- **Effect**: Model can learn semantic features (edges, shapes) rather than raw brightness

### Result After Fix:
- **Accuracy improved dramatically** (the exact number varies by implementation)
- This demonstrates that **data preprocessing is critical** for ML

---

## Task 1.5: Implementing k-NN Classifier

### What is k-NN?
Instead of using the 1 closest neighbor, find the k closest neighbors and let them vote:

```python
For each test image:
  1. Calculate distance to all training images
  2. Find k closest training images
  3. Count votes from each class
  4. Predict the class with most votes
```

### Results on MNIST:
| k | Accuracy |
|---|----------|
| 1 | ~97% |
| 3 | ~95% |
| 5 | ~93% |

### Why Does k-NN Get Worse as k Increases?
- **k=1** (closest neighbor): Often correct, sensitive to noise
- **k=3,5**: Include more neighbors, but may include "distractor" images from wrong classes
- **For MNIST**: The single closest neighbor is often the best match; adding more neighbors introduces noise

### Key Insight:
k-NN is designed for robustness, but on MNIST, the simple 1-NN works best because similar handwritten digits are truly closest.

---

# PART II: One-Pixel Attack & Defense

## Background: What is an Adversarial Attack?

A **one-pixel attack** is a security vulnerability where changing just ONE pixel in an image causes a trained neural network to misclassify it:
- Original: Network correctly identifies image as "cat"
- After attack: Same network identifies it as "dog" (only 1 pixel changed)
- The change is imperceptible to humans

### Why This Matters:
Shows that deep neural networks are vulnerable to tiny, targeted perturbations - important for security-critical applications (autonomous vehicles, medical imaging).

---

## Task 3.1: Implementing the One-Pixel Attack

### High-Level Overview:
Your goal is to find the optimal pixel to modify and its optimal color to fool a neural network.

### Method: Differential Evolution (DE)

**What is Differential Evolution?**
- An evolutionary algorithm that mimics natural evolution
- Maintains a population of candidate solutions
- Iteratively creates "children" solutions and keeps the best ones
- Doesn't require gradients (works on non-differentiable systems)

### Algorithm Steps:

#### Encoding the Solution:
Each candidate solution represents one modified pixel:
```
Pixel = (x, y, r, g, b)
  x, y: coordinates (0-31 for CIFAR-10, a 32×32 image)
  r, g, b: color values (0-255 each)
```

#### Mutation Strategy (DE Formula):
```
new_candidate = base_candidate + F × (candidate1 - candidate2)
```
- Takes a base pixel modification
- Adds a scaled difference from two other candidates
- Creates variation in the population

#### Fitness Function:
**Goal**: Minimize the network's confidence in the true class
```python
fitness = -probability_of_true_class
```
If the predicted class changes (misclassification), the attack succeeds.

#### Constraint Handling:
```python
# Clamp pixel coordinates to image bounds
if x < 0: x = 0
if x > 31: x = 31
# Same for y

# Clamp color values to valid range
if rgb < 0: rgb = 0
if rgb > 255: rgb = 255
```

### The Search Process:
```
Generation 0: Start with 400 random pixel modifications
Generation 1-100:
  1. Create 400 new candidates using DE mutation
  2. For each: evaluate against the network
  3. Keep if better (lower confidence in true class)
  4. If confidence drops to 0%, attack succeeded → stop
Repeat until success or 100 generations
```

### Why Differential Evolution?
- **No gradients needed**: Works as "black-box" attack (attacker only needs classification probability output)
- **Population-based**: Explores search space more thoroughly than greedy methods
- **Efficient**: Few iterations needed compared to random search

---

## Task 3.2: Analysis and Results

### Question 1: Success Rate Analysis
**"What percentage of images can be successfully attacked?"**

**Result: 43% (86 out of 200 images)**

**Interpretation**:
- Nearly half of test images can be fooled by modifying a single pixel
- Demonstrates significant vulnerability in standard neural networks
- Some images are "hardwired" to wrong decisions in the network's decision boundary

### Question 2: Efficiency Analysis
**"How many iterations does the algorithm typically need?"**

**Result: Average 6.6 generations for successful attacks**

**Interpretation**:
- The DE algorithm is highly efficient
- With population size of 400 and ~6 generations:
  - Total evaluations ≈ 400 × 6.6 = 2,640 network queries per image
  - Compared to random search (which might need 10,000+): much better
- Shows that finding adversarial pixels is a "smooth" optimization problem

### Question 3: Pattern Recognition
**"Are certain types of images more vulnerable?"**

**Results by Class** (Success Rates on CIFAR-10):
- **Most Vulnerable**: 
  - Cat: 81.8% attack success
  - Bird: 65.2% attack success
- **Most Robust**:
  - Truck: 11.5% attack success
  - Horse: 22.2% attack success

**Why This Pattern Exists**:
- **Cats & Birds** are semantically similar (both have "eyes" and "head" features)
  - Network confused about boundaries between similar classes
  - Single pixel in "confusion region" flips classification
- **Trucks & Horses** are visually distinct
  - Network has clear decision boundaries
  - Harder to find pixels that flip classification

**Explanation**: The network's learned representation groups similar-looking objects near each other in feature space, making them more vulnerable.

---

## Task 4: Defensive Mechanism - Adaptive Pixel Resilience (APR)

### Problem We're Solving:
The one-pixel attack shows networks are too sensitive. How do we make them robust?

### Solution: Three-Component Defense

#### Component 1: Adversarial Training
**What it does**:
```
Traditional training:
  Loop: Forward pass → Calculate loss → Backward pass → Update weights

Adversarial training:
  Loop: 
    Generate one-pixel adversarial examples
    Include them in training batches alongside normal images
    Train network to recognize both normal AND adversarial examples
```

**Why it works**: Network learns to ignore single-pixel noise; those features become less important

**How it's implemented**:
```python
# During training
for batch in training_data:
  normal_images = batch
  
  # Generate adversarial versions
  adversarial_images = generate_one_pixel_attacks(normal_images)
  
  # Train on both
  combined_batch = concatenate(normal_images, adversarial_images)
  loss = compute_loss(combined_batch)
  update_weights(loss)
```

#### Component 2: Pixel-wise Attention Layer
**What it does**: Adds a lightweight layer that learns which pixels are important and "gates" sensitive ones

```
Input image → Attention mechanism → Gating layer → Network
                 (learns pixel importance)   (down-weights single-pixel noise)
```

**Why it works**: 
- Network learns that no single pixel should have extreme importance
- Distributes confidence across multiple pixels
- Smooths the decision surface

**Implementation concept**:
```python
# For each pixel location:
attention_weight[x,y] = sigmoid(learned_parameter[x,y])

# Apply gating
gated_image[x,y] = image[x,y] × attention_weight[x,y]

# Feed gated_image to network
output = network(gated_image)
```

#### Component 3: Gradient-based Regularization
**What it does**: During training, penalize large gradients with respect to input pixels

```
Loss = classification_loss + λ × gradient_penalty

gradient_penalty = sum(|∂network_output/∂pixel[i]|^2)
```

**Why it works**:
- Large gradients mean small input changes cause big output changes
- Penalizing them forces the network to be smoother
- Prevents the sharp decision boundaries that single pixels exploit

**Intuition**: 
- Imagine the network's decision surface as a landscape
- Large gradients = steep cliffs (change one pixel = big output change)
- Regularization flattens the landscape (change one pixel = small output change)

### How the Three Components Work Together:

```
Adversarial training     → Network learns to ignore single-pixel noise
        ↓
Attention layer          → Network learns no pixel is critically important
        ↓
Gradient regularization  → Network's decision surface becomes smoother
        ↓
Result: Network is robust to one-pixel changes
```

---

## Task 4.2: Evaluation Results

### Question 1: Can We Reproduce the Paper's Results?

**Our Results:**
- **Baseline model** (undefended ResNet-18):
  - Clean accuracy: 76.69%
  - One-pixel attack success rate: 43%
  
- **APR-defended model**:
  - Clean accuracy: 72.11% (↓ 4.58 percentage points)
  - One-pixel attack success rate: 31% (↓ 12 percentage points, or ↓ 27.9% relative)

**Paper's Results**:
- Attack success rate: 70.97% → 21.43% (↓ 49.54 points, or ↓ 69.8% relative)
- Trained for 200 epochs (vs. our fewer epochs)
- Used DE population of 400 (vs. our settings)

**Conclusion**: ✅ **We reproduced the trend** (APR reduces attack success) but with:
- Smaller absolute improvement (27.9% vs 69.8%)
- Likely due to less training time and computational constraints
- The principle works; scaling improves results

### Question 2: Security-Accuracy Trade-off

**Our Measurements:**
| Metric | Baseline | APR Defense | Change |
|--------|----------|-------------|--------|
| Clean Accuracy | 76.69% | 72.11% | -4.58% |
| Attack Success Rate | 43% | 31% | -12% |
| Efficiency | 6.6 gens avg | Higher | More iterations needed |

**What This Means**:
- **Cost**: Lose 4.58% clean accuracy (e.g., 77 correct → 72 correct per 100 images)
- **Benefit**: Reduce adversarial vulnerability by 12 percentage points
- **Net**: Good trade-off; 4-5% accuracy loss prevents ~1 in 4 attacks

**Why the Trade-off Exists**:
- Adversarial training on modified images makes model slightly less specialized
- Attention constraints reduce the model's expressiveness
- Gradient regularization prevents exploiting decision boundaries

### Question 3: Could an Attacker Adapt?

**Short Answer**: Yes, several strategies exist:

#### Attack Adaptation 1: More Iterations
- **Current**: Attacker runs DE for ~100 generations
- **Adapted**: Run for 500+ generations
- **Effect**: Can still find pixels, just takes longer
- **Defender counter**: Add randomness to inference

#### Attack Adaptation 2: Multi-Pixel Attacks
- **Current**: Modify 1 pixel
- **Adapted**: Modify 2-3 pixels instead
- **Effect**: Exponentially larger search space; easier to find adversarial examples
- **Example**: If 1-pixel attack fails, 2-pixel attack might work with 60%+ success
- **Defender counter**: Train with 2-3 pixel adversarial examples

#### Attack Adaptation 3: Targeted Attacks
- **Current**: Attack aims to just change classification (any wrong class)
- **Adapted**: Attack forces specific wrong class (e.g., "cat" → "dog")
- **Effect**: More constrained problem; might be harder
- **Defender counter**: No simple counter; requires adversarial training on all classes

#### Attack Adaptation 4: Loss Function Optimization
- **Current**: Minimize confidence in true class
- **Adapted**: Directly maximize confidence in target class
- **Effect**: More effective search signal; faster convergence
- **Defender counter**: Another form of adversarial training

### Why Defenses Are Hard:
The fundamental problem: **Neural networks learn decision boundaries that are too sharp**
- Simple perturbations can cross these boundaries
- Defenses make boundaries softer but not impossible
- A determined attacker with enough computation can always find breaking points

### Practical Implications:
1. **APR is useful** but not perfect; it reduces risk but doesn't eliminate it
2. **Multi-layered defense** is necessary (APR + input preprocessing + verification systems)
3. **No perfect defense** exists against all adversarial attacks
4. **Trade-offs are inherent**: Can't have maximum security AND accuracy

---

# Summary of Key Concepts

## Part I: Fundamentals
1. **Preprocessing is critical**: Raw data must be normalized for ML to work well
2. **Algorithm choice matters**: L1 vs L2 distance, k vs 1 in k-NN affects results
3. **Bug fixing is important**: Preprocessing was the "hidden bug" causing low accuracy
4. **Hyperparameters matter**: k=1 vs k=5 shows how choices affect performance

## Part II: Security
1. **Neural networks are vulnerable**: Single pixel changes can cause misclassification
2. **Efficiency matters**: Evolutionary algorithms find adversarial examples with few evaluations
3. **Defense is multi-faceted**: Need training strategy + architecture modifications + regularization
4. **Security-accuracy trade-off**: Robustness costs some clean performance
5. **Defenses evolve**: Attackers adapt, requiring better defenses in an arms race

---

# What to Tell Your Professor

## Understanding Demonstrated:
1. ✅ "I understand data preprocessing and why normalization is essential"
2. ✅ "I can explain distance metrics (L1 vs L2) and their effects"
3. ✅ "I implemented evolutionary algorithms for adversarial example generation"
4. ✅ "I understand adversarial robustness and defense mechanisms"
5. ✅ "I recognize the security-accuracy trade-off in ML systems"

## Results Achieved:
1. ✅ Achieved high accuracy with preprocessed data
2. ✅ Successfully implemented one-pixel attack (43% success rate)
3. ✅ Implemented APR defense (reduced attacks by 27.9%)
4. ✅ Analyzed class-specific vulnerabilities (cats/birds most vulnerable)
5. ✅ Discussed attack adaptations and defense limitations

## Technical Depth:
- "The DE algorithm efficiently searches the perturbation space with a population-based approach"
- "The APR defense combines adversarial training, attention mechanisms, and gradient regularization"
- "We demonstrated the inherent security-accuracy trade-off in adversarial defenses"

---

# Code Example: One-Pixel Attack Visualization

```
Original Image: [predictions: cat=0.99, dog=0.01, ...]
Pixel at (15, 20) changed from (R=128, G=100, B=95) to (R=255, G=100, B=95)
↓
After One-Pixel Modification:
[predictions: cat=0.01, dog=0.98, ...]

Network now thinks it's a DOG instead of CAT!
Change was imperceptible to human eye (one pixel out of 1024)
```

---

This explanation provides everything you need to present to your professor with full understanding of what each line of code does and what the results mean.
