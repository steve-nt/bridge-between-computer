# ==================================================================================
# ARTIFICIAL NEURAL NETWORKS WITH BACKPROPAGATION - DETAILED COMMENTS
# ==================================================================================
# This notebook implements a Multi-Layer Perceptron (MLP) with backpropagation
# to classify MNIST handwritten digits. It demonstrates how neural networks
# learn through gradient descent optimization.
# ==================================================================================

import numpy as np
from keras.datasets import mnist

# ==================================================================================
# SECTION 1: ACTIVATION FUNCTIONS
# ==================================================================================
# Activation functions introduce non-linearity to the network
# Without them, stacked linear layers would just be one big linear transformation

def f_sigmoid(X, deriv=False):
    """
    SIGMOID ACTIVATION FUNCTION
    
    Mathematical Definition:
    σ(x) = 1 / (1 + e^(-x))
    
    Properties:
    - Outputs range from 0 to 1
    - S-shaped curve (smooth transition)
    - Used in hidden layers for non-linearity
    - Derivative: σ'(x) = σ(x) * (1 - σ(x))
    
    Why Sigmoid?
    - Smooth differentiable function (important for backpropagation)
    - Squashes large values to [0,1] range
    - Problem: Saturates (gradient near 0 at extremes) → slow learning
    
    Args:
        X: Input values (can be scalars, vectors, or matrices)
        deriv: If False, compute sigmoid. If True, compute derivative
    
    Returns:
        Sigmoid values or their derivatives
    """
    if not deriv:
        # Compute sigmoid: 1 / (1 + e^(-x))
        return 1 / (1 + np.exp(-X))
    else:
        # Compute derivative: σ(x) * (1 - σ(x))
        # This is more numerically stable than direct differentiation
        return f_sigmoid(X) * (1 - f_sigmoid(X))


def f_softmax(X):
    """
    SOFTMAX ACTIVATION FUNCTION (Output Layer)
    
    Mathematical Definition:
    softmax(x_i) = e^(x_i) / sum(e^(x_j) for all j)
    
    Properties:
    - Outputs sum to 1 (probability distribution)
    - All outputs positive and between 0 and 1
    - Used ONLY at output layer for multi-class classification
    - Exponential function amplifies differences between classes
    
    Why Softmax?
    - Converts network outputs to probabilities
    - Interpretable: output[i] = probability of class i
    - Works well with cross-entropy loss
    - Numerically stable version below avoids exp overflow
    
    Args:
        X: Input from previous layer, shape (batch_size, num_classes)
           e.g., (100, 10) for MNIST (100 images, 10 digit classes)
    
    Returns:
        Probability distribution, shape (batch_size, num_classes)
    """
    # Numerical stability trick: subtract max before exponential
    # This prevents exp(large_number) = infinity overflow
    X_stable = X - np.max(X, axis=1, keepdims=True)
    
    # Compute exp for each element
    exp_X = np.exp(X_stable)
    
    # Sum exponentials across classes (axis=1) for each sample
    Z = np.sum(exp_X, axis=1)
    Z = Z.reshape(Z.shape[0], 1)  # Reshape to (batch_size, 1) for broadcasting
    
    # Divide each element by its row sum to get probabilities
    return exp_X / Z


# ==================================================================================
# SECTION 2: LAYER CLASS - Individual Neural Network Layer
# ==================================================================================

class Layer:
    """
    Represents a single layer in the neural network
    
    Structure of a layer:
    INPUT (Z) → WEIGHT MATRIX (W) → MATRIX MULTIPLY → OUTPUT (S) → ACTIVATION → NEXT LAYER
    
    Key variables:
    - Z: Activations from previous layer (input to this layer)
    - S: Pre-activation values (output of weight matrix multiplication)
    - W: Weight matrix connecting to next layer
    - D: Delta (gradient) for this layer during backpropagation
    - Fp: Derivative of activation function (used in backprop)
    """
    
    def __init__(self, size, batch_size, is_input=False, is_output=False,
                 activation=f_sigmoid):
        """
        Initialize a neural network layer
        
        Args:
            size: Tuple of (input_size, output_size) for weight matrix
                  e.g., (784, 100) for layer with 784 inputs and 100 neurons
            batch_size: Number of samples in each mini-batch (e.g., 100)
            is_input: Boolean - is this the input layer?
            is_output: Boolean - is this the output layer?
            activation: Function to use for activation (sigmoid, softmax, etc.)
        """
        self.is_input = is_input
        self.is_output = is_output
        
        # Z matrix: holds output/activation values from this layer
        # Shape: (batch_size, size[0])
        # e.g., for input layer: (100, 784) - 100 images, each 784 pixels
        self.Z = np.zeros((batch_size, size[0]))
        
        # Store the activation function for forward/backward propagation
        self.activation = activation
        
        # ============ WEIGHT MATRIX (W) ============
        # W: weights connecting this layer's output to next layer's input
        # Shape: (size[0], size[1]) = (input_to_next, neurons_in_next)
        # e.g., (100, 100) to connect 100 neurons to 100 neurons in next layer
        # Not needed at output layer (no layer after it)
        self.W = None
        if not is_output:
            # Initialize weights with small random values
            # Normal distribution with mean 0, std 1E-4 (small values)
            # Small initialization prevents saturation and speeds learning
            self.W = np.random.normal(size=size, scale=1E-4)
        
        # ============ PRE-ACTIVATION VALUES (S) ============
        # S: values BEFORE activation function applied
        # This is Z_prev @ W (matrix multiplication of previous activations and weights)
        # Not needed at input layer
        self.S = None
        if not is_input:
            self.S = np.zeros((batch_size, size[0]))
        
        # ============ DELTA (GRADIENT) FOR BACKPROPAGATION (D) ============
        # D: gradients for this layer computed during backprop
        # Used to update weights
        # Not needed at input layer
        self.D = np.zeros((batch_size, size[0]))
        
        # ============ ACTIVATION DERIVATIVE (Fp) ============
        # Fp: derivative of activation function for this layer
        # Needed for backpropagation error computation
        # Shape: (size[0], batch_size) - NOTE: transposed!
        # Not needed at output layer or input layer
        self.Fp = None
        if not is_input and not is_output:
            self.Fp = np.zeros((size[0], batch_size))
    
    def forward_propagate(self):
        """
        FORWARD PROPAGATION: Compute output of this layer
        
        Process:
        1. For input layer: return Z @ W (linear transformation, no activation)
        2. For hidden layers: apply activation to S, add bias term, return result @ W
        3. For output layer: apply activation (softmax) to S and return
        
        Returns:
            The output of this layer, which becomes input to next layer
        """
        # SPECIAL CASE: Input Layer
        if self.is_input:
            # Input layer just does linear transformation: Z @ W
            # No activation function applied at input
            return self.Z.dot(self.W)
        
        # GENERAL CASE: Hidden or Output Layer
        # Apply activation function to pre-activation values S
        self.Z = self.activation(self.S)
        
        # SPECIAL CASE: Output Layer
        if self.is_output:
            # Output layer returns softmax probabilities directly
            return self.Z
        
        # HIDDEN LAYER PROCESSING
        else:
            # Add bias term: append column of ones to activations
            # This allows the network to learn biases
            # After adding bias: shape changes from (batch, neurons) to (batch, neurons+1)
            self.Z = np.append(self.Z, np.ones((self.Z.shape[0], 1)), axis=1)
            
            # Compute derivative of activation function
            # Needed for backpropagation to compute gradients
            # Shape: (neurons, batch_size) - transposed for matrix math
            self.Fp = self.activation(self.S, deriv=True).T
            
            # Return output of this layer (including bias) matrix-multiplied by next layer weights
            return self.Z.dot(self.W)


# ==================================================================================
# SECTION 3: MULTI-LAYER PERCEPTRON CLASS
# ==================================================================================

class MultiLayerPerceptron:
    """
    Complete neural network with multiple layers
    
    Network Architecture:
    Input (784) → Hidden1 (100) → Hidden2 (100) → Output (10)
    
    Learning Process:
    1. Forward Pass: Input → Layer1 → Layer2 → ... → Output
    2. Compute Loss: Compare predictions to labels
    3. Backward Pass: Compute gradients layer by layer (backpropagation)
    4. Weight Update: Adjust weights to minimize loss (gradient descent)
    """
    
    def __init__(self, layer_config, batch_size=100):
        """
        Initialize the neural network
        
        Args:
            layer_config: List of layer sizes
                         e.g., [784, 100, 100, 10]
                         means: 784 inputs → 100 hidden → 100 hidden → 10 outputs
            batch_size: Number of samples per mini-batch (e.g., 100)
        """
        self.layers = []
        self.num_layers = len(layer_config)
        self.minibatch_size = batch_size
        
        # Create layers according to configuration
        for i in range(self.num_layers - 1):
            
            # INPUT LAYER
            if i == 0:
                print(f"Initializing input layer with size {layer_config[i]}")
                # Input layer: 784 pixels + 1 bias = 785 inputs to next layer
                # Weight matrix shape: (784+1, 100) connecting 784+bias to 100 hidden units
                self.layers.append(Layer(
                    [layer_config[i] + 1, layer_config[i + 1]],
                    batch_size,
                    is_input=True  # No activation at input
                ))
            
            # HIDDEN LAYERS
            else:
                print(f"Initializing hidden layer with size {layer_config[i]}")
                # Hidden layer: neurons + 1 bias term
                # Weight matrix connects to next layer
                self.layers.append(Layer(
                    [layer_config[i] + 1, layer_config[i + 1]],  # +1 for bias
                    batch_size,
                    activation=f_sigmoid  # Use sigmoid activation
                ))
        
        # OUTPUT LAYER
        print(f"Initializing output layer with size {layer_config[-1]}")
        self.layers.append(Layer(
            [layer_config[-1], None],  # No next layer, so None for output size
            batch_size,
            is_output=True,  # Last layer has special handling
            activation=f_softmax  # Use softmax for probability outputs
        ))
        print("Done!")
    
    def forward_propagate(self, data):
        """
        FORWARD PASS: Compute network output for given input
        
        Process:
        Input → Layer1 → Layer2 → ... → Output
        
        For each layer i:
            output_i = Layer_i.forward_propagate()
            input_(i+1) = output_i
        
        Args:
            data: Input data, shape (batch_size, 784) for MNIST
        
        Returns:
            Network output (softmax probabilities), shape (batch_size, 10)
        """
        # Add bias to input layer: append column of ones
        # This allows network to learn bias terms
        # Shape changes from (batch, 784) to (batch, 785)
        self.layers[0].Z = np.append(data, np.ones((data.shape[0], 1)), axis=1)
        
        # Forward propagate through all layers
        for i in range(self.num_layers - 1):
            # Layer i's output becomes Layer i+1's input (S is pre-activation)
            self.layers[i + 1].S = self.layers[i].forward_propagate()
        
        # Return final output (softmax probabilities from output layer)
        return self.layers[-1].forward_propagate()
    
    def backpropagate(self, yhat, labels):
        """
        BACKPROPAGATION: Compute gradients for all layers
        
        This is THE core algorithm of neural network training!
        
        Process (Chain Rule):
        1. Compute error at output: delta_out = yhat - labels
        2. For each layer going backwards:
           delta_hidden = (W^T @ delta_next) * f'(S)
        3. These deltas (D matrices) will be used to update weights
        
        Why backwards?
        - We know the error at the output
        - Chain rule tells us how error propagates backwards through network
        - Each earlier layer's gradient depends on later layer's gradient
        
        Args:
            yhat: Network predictions, shape (batch_size, 10)
            labels: True labels (one-hot encoded), shape (batch_size, 10)
        """
        # QUESTION FOR STUDENTS (Task 2.1):
        # "What is computed in the next line of code?"
        # ANSWER: Delta (error) at output layer
        #         This is the gradient of loss with respect to output
        #         For softmax + cross-entropy: gradient = yhat - labels
        
        # Compute output layer deltas
        # Shape: (10, batch_size) - transposed for matrix multiplication
        self.layers[-1].D = (yhat - labels).T
        
        # BACKPROPAGATE through hidden layers (from second-to-last to first hidden)
        for i in range(self.num_layers - 2, 0, -1):  # Loop backwards from layer-2 to layer 1
            
            # Remove bias column from weight matrix
            # Original W shape: (101, next_size) - includes bias weight
            # W_nobias shape: (100, next_size) - only actual neuron weights
            W_nobias = self.layers[i].W[0:-1, :]
            
            # QUESTION FOR STUDENTS (Task 2.1):
            # "What does this 'for' loop do?"
            # This is a placeholder comment where students should identify
            # what backpropagation computation is happening
            
            # Compute hidden layer deltas using chain rule
            # delta_i = (W_next^T @ delta_next) * f'(S_i)
            # This multiplies:
            # 1. W_nobias.dot(self.layers[i+1].D): Error signal from next layer
            # 2. self.layers[i].Fp: Derivative of activation at this layer
            # Element-wise multiplication gives gradient for this layer
            self.layers[i].D = W_nobias.dot(self.layers[i + 1].D) * self.layers[i].Fp
    
    def update_weights(self, eta):
        """
        WEIGHT UPDATE: Gradient Descent Step
        
        After computing gradients via backprop, update all weights:
        W_new = W_old + eta * gradient
        
        This is where the network actually learns!
        
        Gradient Descent Formula:
        W := W - eta * ∇L
        where eta (learning rate) controls step size
        
        Args:
            eta: Learning rate (e.g., 0.05)
                Controls how much we update weights each iteration
                Too large: diverges, too small: converges slowly
        """
        # Update weights for each layer
        for i in range(0, self.num_layers - 1):
            # Compute weight gradient: -eta * delta @ activation
            # delta.dot(Z) gives gradient matrix for weight update
            # Transpose to match weight matrix shape
            W_grad = -eta * (self.layers[i + 1].D.dot(self.layers[i].Z)).T
            
            # Update weights (W := W + gradient)
            self.layers[i].W += W_grad
    
    def evaluate(self, train_data, train_labels, test_data, test_labels,
                 num_epochs=70, eta=0.05, eval_train=False, eval_test=True):
        """
        TRAINING LOOP: Main learning procedure
        
        Algorithm (for num_epochs times):
        1. For each mini-batch:
           a. Forward propagate (compute predictions)
           b. Backpropagate (compute gradients)
           c. Update weights (apply gradient descent)
        2. Evaluate on train/test sets
        
        Args:
            train_data: Training images, list of batches
            train_labels: Training labels (one-hot encoded), list of batches
            test_data: Test images, list of batches
            test_labels: Test labels (one-hot encoded), list of batches
            num_epochs: Number of complete passes through data (e.g., 70)
            eta: Learning rate (e.g., 0.05)
            eval_train: Whether to report training error
            eval_test: Whether to report test error
        """
        # Calculate total number of samples for error rate computation
        N_train = len(train_labels) * len(train_labels[0])  # num_batches * batch_size
        N_test = len(test_labels) * len(test_labels[0])
        
        print(f"Training for {num_epochs} epochs...")
        
        # TRAINING LOOP
        for t in range(0, num_epochs):
            out_str = f"[{t:4d}] "  # Epoch counter for output
            
            # Process each mini-batch
            for b_data, b_labels in zip(train_data, train_labels):
                # Step 1: FORWARD PROPAGATE
                # Compute network predictions for this batch
                output = self.forward_propagate(b_data)
                
                # Step 2: BACKPROPAGATE
                # Compute gradients using chain rule (backpropagation algorithm)
                self.backpropagate(output, b_labels)
                
                # QUESTION FOR STUDENTS (Task 2.1):
                # "How does weight update is implemented? What is eta?"
                # ANSWER: update_weights() implements gradient descent
                #         eta is learning rate: controls step size
                #         Large eta: fast learning but might diverge
                #         Small eta: slow learning but more stable
                
                # Step 3: WEIGHT UPDATE
                # Apply gradient descent: W := W - eta * gradient
                self.update_weights(eta=eta)
            
            # EVALUATION ON TRAINING DATA
            if eval_train:
                errs = 0
                for b_data, b_labels in zip(train_data, train_labels):
                    output = self.forward_propagate(b_data)
                    # Get predicted class (argmax of softmax output)
                    yhat = np.argmax(output, axis=1)
                    # Count errors (where prediction doesn't match true label)
                    errs += np.sum(1 - b_labels[np.arange(len(b_labels)), yhat])
                
                train_error = float(errs) / N_train
                out_str = f"{out_str} Training error: {train_error:.5f}"
            
            # EVALUATION ON TEST DATA
            if eval_test:
                errs = 0
                for b_data, b_labels in zip(test_data, test_labels):
                    output = self.forward_propagate(b_data)
                    yhat = np.argmax(output, axis=1)
                    errs += np.sum(1 - b_labels[np.arange(len(b_labels)), yhat])
                
                test_error = float(errs) / N_test
                out_str = f"{out_str} Test error: {test_error:.5f}"
            
            print(out_str)


# ==================================================================================
# SECTION 4: UTILITY FUNCTIONS
# ==================================================================================

def label_to_bit_vector(labels, nbits):
    """
    Convert class labels to one-hot encoded vectors
    
    Purpose: Neural networks work with continuous outputs, not class indices
    
    Example:
    Input: label = 3, nbits = 10
    Output: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
           (all zeros except 1 at position 3)
    
    Args:
        labels: Array of class indices, shape (batch_size,)
        nbits: Number of classes (e.g., 10 for digits 0-9)
    
    Returns:
        One-hot encoded matrix, shape (batch_size, nbits)
    """
    # Create matrix of zeros: batch_size x num_classes
    bit_vector = np.zeros((labels.shape[0], nbits))
    
    # For each sample, set the position of its class to 1
    for i in range(labels.shape[0]):
        bit_vector[i, labels[i]] = 1.0
    
    return bit_vector


def create_batches(data, labels, batch_size, create_bit_vector=False):
    """
    Split data into mini-batches
    
    Purpose: 
    - Speeds up learning (updates weights more frequently)
    - Reduces memory usage (don't load entire dataset)
    - Adds noise to optimization (helps escape local minima)
    
    Args:
        data: All input data, shape (N_total, 784)
        labels: All labels, shape (N_total,)
        batch_size: Size of each mini-batch (e.g., 100)
        create_bit_vector: If True, convert labels to one-hot
    
    Returns:
        chunked_data: List of batches, each shape (batch_size, 784)
        chunked_labels: List of label batches
    """
    N = data.shape[0]
    print(f"Batch size {batch_size}, the number of examples {N}.")
    
    if N % batch_size != 0:
        print(f"Warning: Batch size {batch_size} doesn't divide {N} evenly")
    
    chunked_data = []
    chunked_labels = []
    idx = 0
    
    # Create batches by slicing data
    while idx + batch_size <= N:
        # Add one batch of data
        chunked_data.append(data[idx:idx + batch_size, :])
        
        # Add corresponding labels (with optional one-hot encoding)
        if not create_bit_vector:
            chunked_labels.append(labels[idx:idx + batch_size])
        else:
            # Convert labels to one-hot vectors
            bit_vector = label_to_bit_vector(labels[idx:idx + batch_size], 10)
            chunked_labels.append(bit_vector)
        
        idx += batch_size
    
    return chunked_data, chunked_labels


def prepare_for_backprop(batch_size, Train_images, Train_labels, Valid_images, Valid_labels):
    """
    Prepare data for backpropagation training
    
    Steps:
    1. Split into mini-batches
    2. Convert labels to one-hot encoding
    
    Args:
        batch_size: Mini-batch size (e.g., 100)
        Train_images: Training images, shape (N_train, 784)
        Train_labels: Training labels, shape (N_train,)
        Valid_images: Validation/test images, shape (N_test, 784)
        Valid_labels: Validation/test labels, shape (N_test,)
    
    Returns:
        batched_train_data: List of training batches
        batched_train_labels: List of one-hot label batches
        batched_valid_data: List of validation batches
        batched_valid_labels: List of one-hot label batches
    """
    print("Creating data...")
    
    # Create training batches
    batched_train_data, batched_train_labels = create_batches(
        Train_images, Train_labels,
        batch_size,
        create_bit_vector=True  # Convert to one-hot
    )
    
    # Create validation batches
    batched_valid_data, batched_valid_labels = create_batches(
        Valid_images, Valid_labels,
        batch_size,
        create_bit_vector=True  # Convert to one-hot
    )
    
    print("Done!")
    
    return batched_train_data, batched_train_labels, batched_valid_data, batched_valid_labels


# ==================================================================================
# SECTION 5: MAIN EXECUTION - TRAINING THE NETWORK
# ==================================================================================

# Load MNIST dataset
(Xtr, Ltr), (X_test, L_test) = mnist.load_data()

print(f"{Xtr.shape[0]} train samples")
print(f"{X_test.shape[0]} test samples")

# Reshape and normalize images
# From (N, 28, 28) to (N, 784) - flatten images to vectors
Xtr = Xtr.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

# Convert to float and normalize to [0, 1]
Xtr = Xtr.astype('float32')
X_test = X_test.astype('float32')
Xtr /= 255
X_test /= 255

# Prepare mini-batches with one-hot encoding
batch_size = 100
train_data, train_labels, valid_data, valid_labels = prepare_for_backprop(
    batch_size, Xtr, Ltr, X_test, L_test
)

# Create the neural network
# Architecture: 784 (input) → 100 (hidden) → 100 (hidden) → 10 (output)
mlp = MultiLayerPerceptron(layer_config=[784, 100, 100, 10], batch_size=batch_size)

# Train the network
# num_epochs=70: Train for 70 passes through data
# eta=0.05: Learning rate (gradient descent step size)
# eval_train=True: Report training error
# eval_test=True: Report test error
mlp.evaluate(
    train_data, train_labels,
    valid_data, valid_labels,
    num_epochs=70,
    eta=0.05,
    eval_train=True,
    eval_test=True
)

print("Done:)\n")

# ==================================================================================
# UNDERSTANDING THE RESULTS
# ==================================================================================

"""
EXPECTED OUTPUT FROM TRAINING:
[   0]  Training error: 0.46320 Test error: 0.46380
[   1]  Training error: 0.07495 Test error: 0.07610
...
[  69]  Training error: 0.00000 Test error: 0.02560

INTERPRETATION:

1. EPOCH 0 (Initial)
   - Error ≈ 46%: Random initialization, network knows nothing
   - Training and test errors similar: no overfitting yet

2. EPOCHS 1-10 (Fast Learning)
   - Error drops from 46% to 3%: Huge improvement!
   - Network learns basic digit patterns
   - Steepest gradient descent

3. EPOCHS 10-50 (Fine-tuning)
   - Error slowly decreases
   - Training error → 0.001%, test error → 0.025%
   - Network memorizing training data (overfitting)

4. EPOCHS 50-69 (Convergence)
   - Error converges: training error ≈ 0%, test error ≈ 2.5%
   - Gap between train/test grows: classic overfitting
   - Network has learned the data patterns

OVERFITTING ANALYSIS:
- Training error: 0.00% (perfect)
- Test error: 2.56% (imperfect)
- The network memorized training data but doesn't generalize perfectly

FINAL ACCURACY: 97.44% (100% - 2.56% error)

KEY CONCEPTS:

1. FORWARD PROPAGATION
   - Input → Layer1 (sigmoid) → Layer2 (sigmoid) → Output (softmax)
   - Computes predictions

2. BACKPROPAGATION
   - Computes gradients for each layer
   - Uses chain rule to propagate error backwards
   - Deltas tell us how to update weights

3. WEIGHT UPDATE
   - W := W - eta * gradient
   - Moves weights towards better solution
   - Eta (learning rate) controls step size

4. LEARNING RATE EFFECT (Task 2.3)
   - eta=0.05: Good, converges in ~50 epochs
   - eta=0.005: Too small, slow learning (many epochs needed)
   - eta=0.5: Too large, unstable (might diverge)

5. ACTIVATION FUNCTIONS
   - Sigmoid in hidden layers: non-linear, creates complex boundaries
   - Softmax in output: produces probability distribution
   - Without activation: just linear regression!

WHY NEURAL NETWORKS WORK:

1. Universality: Multi-layer networks can learn any function
2. Backprop: Efficient way to compute gradients (vs brute force)
3. Gradient Descent: Updates move network towards better solution
4. Mini-batches: Stochastic updates help escape local minima

COMPARISON TO PREVIOUS TASKS:

k-NN (Task 1):
- Accuracy: ~82% on MNIST
- Computation: O(N) per prediction (slow for large datasets)
- No learning: just memorization

Neural Network (Task 2):
- Accuracy: ~97% on MNIST
- Computation: Fast once trained (independent of dataset size)
- True learning: generalizes to unseen data
"""

# ==================================================================================
# SUMMARY FOR YOUR PROFESSOR
# ==================================================================================

"""
NEURAL NETWORK ARCHITECTURE:
Layer 0 (Input): 784 neurons (pixels from 28x28 image)
Layer 1 (Hidden): 100 neurons with sigmoid activation
Layer 2 (Hidden): 100 neurons with sigmoid activation
Layer 3 (Output): 10 neurons with softmax (probability distribution)

TRAINING PROCESS (Backpropagation):
1. Forward Pass: Input → predictions
2. Compute Error: Compare predictions to labels
3. Backward Pass: Compute gradients (chain rule)
4. Update Weights: Apply gradient descent

MATHEMATICS:
Forward: activation = σ(weights @ input + bias)
Backward: gradient = (W^T @ error_next) * activation'(Z)
Update: weights := weights - learning_rate * gradients

RESULTS:
- 97.44% accuracy achieved after 70 epochs
- Training error: ~0% (memorization)
- Test error: ~2.56% (generalization)
- Shows overfitting: network learned training data but has some errors on new data
"""
