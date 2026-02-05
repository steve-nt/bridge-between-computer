# ==================================================================================
# ONE-PIXEL ATTACK AND ADAPTIVE PIXEL RESILIENCE (APR) DEFENSE
# Implementation for CIFAR-10 with ResNet-18 - DETAILED COMMENTS
# ==================================================================================
# This code implements:
# Task 3.1: One-Pixel Attack using Differential Evolution (DE)
# Task 3.2: Analysis and evaluation of the attack
# Task 4: Adaptive Pixel Resilience (APR) defense mechanism
# Task 4.2: Defense evaluation and comparison
# ==================================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import numpy as np
import random
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ==================================================================================
# SECTION 1: DEVICE SETUP (Works on CPU, GPU, Mac M4)
# ==================================================================================

def get_device():
    """
    Automatically detect and return the best available device
    Priority: CUDA (NVIDIA GPU) → MPS (Mac GPU) → CPU
    
    Returns:
        torch.device: The device to use for computations
    """
    if torch.cuda.is_available():
        # NVIDIA GPU available
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Apple Metal Performance Shaders (Mac GPU)
        return torch.device('mps')
    # Fall back to CPU
    return torch.device('cpu')

DEVICE = get_device()
print(f"Using device: {DEVICE}")

# CIFAR-10 class labels
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']

# ==================================================================================
# SECTION 2: DATA LOADING
# ==================================================================================

def get_data_loaders(batch_size=64):
    """
    Load CIFAR-10 dataset with proper normalization
    
    Normalization uses ImageNet pre-trained statistics:
    - Mean: (0.4914, 0.4822, 0.4465) - average pixel values
    - Std: (0.2023, 0.1994, 0.2010) - standard deviation per channel
    
    Args:
        batch_size: Number of samples per batch (64 default)
    
    Returns:
        trainloader: DataLoader for training set (50,000 images)
        testloader: DataLoader for test set (10,000 images)
        trainset: Full training dataset (for index access)
        testset: Full test dataset (for index access)
    """
    # Define transformations: convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL image to torch tensor [0,1]
        # Normalize: (x - mean) / std for each channel
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Download and load training set
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                            download=True, transform=transform)
    
    # Download and load test set
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                           download=True, transform=transform)
    
    # Create data loaders (batch and shuffle for training)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, 
                                              shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, 
                                             shuffle=False, num_workers=0)
    
    return trainloader, testloader, trainset, testset

# ==================================================================================
# SECTION 3: MODEL SETUP - Modified ResNet-18
# ==================================================================================

def get_resnet18(num_classes=10):
    """
    Create a modified ResNet-18 for CIFAR-10
    
    Modifications from standard ImageNet ResNet-18:
    1. conv1: 7x7 kernel → 3x3 kernel (CIFAR-10 images are only 32x32)
    2. maxpool: Removed (would reduce 32x32 to tiny spatial dims)
    3. fc: Change output to 10 classes (CIFAR-10) instead of 1000 (ImageNet)
    
    Args:
        num_classes: Number of output classes (10 for CIFAR-10)
    
    Returns:
        model: Modified ResNet-18 on the appropriate device
    """
    # Load standard ResNet-18 architecture (no pre-trained weights)
    model = resnet18(weights=None)
    
    # Modify first convolution layer
    # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
    # New: Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
    # Why: CIFAR-10 images are 32x32 (vs ImageNet 224x224)
    #      Stride 1 and kernel 3 keeps spatial dimensions
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # Remove max pooling layer
    # Original: MaxPool2d(kernel_size=3, stride=2, padding=1)
    # New: Identity (no-op, passes data through unchanged)
    # Why: 32x32 image would become 8x8 (too much reduction)
    model.maxpool = nn.Identity()
    
    # Modify fully connected output layer
    # Original: fc = Linear(512, 1000)
    # New: fc = Linear(512, num_classes)
    # Why: CIFAR-10 has 10 classes (airplane, car, bird, etc.)
    model.fc = nn.Linear(512, num_classes)
    
    # Move model to appropriate device (GPU or CPU)
    return model.to(DEVICE)

# ==================================================================================
# SECTION 4: TASK 3.1 - ONE-PIXEL ATTACK WITH DIFFERENTIAL EVOLUTION
# ==================================================================================

class OnePixelAttack:
    """
    One-Pixel Attack Implementation using Differential Evolution (DE)
    
    Reference: "One Pixel Attack for Fooling Deep Neural Networks"
    by Su, Vargas, and Sakurai (2019)
    
    Attack Goal: Find a single pixel modification that causes misclassification
    Search Space: 5 dimensions - [row, col, r, g, b] where:
        - row, col: Pixel location (0-31 for 32x32 CIFAR-10 images)
        - r, g, b: RGB color values (0-255)
    
    Algorithm: Differential Evolution (population-based optimization)
        1. Initialize population with random perturbations
        2. For each generation:
           a. Create mutants using DE formula
           b. Evaluate fitness (confidence in true class)
           c. Select better solutions
        3. Return best perturbation found
    """
    
    def __init__(self, model):
        """
        Initialize the attacker
        
        Args:
            model: Neural network to attack (ResNet-18)
        """
        self.model = model
        self.model.eval()  # Set to evaluation mode (no dropout, frozen batchnorm)
        
        # CIFAR-10 normalization constants (used to denormalize predictions)
        # Images are normalized as: (x - mean) / std during loading
        # We need these to convert raw pixel values [0,255] to normalized [-N, N]
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465], device=DEVICE)
        self.std  = torch.tensor([0.2023, 0.1994, 0.2010], device=DEVICE)
    
    def _perturb_image(self, image, perturbation):
        """
        Apply one-pixel perturbation to an image
        
        Args:
            image: Original image (already normalized), shape (3, 32, 32)
            perturbation: [row, col, r, g, b] where:
                - row, col: pixel coordinates
                - r, g, b: RGB values in [0, 255]
        
        Returns:
            perturbed: Image with one pixel modified (normalized)
        """
        # Clone the original image (avoid modifying it)
        perturbed = image.clone().to(DEVICE)
        
        # ============ CONSTRAINT HANDLING: Keep coordinates in bounds ============
        # Clamp coordinates to valid range [0, 31] for 32x32 image
        row = int(max(0, min(perturbation[0], 31)))
        col = int(max(0, min(perturbation[1], 31)))
        
        # ============ MODIFY THE PIXEL ============
        # For each RGB channel (r=0, g=1, b=2)
        for c in range(3):
            # Extract RGB value (in range [0, 255])
            # perturbation[2+c] gets r, g, or b
            # max/min ensures it's in [0, 255]
            pixel_val = max(0, min(perturbation[2 + c], 255)) / 255.0  # Convert to [0, 1]
            
            # Apply CIFAR-10 normalization to match dataset statistics
            # normalized = (pixel - mean) / std
            perturbed[c, row, col] = (pixel_val - self.mean[c]) / self.std[c]
        
        return perturbed
    
    def _fitness(self, perturbation, image, true_label):
        """
        Evaluate fitness of a perturbation (minimize this!)
        
        Fitness = Probability of TRUE CLASS
        Why minimize? Lower prob of true class = more likely to misclassify
        
        Range: [0, 1]
        - 0: Network assigns 0% confidence to true class (GOOD)
        - 1: Network assigns 100% confidence to true class (BAD)
        
        Args:
            perturbation: [row, col, r, g, b]
            image: Original image
            true_label: Ground truth class label
        
        Returns:
            fitness: Probability of true class (lower is better)
        """
        # Apply perturbation
        perturbed = self._perturb_image(image, perturbation)
        
        # Evaluate network on perturbed image
        with torch.no_grad():  # No gradient computation (inference only)
            # Add batch dimension: (3,32,32) → (1,3,32,32)
            output = self.model(perturbed.unsqueeze(0))
            # Get softmax probabilities
            probs = torch.softmax(output, dim=1)[0]  # (10,) probabilities for each class
        
        # Return probability of true class
        return probs[true_label].item()
    
    def _initialize_population(self, pop_size):
        """
        Initialize population with random perturbations
        
        Population = List of individuals
        Individual = [row, col, r, g, b]
        
        Args:
            pop_size: Number of individuals in population (typically 50)
        
        Returns:
            population: List of random individuals
        """
        population = []
        
        for _ in range(pop_size):
            # Create one random individual [row, col, r, g, b]
            individual = [
                random.uniform(0, 31),    # row: uniform in [0, 31]
                random.uniform(0, 31),    # col: uniform in [0, 31]
                random.gauss(128, 127),   # r: Gaussian centered at 128, std 127
                random.gauss(128, 127),   # g: (covers roughly [0, 255])
                random.gauss(128, 127),   # b:
            ]
            
            # Clamp RGB values to valid range [0, 255]
            for i in range(2, 5):
                individual[i] = max(0, min(255, individual[i]))
            
            population.append(individual)
        
        return population
    
    def _mutation(self, population, F=0.5):
        """
        Differential Evolution Mutation Strategy: DE/rand/1
        
        Formula: v_i = x_r1 + F * (x_r2 - x_r3)
        
        For each individual i in population:
        1. Randomly select 3 different individuals (r1, r2, r3)
        2. Create mutant: take r1, add scaled difference (r2-r3)
        3. Clamp to bounds
        
        Args:
            population: Current population (list of individuals)
            F: Scale factor (typically 0.5) controls mutation strength
        
        Returns:
            mutants: List of mutated individuals
        """
        mutants = []
        pop_size = len(population)
        
        for i in range(pop_size):
            # Get list of indices excluding current individual i
            candidates = [j for j in range(pop_size) if j != i]
            # Randomly select 3 distinct individuals
            r1, r2, r3 = random.sample(candidates, 3)
            
            # Create mutant by applying DE formula to each dimension
            mutant = []
            for d in range(5):  # 5 dimensions: row, col, r, g, b
                # v = x_r1 + F * (x_r2 - x_r3)
                # Takes direction from (x_r2 - x_r3), scales by F, adds to x_r1
                v = population[r1][d] + F * (population[r2][d] - population[r3][d])
                mutant.append(v)
            
            # ============ CONSTRAINT HANDLING ============
            # Clamp coordinates to image bounds
            mutant[0] = max(0, min(31, mutant[0]))   # row
            mutant[1] = max(0, min(31, mutant[1]))   # col
            # Clamp RGB values
            for d in range(2, 5):
                mutant[d] = max(0, min(255, mutant[d]))
            
            mutants.append(mutant)
        
        return mutants
    
    def _selection(self, population, trials, fitness_pop, fitness_trials):
        """
        Selection: Keep better individual between parent and offspring
        
        For each position i:
            if fitness_trials[i] < fitness_pop[i]:
                keep trials[i]  (offspring is better)
            else:
                keep population[i]  (parent is better)
        
        Args:
            population: Parent population
            trials: Trial/mutant population
            fitness_pop: Fitness values of parents
            fitness_trials: Fitness values of trials
        
        Returns:
            new_population: Selected individuals
            new_fitness: Corresponding fitness values
        """
        new_population = []
        new_fitness = []
        
        for i in range(len(population)):
            # Lower fitness is better (minimize probability of true class)
            if fitness_trials[i] < fitness_pop[i]:
                # Trial is better - keep it
                new_population.append(trials[i])
                new_fitness.append(fitness_trials[i])
            else:
                # Parent is better - keep it
                new_population.append(population[i])
                new_fitness.append(fitness_pop[i])
        
        return new_population, new_fitness
    
    def attack(self, image, true_label, max_generations=50, pop_size=50):
        """
        Execute one-pixel attack using Differential Evolution
        
        Args:
            image: Image to attack (shape 3,32,32, already normalized)
            true_label: Ground truth class label
            max_generations: Maximum DE generations (typically 50)
            pop_size: Population size (typically 50)
        
        Returns:
            success: Boolean (True if misclassified)
            perturbed: The adversarial image
            new_label: Predicted label of adversarial image
            generations_used: How many generations were needed
        """
        
        # ============ INITIALIZATION ============
        # Initialize population with random perturbations
        population = self._initialize_population(pop_size)
        # Evaluate fitness for each individual
        fitness = [self._fitness(ind, image, true_label) for ind in population]
        
        # Track best solution found so far
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_individual = population[best_idx].copy()
        
        generations_used = 0
        
        # ============ MAIN DE LOOP ============
        for gen in range(max_generations):
            generations_used = gen + 1
            
            # ============ EARLY STOPPING ============
            # Check if best solution already succeeds
            perturbed_best = self._perturb_image(image, best_individual)
            with torch.no_grad():
                out_best = self.model(perturbed_best.unsqueeze(0).to(DEVICE))
                pred_best = out_best.argmax(dim=1).item()
            
            # If misclassified (pred_best != true_label), attack succeeded
            if pred_best != true_label:
                break  # No need to continue searching
            
            # ============ DE OPERATIONS ============
            # Step 1: Mutation - create mutants
            mutants = self._mutation(population, F=0.5)
            trials = mutants  # In DE/rand/1, no crossover, so trials = mutants
            
            # Step 2: Evaluation - compute fitness of trials
            fitness_trials = [self._fitness(t, image, true_label) for t in trials]
            
            # Step 3: Selection - keep better individuals
            population, fitness = self._selection(population, trials, fitness, fitness_trials)
            
            # ============ TRACK BEST ============
            # Update best solution if we found better one this generation
            gen_best_idx = np.argmin(fitness)
            if fitness[gen_best_idx] < best_fitness:
                best_fitness = fitness[gen_best_idx]
                best_individual = population[gen_best_idx].copy()
        
        # ============ RETURN RESULTS ============
        # Apply best perturbation found and get final prediction
        perturbed = self._perturb_image(image, best_individual)
        with torch.no_grad():
            output = self.model(perturbed.unsqueeze(0).to(DEVICE))
            new_label = output.argmax(dim=1).item()
        
        # Success = predicted label changed from true label
        success = (new_label != true_label)
        return success, perturbed, new_label, generations_used


# ==================================================================================
# SECTION 5: TASK 4.1 - ADAPTIVE PIXEL RESILIENCE (APR) DEFENSE
# ==================================================================================

class PixelWiseAttention(nn.Module):
    """
    Pixel-wise Attention Layer (Component 2 of APR)
    
    Purpose: Learn which pixels are important, gate out sensitivity to single pixels
    
    Formula: A(x) = sigmoid(W_a * x + b_a)
             output = x * A(x)
    
    Effect: Each pixel learns a gating weight between 0 and 1
            - 1: Keep this pixel (important)
            - 0: Ignore this pixel (not important)
    
    Why it works:
    - Network learns that no single pixel should have extreme importance
    - Distributes importance across multiple pixels
    - Attacks that change one pixel become less effective
    """
    
    def __init__(self, in_channels):
        """
        Args:
            in_channels: Number of input channels (64 for first layer)
        """
        super().__init__()
        # 1x1 convolution learns attention weights per pixel
        # Maps in_channels → in_channels (same number of channels)
        self.attn = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)
    
    def forward(self, x):
        """
        Apply pixel-wise attention
        
        Args:
            x: Input feature maps (batch_size, in_channels, H, W)
        
        Returns:
            Gated feature maps: x * sigmoid(attention)
        """
        # Compute attention weights using sigmoid (bounded to [0, 1])
        a = torch.sigmoid(self.attn(x))
        # Apply gating: element-wise multiplication
        return x * a


class APRResNet18(nn.Module):
    """
    ResNet-18 with Adaptive Pixel Resilience (APR) Defense
    
    Three Components of APR Defense:
    
    1. ADVERSARIAL TRAINING
       - Include one-pixel adversarial examples in training batches
       - Network learns to be robust to one-pixel changes
       
    2. PIXEL-WISE ATTENTION LAYER
       - Added after first convolution (here: after initial conv + batchnorm)
       - Learns which pixels/features are important
       - Prevents single-pixel changes from being catastrophic
       
    3. GRADIENT-BASED REGULARIZATION
       - Penalizes large gradients w.r.t. input
       - Smooths the loss landscape (decision boundary)
       - Makes network less sensitive to small perturbations
       - Implemented in training loop
    
    Architecture:
    Input → Conv1 → BN → ReLU → ATTENTION → ResLayer1 → ResLayer2 → ... → Output
    """
    
    def __init__(self, num_classes=10):
        """
        Args:
            num_classes: Number of output classes (10 for CIFAR-10)
        """
        super().__init__()
        
        # Load standard ResNet-18
        base = resnet18(weights=None)
        
        # Modify for CIFAR-10 (same as standard model)
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()
        base.fc = nn.Linear(512, num_classes)
        
        # Extract layers from base
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        
        # ADD PIXEL-WISE ATTENTION (Defense Component 2)
        # Applied after initial convolution
        # Input has 64 channels after conv1
        self.pixel_attention = PixelWiseAttention(64)
        
        # Rest of ResNet layers
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.fc = base.fc
    
    def forward(self, x):
        """
        Forward pass with attention
        
        Args:
            x: Input images (batch_size, 3, 32, 32)
        
        Returns:
            logits: Class predictions (batch_size, 10)
        """
        # Initial convolution + normalization + activation
        x = self.relu(self.bn1(self.conv1(x)))
        
        # APPLY PIXEL-WISE ATTENTION (Defense Component 2)
        # After first conv, before residual blocks
        x = self.pixel_attention(x)
        
        # Pass through ResNet blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling + classification
        x = self.avgpool(x)
        return self.fc(torch.flatten(x, 1))


def generate_adversarial_batch(model, images, labels, attacker, num_adv=8):
    """
    Generate adversarial examples for adversarial training (Defense Component 1)
    
    For a batch of images, generate adversarial examples using one-pixel attack
    and mix them with clean images for training.
    
    Args:
        model: Neural network (for attack)
        images: Clean training images (batch_size, 3, 32, 32)
        labels: Corresponding labels (batch_size,)
        attacker: OnePixelAttack instance
        num_adv: Number of adversarial examples to generate (8 by default)
    
    Returns:
        adv_images: Stack of successful adversarial examples
        adv_labels: Corresponding true labels
    """
    model.eval()  # Set to eval mode for attack generation
    adv_images = []
    adv_labels = []
    
    # Randomly select num_adv images from the batch
    indices = random.sample(range(images.size(0)), min(num_adv, images.size(0)))
    
    # Try to generate adversarial example for each selected image
    for idx in indices:
        img = images[idx]  # Already on DEVICE
        label = labels[idx].item()
        
        # Generate adversarial example
        # Use smaller generation/pop size for speed during training
        success, adv_img, _, _ = attacker.attack(img, label, max_generations=10, pop_size=20)
        
        if success:
            # Only use successfully generated adversarial examples
            adv_images.append(adv_img)
            adv_labels.append(label)
    
    model.train()  # Set back to training mode
    
    # Fallback: if no successful attacks, use original images
    if len(adv_images) == 0:
        fallback_imgs = images[indices]
        fallback_lbls = labels[indices]
        return fallback_imgs, fallback_lbls
    
    return torch.stack(adv_images), torch.tensor(adv_labels, device=DEVICE)


def train_apr_model(model, trainloader, epochs=5, alpha=0.7, beta=0.01):
    """
    Train APR model with three defense components
    
    Loss Function:
    L = alpha * L_clean + (1 - alpha) * L_adv + beta * R
    
    Where:
    - L_clean: Cross-entropy loss on clean images
    - L_adv: Cross-entropy loss on adversarial examples (Component 1: Adversarial Training)
    - R: Gradient regularization (Component 3: Input Smoothness)
      R = ||∇_x L||^2 (penalizes large gradients)
    
    Args:
        model: APRResNet18 instance
        trainloader: Training data loader
        epochs: Number of training epochs (5 recommended)
        alpha: Weight for clean loss (0.7 = 70% clean, 30% adversarial)
        beta: Weight for regularization term (0.01 recommended)
    
    Returns:
        model: Trained APR model
    """
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Create attacker for generating adversarial examples during training
    attacker = OnePixelAttack(model)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        
        for i, (images, labels) in enumerate(trainloader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # Enable gradient computation for images (needed for regularization)
            images.requires_grad = True
            
            # ============ CLEAN LOSS ============
            outputs = model(images)
            loss_clean = criterion(outputs, labels)
            
            # ============ GRADIENT REGULARIZATION (Component 3) ============
            # Compute gradients of loss w.r.t. input images
            # This tells us how sensitive the network is to input changes
            grads = torch.autograd.grad(loss_clean, images, create_graph=True)[0]
            
            # Regularization term: penalize large gradients
            # Large gradients = sensitive to small input changes = vulnerable to attack
            # R = beta * ||grad||^2 (sum of squared gradients)
            reg_loss = beta * (grads ** 2).mean()
            
            # ============ ADVERSARIAL TRAINING (Component 1) ============
            # Generate adversarial examples and train on them
            adv_images, adv_labels = generate_adversarial_batch(
                model, images.detach(), labels, attacker, num_adv=8
            )
            adv_outputs = model(adv_images)
            loss_adv = criterion(adv_outputs, adv_labels)
            
            # ============ COMBINED LOSS ============
            # L = alpha * L_clean + (1 - alpha) * L_adv + beta * R
            # alpha=0.7: 70% clean loss, 30% adversarial loss
            loss = alpha * loss_clean + (1 - alpha) * loss_adv + reg_loss
            
            # ============ BACKPROPAGATION & WEIGHT UPDATE ============
            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()         # Compute gradients
            optimizer.step()        # Update weights
            
            # Track statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)  # Get predicted labels
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # Print progress
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(trainloader):.4f}, '
              f'Acc: {100.*correct/total:.2f}%')
    
    return model


def train_standard_model(model, trainloader, epochs=5):
    """
    Train standard ResNet-18 (baseline, no defense)
    
    Simple supervised learning:
    L = CrossEntropyLoss(predictions, true_labels)
    
    Args:
        model: ResNet18 instance
        trainloader: Training data loader
        epochs: Number of epochs
    
    Returns:
        model: Trained standard model
    """
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(trainloader):.4f}, '
              f'Acc: {100.*correct/total:.2f}%')
    
    return model


def evaluate_model(model, testloader):
    """
    Evaluate model on clean test set
    
    Args:
        model: Neural network
        testloader: Test data loader
    
    Returns:
        accuracy: Percentage of correct predictions (0-100)
    """
    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100. * correct / total


def get_shared_correct_indices(model_a, model_b, dataset, k, seed=0):
    """
    Find k images that BOTH models classify correctly
    
    Why? Fair comparison: both models tested on same correctly-classified images
    (Otherwise, comparing ASR on different subsets would be unfair)
    
    Args:
        model_a: First model (standard)
        model_b: Second model (APR defense)
        dataset: Test dataset
        k: Number of shared correct images needed
        seed: Random seed for reproducibility
    
    Returns:
        shared: List of k image indices both models get correct
    """
    model_a.eval()
    model_b.eval()
    
    # Shuffle dataset randomly for unbiased selection
    rng = random.Random(seed)
    all_idx = list(range(len(dataset)))
    rng.shuffle(all_idx)
    
    shared = []
    with torch.no_grad():
        for i in all_idx:
            x, y = dataset[i]
            x = x.unsqueeze(0).to(DEVICE)
            
            # Check if both models classify this image correctly
            pred_a = model_a(x).argmax(dim=1).item()
            pred_b = model_b(x).argmax(dim=1).item()
            
            if pred_a == y and pred_b == y:
                shared.append(i)
                if len(shared) == k:
                    break
    
    if len(shared) == 0:
        raise RuntimeError("No shared correctly-classified images found.")
    
    if len(shared) < k:
        print(f"[Warning] Requested {k}, got only {len(shared)} shared correct samples.")
    
    return shared


# ==================================================================================
# SECTION 6: TASK 3.2 & 4.2 - EXPERIMENTS AND ANALYSIS
# ==================================================================================

def run_attack_experiments(model, testset, indices, model_name="Model"):
    """
    Run one-pixel attack on a set of images and collect statistics
    
    Answers Task 3.2 Questions:
    1. Success Rate: % of images successfully attacked
    2. Efficiency: Average generations needed for success
    3. Pattern Recognition: Which classes are most vulnerable?
    
    Args:
        model: Neural network to attack
        testset: Test dataset
        indices: List of image indices to attack
        model_name: Name for printing ("STANDARD" or "APR DEFENSE")
    
    Returns:
        success_rate: Percentage of successful attacks (0-100)
    """
    print(f"\n{'='*50}")
    print(f"ATTACK ANALYSIS: {model_name}")
    print(f"{'='*50}")
    
    attacker = OnePixelAttack(model)
    model.eval()
    
    # Data structures to track results
    results = {
        'success': 0,  # Number of successful attacks
        'total': 0,    # Total images attacked
        'generations': [],  # Generations used for each successful attack
        # For each class: {'attacked': count, 'success': count}
        'class_vulnerability': defaultdict(lambda: {'attacked': 0, 'success': 0})
    }
    
    num_samples = len(indices)
    
    # Attack each image
    for idx in indices:
        image, label = testset[idx]
        
        results['total'] += 1
        
        # Perform attack with full DE settings
        success, _, _, gens = attacker.attack(image, label, max_generations=50, pop_size=50)
        
        # Track per-class statistics
        results['class_vulnerability'][CLASSES[label]]['attacked'] += 1
        
        if success:
            results['success'] += 1
            results['generations'].append(gens)
            results['class_vulnerability'][CLASSES[label]]['success'] += 1
        
        # Print progress every 10 images
        if results['total'] % 10 == 0:
            print(f"Progress: {results['total']}/{num_samples}")
    
    # ============ ANALYSIS ============
    
    # Q1: Success Rate
    success_rate = 100 * results['success'] / max(1, results['total'])
    print(f"\n1. SUCCESS RATE: {results['success']}/{results['total']} = {success_rate:.2f}%")
    
    # Q2: Efficiency
    if results['generations']:
        print(f"2. EFFICIENCY: Avg generations for success: {np.mean(results['generations']):.1f}")
    
    # Q3: Pattern Recognition (Class Vulnerability)
    print(f"3. CLASS VULNERABILITY:")
    for cls in CLASSES:
        data = results['class_vulnerability'][cls]
        if data['attacked'] > 0:
            vuln = 100 * data['success'] / data['attacked']
            print(f"   {cls}: {vuln:.1f}% ({data['success']}/{data['attacked']})")
    
    return success_rate


# ==================================================================================
# SECTION 7: MAIN EXECUTION
# ==================================================================================

def main():
    """
    Main execution: Train both models and run attack experiments
    
    Flow:
    1. Load CIFAR-10 data
    2. Train standard ResNet-18 (baseline)
    3. Train APR-defended ResNet-18
    4. Find shared correctly-classified test images
    5. Attack both models with one-pixel attack
    6. Compare results
    """
    
    print("="*50)
    print("ONE-PIXEL ATTACK & APR DEFENSE")
    print("="*50)
    
    # ============ STEP 1: LOAD DATA ============
    print("\nLoading CIFAR-10...")
    trainloader, testloader, trainset, testset = get_data_loaders(batch_size=128)
    
    # ============ STEP 2: TRAIN STANDARD MODEL ============
    print("\n--- Training Standard ResNet-18 ---")
    standard_model = get_resnet18()
    standard_model = train_standard_model(standard_model, trainloader, epochs=5)
    standard_acc = evaluate_model(standard_model, testloader)
    print(f"Standard Model Accuracy: {standard_acc:.2f}%")
    
    # ============ STEP 3: TRAIN APR MODEL ============
    print("\n--- Training APR Defense Model ---")
    apr_model = APRResNet18().to(DEVICE)
    apr_model = train_apr_model(apr_model, trainloader, epochs=5, alpha=0.7, beta=0.01)
    apr_acc = evaluate_model(apr_model, testloader)
    print(f"APR Model Accuracy: {apr_acc:.2f}%")
    
    # Development vs Report mode
    DEV_MODE = False  # True for quick testing, False for full evaluation
    NUM_TEST = 50 if DEV_MODE else 100  # 100 test images for fair comparison
    
    # ============ STEP 4: FIND SHARED CORRECT IMAGES ============
    # Important: Test both models on same set for fair comparison
    FIXED_SEED = 0
    shared_indices = get_shared_correct_indices(
        standard_model, apr_model, testset, NUM_TEST, seed=FIXED_SEED
    )
    print(f"\nUsing {len(shared_indices)} shared correctly-classified test images for BOTH models.")
    
    # ============ STEP 5: ATTACK BOTH MODELS ============
    standard_asr = run_attack_experiments(standard_model, testset, shared_indices, "STANDARD")
    apr_asr = run_attack_experiments(apr_model, testset, shared_indices, "APR DEFENSE")
    
    # ============ STEP 6: COMPARISON & ANALYSIS ============
    print("\n" + "="*50)
    print("FINAL COMPARISON")
    print("="*50)
    
    # Create comparison table
    print(f"{'Metric':<25} {'Standard':<12} {'APR':<12}")
    print("-"*50)
    print(f"{'Clean Accuracy':<25} {standard_acc:.2f}%{'':<6} {apr_acc:.2f}%")
    print(f"{'Attack Success Rate':<25} {standard_asr:.2f}%{'':<6} {apr_asr:.2f}%")
    print(f"{'Robustness':<25} {100-standard_asr:.2f}%{'':<6} {100-apr_asr:.2f}%")
    
    # Calculate reduction metrics
    abs_reduction = standard_asr - apr_asr
    rel_reduction = (abs_reduction / standard_asr * 100) if standard_asr > 0 else 0
    
    # ============ ANSWER TASK 4.2 QUESTIONS ============
    print(f"\n--- Task 4.2 Analysis ---")
    
    # Q1: Can we reproduce paper's effectiveness?
    print(f"Q1: Paper effectiveness reproduction:")
    print(f"    Absolute ASR reduction: {abs_reduction:.2f} percentage points")
    print(f"    Relative ASR reduction: {rel_reduction:.2f}%")
    
    # Q2: Security vs Accuracy trade-off?
    print(f"Q2: Security vs Accuracy trade-off:")
    print(f"    Accuracy drop: {standard_acc-apr_acc:.2f} percentage points")
    print(f"    Robustness gain (absolute): {abs_reduction:.2f} percentage points")
    
    # Q3: Can attacker adapt?
    print(f"Q3: Attacker adaptations:")
    print(f"    - Use multi-pixel attacks (3-5 pixels)")
    print(f"    - Increase DE generations/population")
    print(f"    - Attack with targeted objective instead of untargeted")


if __name__ == "__main__":
    main()
