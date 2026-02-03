"""
One-Pixel Attack and Adaptive Pixel Resilience Defense
Implementation for CIFAR-10 with ResNet-18
"""

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

# Device setup - works on Mac M4, Windows, Linux
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

DEVICE = get_device()
print(f"Using device: {DEVICE}")

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']

# ============== Data Loading ==============
def get_data_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                           download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, 
                                              shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, 
                                             shuffle=False, num_workers=0)
    return trainloader, testloader, trainset, testset

# ============== Model Setup ==============
def get_resnet18(num_classes=10):
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, num_classes)
    return model.to(DEVICE)

# ============== Task 3.1: One-Pixel Attack with Custom DE ==============
class OnePixelAttack:
    """
    One-pixel attack using custom Differential Evolution implementation.
    Based on the paper One Pixel Attack for Fooling Deep Neural Networks.
    """
    
    def __init__(self, model):
        self.model = model
        self.model.eval()
        # Normalization constants for CIFAR-10
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465], device=DEVICE)
        self.std  = torch.tensor([0.2023, 0.1994, 0.2010], device=DEVICE)

    
    def _perturb_image(self, image, perturbation):
        """Apply one-pixel perturbation. perturbation: [row, col, r, g, b]"""
        perturbed = image.clone().to(DEVICE)
        # Constraint handling: clamp coordinates to valid range
        row = int(max(0, min(perturbation[0], 31)))
        col = int(max(0, min(perturbation[1], 31)))
        
        for c in range(3):
            # Convert RGB [0,255] to normalized value
            pixel_val = max(0, min(perturbation[2 + c], 255)) / 255.0
            perturbed[c, row, col] = (pixel_val - self.mean[c]) / self.std[c]
        return perturbed
    
    def _fitness(self, perturbation, image, true_label):
        """Fitness: probability of true class (minimize to attack)"""
        perturbed = self._perturb_image(image, perturbation)
        with torch.no_grad():
            output = self.model(perturbed.unsqueeze(0))
            probs = torch.softmax(output, dim=1)[0]
        return probs[true_label].item()
    
    def _initialize_population(self, pop_size):
        """Initialize population with random perturbations"""
        population = []
        for _ in range(pop_size):
            # [row, col, r, g, b]
            individual = [
                random.uniform(0, 31),   # row
                random.uniform(0, 31),   # col
                random.gauss(128, 127),  # r
                random.gauss(128, 127),  # g
                random.gauss(128, 127),  # b
            ]
            # Clamp RGB values
            for i in range(2, 5):
                individual[i] = max(0, min(255, individual[i]))
            population.append(individual)
        return population
    
    def _mutation(self, population, F=0.5):
        """
        DE Mutation Strategy: DE/rand/1
        v_i = x_r1 + F * (x_r2 - x_r3)
        """
        mutants = []
        pop_size = len(population)
        
        for i in range(pop_size):
            # Select 3 distinct random indices different from i
            candidates = [j for j in range(pop_size) if j != i]
            r1, r2, r3 = random.sample(candidates, 3)
            
            mutant = []
            for d in range(5):  # 5 dimensions: row, col, r, g, b
                v = population[r1][d] + F * (population[r2][d] - population[r3][d])
                mutant.append(v)
            
            # Constraint handling: clamp to bounds
            mutant[0] = max(0, min(31, mutant[0]))   # row
            mutant[1] = max(0, min(31, mutant[1]))   # col
            for d in range(2, 5):
                mutant[d] = max(0, min(255, mutant[d]))  # RGB
            
            mutants.append(mutant)
        return mutants
    
    def _selection(self, population, trials, fitness_pop, fitness_trials):
        """Selection: keep better individual"""
        new_population = []
        new_fitness = []
        
        for i in range(len(population)):
            if fitness_trials[i] < fitness_pop[i]:  # Lower is better
                new_population.append(trials[i])
                new_fitness.append(fitness_trials[i])
            else:
                new_population.append(population[i])
                new_fitness.append(fitness_pop[i])
        
        return new_population, new_fitness

    def attack(self, image, true_label, max_generations=50, pop_size=50):
        # Initialize
        population = self._initialize_population(pop_size)
        fitness = [self._fitness(ind, image, true_label) for ind in population]
        
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_individual = population[best_idx].copy()
        
        generations_used = 0
        
        for gen in range(max_generations):
            generations_used = gen + 1
            # Early stop if attack already succeeds (misclassified)
            perturbed_best = self._perturb_image(image, best_individual)
            with torch.no_grad():
                out_best = self.model(perturbed_best.unsqueeze(0).to(DEVICE))
                pred_best = out_best.argmax(dim=1).item()
            if pred_best != true_label:
                break
            
            # DE operations
            mutants = self._mutation(population, F=0.5)
            trials = mutants
            fitness_trials = [self._fitness(t, image, true_label) for t in trials]

            population, fitness = self._selection(population, trials, fitness, fitness_trials)

            
            # Update best
            gen_best_idx = np.argmin(fitness)
            if fitness[gen_best_idx] < best_fitness:
                best_fitness = fitness[gen_best_idx]
                best_individual = population[gen_best_idx].copy()
        
        # Apply best perturbation and check result
        perturbed = self._perturb_image(image, best_individual)
        with torch.no_grad():
            output = self.model(perturbed.unsqueeze(0).to(DEVICE))
            new_label = output.argmax(dim=1).item()
        
        success = (new_label != true_label)
        return success, perturbed, new_label, generations_used


# ============== Task 4.1: APR Defense ==============
class PixelWiseAttention(nn.Module):
    """Pixel-wise attention: A(x) = sigmoid(W_a * x + b_a)"""
    def __init__(self, in_channels):
        super().__init__()
        self.attn = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)

    def forward(self, x):
        a = torch.sigmoid(self.attn(x))
        return x * a

class APRResNet18(nn.Module):
    """ResNet-18 with Adaptive Pixel Resilience defense"""
    def __init__(self, num_classes=10):
        super().__init__()
        base = resnet18(weights=None)
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()
        base.fc = nn.Linear(512, num_classes)
        
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.pixel_attention = PixelWiseAttention(64)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.fc = base.fc
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pixel_attention(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return self.fc(torch.flatten(x, 1))


def generate_adversarial_batch(model, images, labels, attacker, num_adv=8):
    model.eval()
    adv_images = []
    adv_labels = []

    indices = random.sample(range(images.size(0)), min(num_adv, images.size(0)))

    for idx in indices:
        img = images[idx]  # already on DEVICE
        label = labels[idx].item()

        success, adv_img, _, _ = attacker.attack(img, label, max_generations=10, pop_size=20)
        if success:
            adv_images.append(adv_img)
            adv_labels.append(label)

    model.train()

    # If none succeeded, fall back to using the original selected images
    if len(adv_images) == 0:
        fallback_imgs = images[indices]
        fallback_lbls = labels[indices]
        return fallback_imgs, fallback_lbls

    return torch.stack(adv_images), torch.tensor(adv_labels, device=DEVICE)



def train_apr_model(model, trainloader, epochs=5, alpha=0.7, beta=0.01):
    """
    Train APR model with:
    1. Adversarial training using actual one-pixel attacks
    2. Gradient regularization: R = beta * ||grad_x L||^2
    """
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Create attacker for adversarial training
    attacker = OnePixelAttack(model)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        
        for i, (images, labels) in enumerate(trainloader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            images.requires_grad = True
            
            # Clean loss
            outputs = model(images)
            loss_clean = criterion(outputs, labels)
            
            # Gradient regularization: R = beta * ||grad_x Loss||^2
            grads = torch.autograd.grad(loss_clean, images, create_graph=True)[0]
            reg_loss = beta * (grads ** 2).mean()
            
            # Adversarial training (every batch, small subset for speed)
            adv_images, adv_labels = generate_adversarial_batch(
                model, images.detach(), labels, attacker, num_adv=8
            )
            adv_outputs = model(adv_images)
            loss_adv = criterion(adv_outputs, adv_labels)

            
            # Combined loss: L = alpha * L_clean + (1-alpha) * L_adv + R
            loss = alpha * loss_clean + (1 - alpha) * loss_adv + reg_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(trainloader):.4f}, '
              f'Acc: {100.*correct/total:.2f}%')
    
    return model


def train_standard_model(model, trainloader, epochs=5):
    """Train standard ResNet-18"""
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(trainloader):.4f}, '
              f'Acc: {100.*correct/total:.2f}%')
    return model


def evaluate_model(model, testloader):
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
    model_a.eval()
    model_b.eval()

    rng = random.Random(seed)
    all_idx = list(range(len(dataset)))
    rng.shuffle(all_idx)

    shared = []
    with torch.no_grad():
        for i in all_idx:
            x, y = dataset[i]
            x = x.unsqueeze(0).to(DEVICE)

            if model_a(x).argmax(dim=1).item() == y and model_b(x).argmax(dim=1).item() == y:
                shared.append(i)
                if len(shared) == k:
                    break

    if len(shared) == 0:
        raise RuntimeError("No shared correctly-classified images found between the two models.")

    if len(shared) < k:
        print(f"[Warning] Requested {k}, got only {len(shared)} shared correct samples.")

    return shared


# ============== Task 3.2 & 4.2: Experiments ==============
def run_attack_experiments(model, testset, indices, model_name="Model"):
    """Run attack experiments and analyze results"""
    print(f"\n{'='*50}")
    print(f"ATTACK ANALYSIS: {model_name}")
    print(f"{'='*50}")
    
    attacker = OnePixelAttack(model)
    model.eval()
    
    results = {
        'success': 0, 'total': 0, 'generations': [],
        'class_vulnerability': defaultdict(lambda: {'attacked': 0, 'success': 0})
    }

    
    num_samples = len(indices)
    for idx in indices:

        
        image, label = testset[idx]

        results['total'] += 1
        success, _, _, gens = attacker.attack(image, label, max_generations=50, pop_size=50)
        
        results['class_vulnerability'][CLASSES[label]]['attacked'] += 1
        if success:
            results['success'] += 1
            results['generations'].append(gens)
            results['class_vulnerability'][CLASSES[label]]['success'] += 1
        
        if results['total'] % 10 == 0:
            print(f"Progress: {results['total']}/{num_samples}")
    
    # Results
    success_rate = 100 * results['success'] / max(1, results['total'])
    print(f"\n1. SUCCESS RATE: {results['success']}/{results['total']} = {success_rate:.2f}%")
    
    if results['generations']:
        print(f"2. EFFICIENCY: Avg generations for success: {np.mean(results['generations']):.1f}")
    
    print(f"3. CLASS VULNERABILITY:")
    for cls in CLASSES:
        data = results['class_vulnerability'][cls]
        if data['attacked'] > 0:
            vuln = 100 * data['success'] / data['attacked']
            print(f"   {cls}: {vuln:.1f}% ({data['success']}/{data['attacked']})")
    
    return success_rate


def main():
    print("="*50)
    print("ONE-PIXEL ATTACK & APR DEFENSE")
    print("="*50)
    
    # Load data
    print("\nLoading CIFAR-10...")
    trainloader, testloader, trainset, testset = get_data_loaders(batch_size=128)
    
    # Train standard model
    print("\n--- Training Standard ResNet-18 ---")
    standard_model = get_resnet18()
    standard_model = train_standard_model(standard_model, trainloader, epochs=5)
    standard_acc = evaluate_model(standard_model, testloader)
    print(f"Standard Model Accuracy: {standard_acc:.2f}%")
    
    # Train APR model
    print("\n--- Training APR Defense Model ---")
    apr_model = APRResNet18().to(DEVICE)
    apr_model = train_apr_model(apr_model, trainloader, epochs=5, alpha=0.7, beta=0.01)
    apr_acc = evaluate_model(apr_model, testloader)
    print(f"APR Model Accuracy: {apr_acc:.2f}%")
    
    DEV_MODE = False  # True while coding, False for report run
    NUM_TEST = 50 if DEV_MODE else 100

    # --- FIX: evaluate on the same images for both models ---
    FIXED_SEED = 0
    shared_indices = get_shared_correct_indices(standard_model, apr_model, testset, NUM_TEST, seed=FIXED_SEED)
    print(f"\nUsing {len(shared_indices)} shared correctly-classified test images for BOTH models.")

    standard_asr = run_attack_experiments(standard_model, testset, shared_indices, "STANDARD")
    apr_asr = run_attack_experiments(apr_model, testset, shared_indices, "APR DEFENSE")

    
    # Final comparison
    print("\n" + "="*50)
    print("FINAL COMPARISON")
    print("="*50)
    print(f"{'Metric':<25} {'Standard':<12} {'APR':<12}")
    print("-"*50)
    print(f"{'Clean Accuracy':<25} {standard_acc:.2f}%{'':<6} {apr_acc:.2f}%")
    print(f"{'Attack Success Rate':<25} {standard_asr:.2f}%{'':<6} {apr_asr:.2f}%")
    print(f"{'Robustness':<25} {100-standard_asr:.2f}%{'':<6} {100-apr_asr:.2f}%")
    
    abs_reduction = standard_asr - apr_asr
    rel_reduction = (abs_reduction / standard_asr * 100) if standard_asr > 0 else 0

    print(f"\n--- Task 4.2 Analysis ---")
    print(f"Q1: Paper effectiveness reproduction:")
    print(f"    Absolute ASR reduction: {abs_reduction:.2f} percentage points")
    print(f"    Relative ASR reduction: {rel_reduction:.2f}%")
    print(f"Q2: Security vs Accuracy trade-off:")
    print(f"    Accuracy drop: {standard_acc-apr_acc:.2f} percentage points")
    print(f"    Robustness gain (absolute): {abs_reduction:.2f} percentage points")
    print(f"Q3: Attacker adaptations:")
    print(f"    - Use multi-pixel attacks (3-5 pixels)")
    print(f"    - Increase DE generations/population")
    print(f"    - Attack with targeted objective instead of untargeted")



if __name__ == "__main__":
    main()