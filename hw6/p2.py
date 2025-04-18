import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from transformers import SwinForImageClassification, SwinConfig
import time
import numpy as np

# Data Transforms with ImageNet normalization
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load CIFAR-100 datasets
train_set = torchvision.datasets.CIFAR100(
    root='./data', train=True, download=True, transform=train_transform)
test_set = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4)

def create_model(model_type, pretrained=True):
    """Create Swin Transformer model with 100-class classifier"""
    if pretrained:
        model = SwinForImageClassification.from_pretrained(
            f"microsoft/{model_type}-patch4-window7-224",
            ignore_mismatched_sizes=True
        )
        # Freeze all parameters except classifier
        for param in model.parameters():
            param.requires_grad = False
        # Replace and unfreeze classifier
        model.classifier = nn.Linear(model.config.hidden_size, 100)
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        # Create from config for scratch training
        config = SwinConfig.from_pretrained(f"microsoft/{model_type}-patch4-window7-224")
        config.num_labels = 100
        model = SwinForImageClassification(config)

    return model

def train_model(model, name, epochs=5, lr=2e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    results = {'name': name, 'times': [], 'accuracies': []}

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        correct = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).logits
                preds = outputs.argmax(dim=1)
                correct += preds.eq(labels).sum().item()

        epoch_time = time.time() - start_time
        accuracy = 100 * correct / len(test_set)

        results['times'].append(epoch_time)
        results['accuracies'].append(accuracy)
        print(f"{name} Epoch {epoch+1}/{epochs} | Time: {epoch_time:.1f}s | Acc: {accuracy:.2f}%")

    return results

# Initialize models
models = {
    "Swin-Tiny (Pretrained)": create_model("swin-tiny", pretrained=True),
    "Swin-Small (Pretrained)": create_model("swin-small", pretrained=True),
    "Swin-Tiny (Scratch)": create_model("swin-tiny", pretrained=False),
    "Swin-Small (Scratch)": create_model("swin-small", pretrained=False),
}

# Train and evaluate models
all_results = []
for name, model in models.items():
    print(f"\n=== Training {name} ===")
    results = train_model(model, name)
    all_results.append({
        'Model': name,
        'Avg Time/Epoch (s)': np.mean(results['times']),
        'Best Accuracy (%)': np.max(results['accuracies']),
        'Params (M)': sum(p.numel() for p in model.parameters()) / 1e6
    })

# Print results table
print("\n=== Final Results ===")
print("{:<25} {:<15} {:<15} {:<15}".format(
    'Model', 'Params(M)', 'Time/Epoch', 'Accuracy'))
for res in all_results:
    print("{:<25} {:<15.1f} {:<15.1f} {:<15.2f}".format(
        res['Model'], res['Params (M)'],
        res['Avg Time/Epoch (s)'], res['Best Accuracy (%)']))