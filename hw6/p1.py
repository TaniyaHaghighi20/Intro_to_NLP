import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchinfo import summary
import time
import numpy as np

# Vision Transformer (ViT) Implementation
class ViT(nn.Module):
    def __init__(self, image_size=32, patch_size=4, in_channels=3, num_classes=100,
                 embed_dim=256, depth=4, num_heads=4, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim,
                                   kernel_size=patch_size, stride=patch_size)

        # CLS token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder
        self.encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim*mlp_ratio,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ) for _ in range(depth)])

        # Layer normalization and classifier
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)  # [B, E, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, E]

        # Add CLS token and position embeddings
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x += self.pos_embed
        x = self.dropout(x)

        # Transformer encoder
        for layer in self.encoder:
            x = layer(x)

        # Classification
        x = self.norm(x[:, 0])
        return self.head(x)

# Modified ResNet-18 for CIFAR-100
def resnet18():
    model = torchvision.models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 100)
    return model

# Training Utilities
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    losses = []
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += preds.eq(targets).sum().item()
    acc = 100 * correct / len(loader.dataset)
    return np.mean(total_loss/len(loader)), acc

# Main Training Procedure
def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 256
    epochs = 30
    lr = 3e-4
    weight_decay = 0.05

    # Data Loading
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    train_set = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model Configurations
    models = {
        'ViT-Tiny (4/256/4)': ViT(patch_size=4, embed_dim=256, depth=4, num_heads=4),
        'ViT-Small (4/512/8)': ViT(patch_size=4, embed_dim=512, depth=8, num_heads=8),
        'ViT-Base (8/512/8)': ViT(patch_size=8, embed_dim=512, depth=8, num_heads=8),
        'ViT-Large (8/768/12)': ViT(patch_size=8, embed_dim=768, depth=12, num_heads=12),
        'ResNet-18': resnet18()
    }

    results = []

    for name, model in models.items():
        print(f"\n=== Training {name} ===")
        model = model.to(device)

        model_stats = summary(model.to('cpu'), (1, 3, 32, 32), verbose=0)
        macs = model_stats.total_mult_adds  # Use correct attribute name
        flops = 2 * macs  # Convert MACs to FLOPs
        params = model_stats.total_params

        # Training Setup
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()

        # Training Loop
        start_time = time.time()
        best_acc = 0
        for epoch in range(epochs):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            scheduler.step()

            if test_acc > best_acc:
                best_acc = test_acc

            print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Test Acc: {test_acc:.2f}%")

        # Record Results
        training_time = time.time() - start_time
        results.append({
            'Model': name,
            'Params (M)': params/1e6,
            'FLOPs (G)': macs/1e9,
            'Time (min)': training_time/60,
            'Accuracy (%)': best_acc
        })

    # Results Table
    print("\n=== Final Results ===")
    print("{:<20} {:<10} {:<10} {:<12} {:<10}".format(
        'Model', 'Params(M)', 'FLOPs(G)', 'Time (min)', 'Accuracy'))
    for res in results:
        print("{:<20} {:<10.2f} {:<10.2f} {:<12.1f} {:<10.2f}".format(
            res['Model'], res['Params (M)'], res['FLOPs (G)'],
            res['Time (min)'], res['Accuracy (%)']))

if __name__ == '__main__':
    main()