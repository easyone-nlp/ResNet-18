import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def train_model(model, num_epochs=10, resume=False, checkpoint_path="checkpoint.pth", num_classes=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    batch_size = 32
    learning_rate = 0.001

    # ResNet expects 3-channel inputs, so MNIST is expanded from 1 channel to 3.
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)),
        ]
    )

    full_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    train_size = int(0.7 * len(full_dataset))
    valid_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - valid_size
    train_set, valid_set, test_set = random_split(full_dataset, [train_size, valid_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    start_epoch = 0
    if resume and os.path.exists(checkpoint_path):
        print(f"🔁 Checkpoint found. Loading from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"✅ Resumed from epoch {start_epoch}")

    train_accuracies = []
    valid_accuracies = []
    train_losses = []

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            _, pred = output.max(1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)

        train_acc = 100 * correct / total
        train_loss = total_loss / total
        train_accuracies.append(train_acc)
        train_losses.append(train_loss)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                _, pred = output.max(1)
                correct += pred.eq(y).sum().item()
                total += y.size(0)

        valid_acc = 100 * correct / total
        valid_accuracies.append(valid_acc)

        print(
            f"📘 Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Valid Acc: {valid_acc:.2f}%"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            checkpoint_path,
        )

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            _, pred = output.max(1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)

    print(f"🧪 Test Accuracy: {100 * correct / total:.2f}%")

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(train_accuracies, label="Train Acc")
        plt.plot(valid_accuracies, label="Valid Acc")
        plt.plot(train_losses, label="Train Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("Training Progress on MNIST")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception as exc:
        print(f"Skipping plot display due to matplotlib import/runtime issue: {exc}")
