import torch
import validate as ev
import data
import models as models

import torch.nn as nn

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = total_correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss    += loss.item() * images.size(0)
        total_correct += (outputs.argmax(1) == labels).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    avg_acc  = total_correct / len(loader.dataset)
    return avg_loss, avg_acc

def run_training(data_dir, epochs, batch_size, lr):
    train_loader, test_loader, classes = data.get_dataloaders(data_dir=data_dir, batch_size=batch_size)

    model = models.build_model(classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    for epoch in range(1, epochs+1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = ev.validate_one_epoch(model, test_loader, criterion, device)
        
        print(f"Epoch {epoch}: train {train_loss:.3f}/{train_acc:.3f}, "
              f"val {val_loss:.3f}/{val_acc:.3f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_resnet18.pth")