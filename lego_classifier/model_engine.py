import torch

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = total_correct = 0
    for images, ground_truths in loader:
        images, ground_truths = images.to(device), ground_truths.to(device)

        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, ground_truths)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (predictions.argmax(1) == ground_truths).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    avg_acc  = total_correct / len(loader.dataset)
    return avg_loss, avg_acc

def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = total_correct = 0
    with torch.no_grad():
        for images, ground_truths in loader:
            images, ground_truths = images.to(device), ground_truths.to(device)

            predictions = model(images)
            loss = criterion(predictions, ground_truths)

            total_loss += loss.item() * images.size(0)
            total_correct += (predictions.argmax(1) == ground_truths).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    avg_acc  = total_correct / len(loader.dataset)
    return avg_loss, avg_acc