import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src.dataset import get_dataloaders
from src.model import NEU_CNN

# We pass the 'writer' into the function so it can log data


def train_model(dataloader, model, criterion, optimizer, device, writer, epoch):
    dataset_size = len(dataloader.dataset)
    model.train()
    train_loss, correct = 0, 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        # Forward
        yhat = model(X)
        loss = criterion(yhat, y)

        # Robustness: Skip if math break
        if torch.isnan(loss):
            continue

        # Backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item() * len(X)
        correct += (yhat.argmax(1) == y).type(torch.float).sum().item()

    avg_loss = train_loss / dataset_size
    avg_acc = correct / dataset_size

    # Log training metrics to TensorBoard
    writer.add_scalar('Loss/Train', avg_loss, epoch)
    writer.add_scalar('Accuracy/Train', avg_acc, epoch)

    return avg_loss, avg_acc


def validate_model(dataloader, model, criterion, device, writer, epoch):
    dataset_size = len(dataloader.dataset)
    model.eval()
    val_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            yhat = model(X)
            loss = criterion(yhat, y)

            val_loss += loss.item() * len(X)
            correct += (yhat.argmax(1) == y).type(torch.float).sum().item()

    avg_loss = val_loss / dataset_size
    avg_acc = correct / dataset_size

    # Log validation metrics to TensorBoard
    writer.add_scalar('Loss/Val', avg_loss, epoch)
    writer.add_scalar('Accuracy/Val', avg_acc, epoch)

    return avg_loss, avg_acc


def main():
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Hyperparameters
    Epochs = 60
    base_path = os.path.join(os.getcwd(), 'NEU-DET')

    # initialize tools
    train_loader, val_loader, classes = get_dataloaders(base_path)
    model = NEU_CNN(num_classes=len(classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=3, factor=0.5)

    # Initialize TensorBoard (Create a 'runs' folder)
    writer = SummaryWriter('runs/neu_det_experiment')

    best_val_loss = float('inf')

    # The main loop
    for epoch in range(Epochs):
        print(f"Epoch {epoch+1}/{Epochs}")

        train_loss, train_acc = train_model(
            train_loader, model, criterion, optimizer, device, writer, epoch)
        val_loss, val_acc = validate_model(
            val_loader, model, criterion, device, writer, epoch)

        # Logic for "High Score" and Scheduler
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_neu_model.pth')
            print(f"--> Saved Best Model! (Val Loss: {val_loss:.4f})")

        print(
            f"Train Acc: {100*train_acc:.1f}% | Val Acc: {100*val_acc:.1f}%\n")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}\n")

    # Cleanup
    writer.close()
    print("Training Finished!")


if __name__ == "__main__":
    main()
