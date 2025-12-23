import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Import your own tools
from model import NEU_CNN
from dataset import get_dataloaders


def generate_metrics(model_path, base_path='./NEU-DET'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data (We only need the validation loader)
    _, val_loader, classes = get_dataloaders(base_path)

    # 2. Load Model
    model = NEU_CNN(num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    # 3. Collect Predictions
    print("Gathering predictions from validation set...")
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 4. Create the Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)

    # 5. Visualize with Seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)

    plt.xlabel('Predicted Label', fontweight='bold')
    plt.ylabel('Actual Label', fontweight='bold')
    plt.title('NEU-DET Confusion Matrix', fontsize=14)
    plt.show()

    # 6. Print the Text Report (Precision, Recall, F1)
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))


if __name__ == "__main__":
    generate_metrics('best_neu_model.pth')
