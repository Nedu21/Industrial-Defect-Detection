import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from model import NEU_CNN


def predict_single_image(image_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = ['Crazing', 'Inclusion', 'Patches',
               'Pitted', 'Rolled-in Scale', 'Scratches']

    # --- STEP 1: SOLVE COVARIANCE SHIFT (DOMAIN ADAPTATION) ---
    # Load with OpenCV for advanced preprocessing
    raw_img = cv2.imread(image_path)

    # 1. Convert to Grayscale (Neutralizes phone color sensor bias)
    gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)

    # 2. Apply CLAHE (Standardizes lighting and brings out faint textures)
    # This makes your camera photo "feel" like an industrial sensor image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)

    # 3. Revert to 3-channel (RGB) because your model expects 3 channels
    processed_img = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)

    # 4. Convert back to PIL for the standard PyTorch transforms
    img = Image.fromarray(processed_img)

    # --- STEP 2: STANDARD TRANSFORMS ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5042, 0.5042, 0.5042],
                             std=[0.2058, 0.2058, 0.2058])
    ])

    img_tensor = transform(img).unsqueeze(0).to(device)

    # --- STEP 3: PREDICTION ---
    model = NEU_CNN(num_classes=6).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        output = model(img_tensor)
        percentage = torch.nn.functional.softmax(output, dim=1)[0] * 100
        conf, pred_idx = torch.max(percentage, 0)

    print(f"\n--- Prediction Results  ---")
    print(f"Defect Detected: {classes[pred_idx]}")
    print(f"Confidence: {conf.item():.2f}%")

    print("\nFull Breakdown:")
    for i, score in enumerate(percentage):
        print(f"{classes[i]}: {score.item():.2f}%")


if __name__ == "__main__":
    # Ensure you have 'opencv-python' installed: pip install opencv-python
    predict_single_image('pitted.jpg', 'best_neu_model.pth')
