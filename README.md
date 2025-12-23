# NEU-DET Surface Defect Detector

A deep learning pipeline for identifying industrial steel surface defects, achieving **98.9% accuracy** on the NEU-DET dataset.

## 🚀 The Challenge: Domain Adaptation
While the model achieved high accuracy on the training set, initial tests on smartphone photos (real-world data) showed a **Domain Shift**. The model was biased toward the specific "industrial sensor" grain of the original dataset.

### The Solution
I implemented a **Shift-Correction Preprocessing** pipeline using **CLAHE** (Contrast Limited Adaptive Histogram Equalization) and grayscale normalization. This bridges the gap between consumer cameras and industrial sensors by normalizing lighting and enhancing local edge textures.



## 📊 Results
* **Accuracy:** 98.9%
* **Classes:** Crazing, Inclusion, Patches, Pitted, Rolled-in Scale, Scratches.

### Generalization Test: Paint Peeling
When tested on a non-steel surface (peeling paint), the model correctly identified the irregular geometry as **'Patches' with 100% confidence**, proving the robustness of the feature extraction.



## 🛠️ Installation & Usage
1. Clone the repo: `git clone https://github.com/Nedu21/Industrial-Defect-Detection.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run inference on your own image: 
   ```bash
   Usage: python src/predict_new.py <IMAGE_PATH> <MODEL_PATH>
   python src/predict_new.py pitted.jpg best_neu_model.pth
   ```