# Cataract Detector from Webcam

A deep learning pipeline that screens for cataracts from a live webcam feed or still images. A MediaPipe face landmarker locates and crops the iris/eye region, and a custom PyTorch classifier predicts whether a cataract is present.

> ⚠️ **Disclaimer:** This is an educational/research project. It is **not** a medical device and must not be used for diagnosis. If you have concerns about your vision, see a qualified eye-care professional.

---

## 📑 Table of Contents

- [Key Features](#-key-features)
- [How It Works](#-how-it-works)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Training Details](#-training-details)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🌟 Key Features

- **High accuracy:** 91.58% test accuracy on the evaluation dataset.
- **Smart preprocessing:** MediaPipe FaceLandmarker automatically detects the face and crops the iris/eye region, so the model only sees the features that matter.
- **PyTorch backend:** a custom deep learning architecture built and trained from scratch with PyTorch.
- **Runs locally:** trained on GPU in Google Colab, but inference runs on your own machine with a regular webcam.

---

## 🔬 How It Works

```
Webcam / Image  →  MediaPipe FaceLandmarker  →  Iris/Eye Crop  →  PyTorch Classifier  →  Prediction
```

1. **Capture:** a frame is read from the webcam (or loaded from an image).
2. **Detect:** MediaPipe FaceLandmarker finds facial landmarks, including the iris.
3. **Crop:** the eye/iris region is extracted and preprocessed.
4. **Classify:** the cropped region is passed to the trained PyTorch model (`catmodel_weights.pth`).
5. **Output:** the predicted result is displayed.

---

## 📁 Repository Structure

| File | Description |
|------|-------------|
| `main.py` | Main script to run the cataract detection application |
| `utilis.py` | Helper functions for image processing, cropping, and model operations |
| `catmodel_weights.pth` | Trained PyTorch model weights |
| `face_landmarker.task` | MediaPipe task file required for facial landmark and iris detection |
| `Model_Training.ipynb` | Google Colab notebook used to build and train the model |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or newer
- A working webcam (for live detection)
- `pip` for installing dependencies

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/not7devansh/Cataract-Detector-from-webcam.git
   ```

2. **Navigate to the project folder**

   ```bash
   cd Cataract-Detector-from-webcam
   ```

3. **(Recommended) Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate      # macOS / Linux
   venv\Scripts\activate         # Windows
   ```

4. **Install dependencies**

   ```bash
   pip install torch torchvision mediapipe opencv-python
   ```

---

## ▶️ Usage

Run the main inference script:

```bash
python main.py
```

**Tips for best results:**

- Face the camera directly in a well-lit room.
- Keep your eyes open and look straight ahead.
- Avoid glare, heavy shadows, or glasses reflections over the eyes.
- Keep your face reasonably close to the camera so the iris is clearly visible.

---

## 🧠 Training Details

The model was trained in **Google Colab** to take advantage of GPU acceleration. The full pipeline is documented in [`Model_Training.ipynb`](Model_Training.ipynb):

1. Loading the dataset
2. Applying MediaPipe iris extraction
3. Building the PyTorch model
4. Training and evaluation, reaching **91.58% test accuracy**

---

## ⚠️ Limitations

- Accuracy was measured on a specific dataset and may drop on real-world webcam conditions (lighting, camera quality, angle).
- Webcam images are far lower quality than clinical slit-lamp or fundus imaging.
- Results are a screening-style prediction only, not a diagnosis.
- Performance may vary across ages, skin tones, and eye colors depending on dataset coverage.

---

## 🔮 Future Improvements

- [ ] Add a confusion matrix and precision/recall/F1 metrics
- [ ] Support batch prediction on a folder of images
- [ ] Add on-screen confidence scores
- [ ] Test on more diverse, real-world data
- [ ] Package as a simple GUI or web app

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to open an issue or submit a pull request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m "Add amazing feature"`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a pull request

---

## 👤 Author

**Devansh** — [@not7devansh](https://github.com/not7devansh)
