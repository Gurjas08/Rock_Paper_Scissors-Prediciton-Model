# Rock-Paper-Scissors Gesture Classification (Transfer Learning)

A computer vision project that classifies hand-gesture images into **rock**, **paper**, or **scissors** using **transfer learning** with **MobileNetV2** from TensorFlow Hub. The workflow covers dataset setup, preprocessing, model training, evaluation, and single-image inference.

## Overview
**Goal:** Predict rock/paper/scissors from an input image.  
**Approach:** Use a pre-trained MobileNetV2 feature extractor and train a small classification head for 3 classes.

## Dataset
- Source: Kaggle Rock-Paper-Scissors Dataset  
- Classes: `rock`, `paper`, `scissors`  
- Structure used:
  - `Rock-Paper-Scissors/train/`
  - `Rock-Paper-Scissors/test/`
  - `Rock-Paper-Scissors/validation/` (used for inference examples)

## Tech Stack
- Python
- TensorFlow / Keras (`tf_keras`)
- TensorFlow Hub (`tensorflow_hub`)
- NumPy, Pillow
- Matplotlib

## Method
### 1) Data Loading and Preprocessing
- Loaded images from class folders (`rock`, `paper`, `scissors`)
- Resized images to **224×224**
- Normalized pixel values to **[0, 1]**
- One-hot encoded labels for 3-class classification

### 2) Model Architecture (Transfer Learning)
- Feature extractor: **MobileNetV2** from TF Hub  
  `https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4`
- Classification head: `Dense(3, activation="softmax")`
- The feature extractor is frozen (`trainable=False`) and only the final dense layer is trained.

### 3) Training
- Optimizer: Adam  
- Loss: Categorical Crossentropy  
- Metric: Accuracy  
- Trained for a small number of epochs (example run: 5 epochs)

## Results
On the held-out test set, an example evaluation run achieved:
- **Test Accuracy:** **0.7312**
- **Test Loss:** **0.6454**

Note: Results can vary depending on training time/epochs and runtime environment.

## Inference (Single Image)
The notebook includes an example that:
- Loads one image (e.g., `Rock-Paper-Scissors/validation/rock-hires2.png`)
- Applies the same resize and normalization steps
- Runs `model.predict(...)`
- Outputs predicted class among `rock`, `paper`, `scissors`

## How to Run
### Option A: Google Colab (recommended)
1. Open the notebook in Colab
2. Upload your `kaggle.json` (Kaggle API key)
3. Run dataset download + unzip cells
4. Run training and evaluation
5. Try the inference cell on a sample image

### Option B: Local Setup
Install dependencies:
```bash
pip install tensorflow tensorflow-hub tf_keras pillow numpy matplotlib
```

## Notes / Improvements
If I extend this project further I would:
- Train for more epochs with early stopping
- Add a confusion matrix + per-class precision/recall
- Fine-tune upper MobileNetV2 layers (instead of fully freezing)
- Package inference as a small web app (Streamlit/Flask)

## Acknowledgement
- MobileNetV2 feature extractor from TensorFlow Hub
- Kaggle Rock-Paper-Scissors dataset
