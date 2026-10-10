# Real-Time Hand Gesture Recognition

## 🎯 Project Goal
Transition hand gesture classification from static image processing to a **real-time, live webcam application** using lightweight landmark features.

---

## 🏗️ Architecture (Clean Enterprise Structure)
We separate code into 4 simple layers so it is easy to test, maintain, and scale:

```
├── core/            # Domain Logic (Landmark normalization & feature rules)
├── services/        # Business Logic (Model inference engine & frame smoothing buffer)
├── adapters/        # External Drivers (MediaPipe hand detector & OpenCV webcam feed)
└── main.py          # Entry Point (Main application runner loop)
```

---

## 🤖 AI Concepts (Simplified)
- **MediaPipe HandLandmarker**: Converts raw camera pixels into **21 3D coordinate points** (landmarks) for each hand.
- **Landmark MLP (Multi-Layer Perceptron)**: A small, fast neural network trained on normalized coordinates instead of heavy images.
- **Feature Normalization**: Shifts and scales landmark coordinates so predictions work regardless of hand size or position on camera.
- **Prediction Smoothing**: A rolling buffer that prevents rapid flickering between `CORRECT`, `INCORRECT`, and `UNCERTAIN`.

---

## 🚀 5-Step Roadmap

### Step 1: Build the Dataset
Organize gesture images into structured folders:
```
dataset/
├── train/
│   └── gesture_name/
│       ├── correct/
│       └── wrong/
└── val/
```

### Step 2: Generate Pairs & Train Model
1. Run `make-pairs` to create `pairs.csv`.
2. Train the model:
   ```bash
   python train.py --model landmark_mlp
   ```
3. Output saved to: `outputs/models/landmark_mlp/` (`model.pt` & `meta.json`).

### Step 3: Write Real-Time Inference Script
Create a script combining **OpenCV** (webcam input) + **MediaPipe** (landmarks) + **PyTorch Model** (prediction).

### Step 4: Add Smoothing & Cooldown
Add a frame queue buffer so consecutive predictions remain stable.

### Step 5: Test & Deploy Locally
1. Install requirements:
   ```bash
   py -m pip install -r requirements.txt
   ```
2. Run the live webcam loop:
   ```bash
   py main.py
   ```


