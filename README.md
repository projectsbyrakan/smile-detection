# Smile Detection

Binary classification of facial expressions (**smiling** vs **not smiling**) using YOLO for face detection and CNN classifiers for smile prediction.

**Live Demo**: https://smile-detection-vwm9.onrender.com

## Project Structure

```
├── api/
│   └── main.py                  # FastAPI backend (/predict, /predict-video)
├── app/
│   ├── streamlit_app.py         # Streamlit frontend (detection page)
│   └── pages/
│       └── 1_Report.py          # Model comparison report page
├── models/
│   ├── efficientnet.py          # EfficientNet V2-S config
│   └── mobilenet.py             # MobileNetV3 Large config
├── training/
│   ├── train.py                 # Unified training script
│   ├── data_loading.py          # PyTorch Dataset + transforms
│   └── crop_faces.py            # YOLO face cropping preprocessor
├── weights/
│   ├── best_efficientnet.pth    # Trained EfficientNet weights
│   └── best_mobilenet.pth       # Trained MobileNet weights
├── data/
│   └── image_with_label.csv     # Image labels (yes/no)
├── notebooks/
│   └── EDA.ipynb                # Exploratory data analysis
├── Dockerfile
├── Requirements.txt
└── README.md
```

## How to Use

### Option 1: Live Demo

Visit https://smile-detection-vwm9.onrender.com

1. Select a model (EfficientNet or MobileNet).
2. Choose input type (Image or Video).
3. Upload a file and click **Detect Smiles**.

### Option 2: Run Locally

```bash
# Clone the repo
git clone https://github.com/projectsbyrakan/smile-detection.git
cd smile-detection

# Install dependencies
pip install -r Requirements.txt

# Start the API (terminal 1)
uvicorn api.main:app --port 8000

# Start the frontend (terminal 2)
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501` in your browser.

### Option 3: Docker

```bash
docker build -t smile-detection .
docker run -p 8000:8000 -p 8501:8501 smile-detection
```

## Report

### EDA Findings

- **Problem**: Binary classification of facial expression (smiling vs not smiling) from images.
- **Images**: Faces are always centered; lighting, pose, and background vary significantly.
- **Class imbalance**: The dataset is skewed toward smiling (test support: 1055 smiling vs 445 not smiling), which can bias models toward predicting the majority class.
- **Key insight**: A major accuracy risk is train-inference mismatch (training on full images but inferring on cropped faces). This was addressed by cropping faces for training/inference consistency.

### Data Preprocessing

- **Face cropping**: YOLOv8 Face Detection to detect and crop the largest face per image, ensuring training data matches inference input.
- **Transforms**: Resize to 256, CenterCrop to 224x224, ImageNet normalization.
- **Augmentation (train only)**: Random horizontal flip, color jitter (brightness/contrast/saturation).
- **Split**: 70/15/15 train/validation/test with a held-out test set.

### Training Process

Both models followed the same protocol, logged with MLflow.

**Experiment 1: EfficientNet V2-S**
- Base model: `efficientnet_v2_s` pretrained on ImageNet.
- Two-stage fine-tuning:
  - Stage 1: freeze backbone, train classifier head.
  - Stage 2: unfreeze backbone, fine-tune with differential learning rates.
- Optimization: Adam, ReduceLROnPlateau, BCEWithLogitsLoss with pos_weight.

**Experiment 2: MobileNetV3 Large**
- Base model: `mobilenet_v3_large` pretrained on ImageNet.
- Same two-stage strategy and data pipeline.
- Goal: compare a lighter/faster model vs EfficientNet.

### Evaluation Results

Test set: 1500 samples (not_smiling=445, smiling=1055). Positive class = smiling.

| Metric | EfficientNet V2-S | MobileNetV3 Large |
|--------|-------------------|-------------------|
| Accuracy | **0.89** | 0.87 |
| Weighted F1 | **0.89** | 0.88 |
| Type I errors (FP) | **34** (FPR: 7.64%) | 51 (FPR: 11.46%) |
| Type II errors (FN) | **136** (FNR: 12.89%) | 138 (FNR: 13.08%) |
| Smiling Precision | **0.96** | 0.95 |
| Smiling Recall | 0.87 | 0.87 |
| Not Smiling Precision | **0.75** | 0.74 |
| Not Smiling Recall | **0.92** | 0.89 |

**Strengths**: High precision for smiling (0.95-0.96) means when the model says "smiling", it is usually correct.

**Weaknesses**: Both models miss ~13% of true smiles (FNR ~13%), driven by smiling recall of 0.87.

**Key difference**: EfficientNet reduces Type I (false smile) errors more effectively: 7.64% vs 11.46%.

### Deployment

- **Backend**: FastAPI with `/predict` (images) and `/predict-video` (videos) endpoints.
- **Frontend**: Streamlit with Detection page and Report page.
- **Containerization**: Docker packages the full app (API + frontend + weights).
- **Cloud**: Deployed on Render with automatic redeployment from GitHub.

### Future Improvements

- Add more augmentations (rotations, blur, random crop) to handle motion blur and pose variation.
- Video optimization: process every Nth frame and batch faces for faster inference.
- Build a model that does classification on semantic embeddings.
