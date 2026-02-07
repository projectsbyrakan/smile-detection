import os
import base64
import cv2
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from torchvision.transforms import v2
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from models.efficientnet import get_model as get_efficientnet, config as effnet_config
from models.mobilenet import get_model as get_mobilenet, config as mobile_config

app = FastAPI(title="Smile Detection API")

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Global variables (loaded at startup)
yolo_model = None
classifiers = {}
transform = None
device = None


@app.on_event("startup")
def load_models():
    global yolo_model, classifiers, transform, device

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # YOLO
    model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
    yolo_model = YOLO(model_path)

    # EfficientNet
    effnet = get_efficientnet()
    effnet.load_state_dict(torch.load(effnet_config["save_path"], map_location=device))
    effnet.to(device)
    effnet.eval()
    classifiers["efficientnet"] = effnet

    # MobileNet
    mobnet = get_mobilenet()
    mobnet.load_state_dict(torch.load(mobile_config["save_path"], map_location=device))
    mobnet.to(device)
    mobnet.eval()
    classifiers["mobilenet"] = mobnet

    # Transform
    transform = v2.Compose([
        v2.Resize(256),
        v2.CenterCrop((224, 224)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("All models loaded!")


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model: str = Query("efficientnet", enum=["efficientnet", "mobilenet"])
):
    # Validate extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Use: jpg, png, webp")

    # Read and validate size
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(400, "Empty file")
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(413, "File too large. Max 10MB")

    # Decode image
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "Could not decode image. File may be corrupted")

    # Run YOLO
    classifier = classifiers[model]
    results = yolo_model(image, conf=0.5, verbose=False)

    # Process each face
    faces = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face_crop = image[y1:y2, x1:x2]

            if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
                continue

            # Classify
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            face_tensor = torch.from_numpy(face_rgb).permute(2, 0, 1)
            face_tensor = transform(face_tensor).unsqueeze(0).to(device)

            with torch.no_grad():
                output = classifier(face_tensor).squeeze()
                prob = torch.sigmoid(output).item()

            is_smiling = prob > 0.5
            label = f"Smiling: {prob:.2f}" if is_smiling else f"Not Smiling: {1-prob:.2f}"
            color = (0, 255, 0) if is_smiling else (0, 0, 255)

            # Draw on image
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            faces.append({
                "bbox": [x1, y1, x2, y2],
                "label": "Smiling" if is_smiling else "Not Smiling",
                "confidence": round(prob if is_smiling else 1 - prob, 4)
            })

    # Encode annotated image to base64
    _, buffer = cv2.imencode(".jpg", image)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "faces_detected": len(faces),
        "model_used": model,
        "results": faces,
        "annotated_image": img_base64
    }