import os
import base64
import tempfile
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

ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_VIDEO_EXT = {".mp4", ".avi", ".mov"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


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


def process_face(frame, box, classifier):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    face_crop = frame[y1:y2, x1:x2]

    if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
        return None

    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    face_tensor = torch.from_numpy(face_rgb).permute(2, 0, 1)
    face_tensor = transform(face_tensor).unsqueeze(0).to(device)

    with torch.no_grad():
        output = classifier(face_tensor).squeeze()
        prob = torch.sigmoid(output).item()

    is_smiling = prob > 0.5
    label = f"Smiling: {prob:.2f}" if is_smiling else f"Not Smiling: {1-prob:.2f}"
    color = (0, 255, 0) if is_smiling else (0, 0, 255)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, y1 - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return {
        "bbox": [x1, y1, x2, y2],
        "label": "Smiling" if is_smiling else "Not Smiling",
        "confidence": round(prob if is_smiling else 1 - prob, 4)
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model: str = Query("efficientnet", enum=["efficientnet", "mobilenet"])
):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_IMAGE_EXT:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Use: jpg, png, webp")

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(400, "Empty file")
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(413, "File too large. Max 50MB")

    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "Could not decode image. File may be corrupted")

    classifier = classifiers[model]
    results = yolo_model(image, conf=0.5, verbose=False)

    faces = []
    for result in results:
        for box in result.boxes:
            face_info = process_face(image, box, classifier)
            if face_info:
                faces.append(face_info)

    _, buffer = cv2.imencode(".jpg", image)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "faces_detected": len(faces),
        "model_used": model,
        "results": faces,
        "annotated_image": img_base64
    }


@app.post("/predict-video")
async def predict_video(
    file: UploadFile = File(...),
    model: str = Query("efficientnet", enum=["efficientnet", "mobilenet"])
):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_VIDEO_EXT:
        raise HTTPException(400, f"Unsupported video type '{ext}'. Use: mp4, avi, mov")

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(400, "Empty file")
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(413, "File too large. Max 50MB")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_in:
        tmp_in.write(contents)
        tmp_in_path = tmp_in.name

    cap = cv2.VideoCapture(tmp_in_path)
    if not cap.isOpened():
        os.unlink(tmp_in_path)
        raise HTTPException(400, "Could not open video")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tmp_out_path = tmp_in_path.replace(ext, "_result.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(tmp_out_path, fourcc, fps, (width, height))

    classifier = classifiers[model]
    frame_count = 0
    total_faces = 0
    skip = 3
    last_faces = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % skip == 0:
            results = yolo_model(frame, conf=0.5, verbose=False)
            last_faces = []
            for result in results:
                for box in result.boxes:
                    face_info = process_face(frame, box, classifier)
                    if face_info:
                        last_faces.append(face_info)
                        total_faces += 1
        else:
            for face in last_faces:
                x1, y1, x2, y2 = face["bbox"]
                color = (0, 255, 0) if face["label"] == "Smiling" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{face['label']}: {face['confidence']:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    with open(tmp_out_path, "rb") as f:
        video_base64 = base64.b64encode(f.read()).decode("utf-8")

    os.unlink(tmp_in_path)
    os.unlink(tmp_out_path)

    return {
        "frames_processed": frame_count,
        "total_faces_detected": total_faces,
        "model_used": model,
        "annotated_video": video_base64
    }