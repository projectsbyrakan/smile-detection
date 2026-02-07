import os
import cv2
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# Load YOLO face detector
model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
yolo = YOLO(model_path)

# Create output folder
os.makedirs('cropped_faces', exist_ok=True)

# Process all images
failed = []
for i in range(1, 10001):
    img_path = f'images/{i}.jpg'
    image = cv2.imread(img_path)
    if image is None:
        failed.append(i)
        continue
    
    results = yolo(image, conf=0.5, verbose=False)
    
    # Get largest face
    best_box = None
    best_area = 0
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best_box = (x1, y1, x2, y2)
    
    if best_box:
        x1, y1, x2, y2 = best_box
        face_crop = image[y1:y2, x1:x2]
        cv2.imwrite(f'cropped_faces/{i}.jpg', face_crop)
    else:
        # No face detected - copy original
        cv2.imwrite(f'cropped_faces/{i}.jpg', image)
        failed.append(i)
    
    if i % 500 == 0:
        print(f"Processed {i}/10000")

print(f"\nDone! Failed to detect face in {len(failed)} images")
print(f"Failed images: {failed[:20]}{'...' if len(failed) > 20 else ''}")