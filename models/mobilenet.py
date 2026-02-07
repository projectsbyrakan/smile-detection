import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

def get_model():
    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    model.classifier = nn.Sequential(
        nn.Linear(960, 1280),
        nn.Hardswish(),
        nn.Dropout(p=0.2),
        nn.Linear(1280, 1)
    )
    return model

config = {
    "architecture": "MobileNet_V3_Large",
    "run_name": "mobilenet_v3_large",
    "save_path": "weights/best_mobilenet.pth",
}