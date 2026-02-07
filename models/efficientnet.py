import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

def get_model():
    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(1280, 1)
    )
    return model

config = {
    "architecture": "EfficientNet_V2_S",
    "run_name": "efficientnet_v2_s",
    "save_path": "weights/best_efficientnet.pth",
}