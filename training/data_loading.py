import os
import pandas as pd
import torch
from torchvision.io import read_image         
from torchvision.transforms import v2
from torch.utils.data import Dataset          

train_transform = v2.Compose([
    v2.Resize(256),
    v2.CenterCrop((224, 224)),
    v2.RandomHorizontalFlip(),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_transform  = v2.Compose([
    v2.Resize(256),
    v2.CenterCrop((224, 224)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
class mydata(Dataset):                          
    def __init__(self, images_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(images_file)  
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.label_map = {'yes': 1, 'no': 0}    

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)            
        label = self.label_map[self.img_labels.iloc[idx, 1]]  
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label