import cv2 as cv
import torch
from torchvision import transforms

# Standard transform pipeline matching model input
transform_pipeline = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_crop(crop_bgr):
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    
    # 1. BGR -> RGB
    crop_rgb = cv.cvtColor(crop_bgr, cv.COLOR_BGR2RGB)
    
    # 2. Apply transforms: (128, 128, 3) -> Tensor (3, 128, 128)
    tensor = transform_pipeline(crop_rgb)
    
    # 3. Add batch dimension: (3, 128, 128) -> (1, 3, 128, 128)
    tensor = tensor.unsqueeze(0)
    
    return tensor