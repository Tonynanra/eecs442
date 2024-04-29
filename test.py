import torch
from generator import Generator
from torchvision.utils import save_image
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

def transforms(image):
    transforms = A.Compose([   
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
        ])
    return transforms(image=image)
scenery= transforms(np.array(Image.open("test3.jpg").convert("RGB")))['image']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G_pix = Generator(img_channels=3, num_residuals=9).to(device)
checkpoint=torch.load("train_result.pth")
G_pix.load_state_dict(checkpoint['G_pix_state_dict'])
G_pix.train()
scenery = scenery.to(device)
with torch.cuda.amp.autocast():

    fake_pixel = G_pix(scenery)
    save_image(fake_pixel * 0.5 + 0.5, "pixel_test.png")