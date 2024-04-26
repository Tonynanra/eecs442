from PIL import Image
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import numpy as np
from random import shuffle

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)

class PixelSceneryDataset(Dataset):
    '''
    Adopted from https://github.com/Hexanol777/PixelGAN/blob/main/dataset.py
    '''
    def __init__(self, root_pixel, root_scenery, transform=transforms):
        self.root_pixel = root_pixel
        self.root_scenery = root_scenery
        self.transform = transform

        self.pixel_images = shuffle(os.listdir(root_pixel))
        self.scenery_images = shuffle(os.listdir(root_scenery))
        self.length_dataset = min(len(self.pixel_images), len(self.scenery_images)) # in case of datasets not having the same length
        self.pixel_len = len(self.pixel_images)
        self.scenery_len = len(self.scenery_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        pixel_img = self.pixel_images[index % self.pixel_len]
        scenery_img = self.scenery_images[index % self.scenery_len]

        pixel_path = os.path.join(self.root_pixel, pixel_img)
        scenery_path = os.path.join(self.root_scenery, scenery_img)

        pixel_img = np.array(Image.open(pixel_path).convert("RGB"))
        scenery_img = np.array(Image.open(scenery_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=pixel_img, image0=scenery_img)
            pixel_img = augmentations["image"]
            scenery_img = augmentations["image0"]

        return pixel_img, scenery_img




