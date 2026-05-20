import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import Resize, PILToTensor, Compose, InterpolationMode
from albumentations import (
    Compose as AlbCompose,
    HorizontalFlip,
    VerticalFlip,
    RandomGamma,
    RGBShift,
)

# Default transforms
DEFAULT_TRANSFORM = Compose([
    Resize((512, 512), interpolation=InterpolationMode.BILINEAR),
    PILToTensor()
])

# Default augmentations
DEFAULT_AUGMENTATION = AlbCompose([
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    RandomGamma(gamma_limit=(70, 130), eps=None, always_apply=False, p=0.2),
    RGBShift(p=0.3, r_shift_limit=10, g_shift_limit=10, b_shift_limit=10),
])


class UNetDataClass(Dataset):
    def __init__(self, images_path, masks_path, transform=None):
        super(UNetDataClass, self).__init__()
        
        images_list = sorted(os.listdir(images_path))
        masks_list = sorted(os.listdir(masks_path))
        
        self.images_list = [os.path.join(images_path, img_name) for img_name in images_list]
        self.masks_list = [os.path.join(masks_path, mask_name) for mask_name in masks_list]
        self.transform = transform if transform is not None else DEFAULT_TRANSFORM
        
    def __getitem__(self, index):
        img_path = self.images_list[index]
        mask_path = self.masks_list[index]
        
        # Open image and mask
        data = Image.open(img_path)
        label = Image.open(mask_path)
        
        # Normalize
        data = self.transform(data) / 255.0
        label = self.transform(label) / 255.0
        
        label = torch.where(label > 0.65, 1.0, 0.0)
        
        label[2, :, :] = 0.0001
        label = torch.argmax(label, 0).type(torch.int64)
        
        return data, label
    
    def __len__(self):
        return len(self.images_list)


class SegDataClass(Dataset):
    def __init__(self, images_path, masks_path, transform=None, augmentation=None):
        super(SegDataClass, self).__init__()
        
        images_list = sorted(os.listdir(images_path))
        masks_list = sorted(os.listdir(masks_path))
        
        self.images_list = [os.path.join(images_path, image_name) for image_name in images_list]
        self.masks_list = [os.path.join(masks_path, mask_name) for mask_name in masks_list]
        self.transform = transform if transform is not None else DEFAULT_TRANSFORM
        self.augmentation = augmentation if augmentation is not None else DEFAULT_AUGMENTATION
        
    def __getitem__(self, index):
        img_path = self.images_list[index]
        mask_path = self.masks_list[index]
        
        # Open image and mask
        data = Image.open(img_path)
        label = Image.open(mask_path)
        
        # Augmentation
        if self.augmentation:
            augmented = self.augmentation(image=np.array(data), mask=np.array(label))
            data = Image.fromarray(augmented['image'])
            label = Image.fromarray(augmented['mask'])
        
        # Normalize
        data = self.transform(data) / 255.0
        label = self.transform(label) / 255.0
        
        label = torch.where(label > 0.65, 1.0, 0.0)
        label[2, :, :] = 0.0001
        label = torch.argmax(label, 0).type(torch.int64)
        
        return data, label
    
    def __len__(self):
        return len(self.images_list)


class UNetTestDataClass(Dataset):
    def __init__(self, images_path, transform=None):
        super(UNetTestDataClass, self).__init__()
        
        images_list = sorted(os.listdir(images_path))
        self.images_list = [os.path.join(images_path, i) for i in images_list]
        self.transform = transform if transform is not None else DEFAULT_TRANSFORM
        
    def __getitem__(self, index):
        img_path = self.images_list[index]
        data = Image.open(img_path)
        h = data.size[1]
        w = data.size[0]
        data = self.transform(data) / 255.0        
        return data, img_path, h, w
    
    def __len__(self):
        return len(self.images_list)
