import os
import cv2
import numpy as np
import random
from PIL import Image
import pandas as pd

import torch
import torch.utils.data as data
from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset

import albumentations as A
from albumentations.pytorch import ToTensorV2


class ImageDataset(data.Dataset):
    def __init__(
            self,
            data,
            transform=None,
            mode='train',
    ):
        self.data = data
        self.transform = transform
        self.mode = mode

    def __getitem__(self, index):
        img, target = self.data[index]
        

        img = cv2.resize(img, (512, 512))
        # img = cv2.resize(img, None, fx=3, fy=3)
        # img = cv2.resize(img, (img.shape[1]//32*32, img.shape[0]//32*32))
        img = img.transpose(2,1,0)
        img = torch.from_numpy(img).float()

        # smooth label
        if self.mode == 'train' and 1:
            target = 0.8*float(pred) + 0.2*float(target)

        if target is None:
            target = torch.tensor(-1, dtype=torch.float)
        else:
            target = torch.tensor(float(target), dtype=torch.float)

        return img, target

    def __len__(self):
        return len(self.data)

def get_transforms(mode):
    if mode == 'train':
        return A.Compose([
            A.OneOf([
                A.Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, p=0.3),
                A.GridDropout (ratio=0.5, p=0.3),
                A.CoarseDropout (max_holes=8, max_height=8, max_width=8, p=0.3),
                ], p=1),
            # A.ChannelDropout(p=0.5),
        ])

    elif mode == 'valid':
        return None
    else:
        return None
