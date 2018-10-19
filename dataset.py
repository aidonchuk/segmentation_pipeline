import random

import cv2
import numpy as np
import torch
from albumentations.torch.functional import img_to_tensor
from torch.utils.data import Dataset

from feature_ing import add_depth_channels


class SegDataSet(Dataset):
    def __init__(self, file_names, to_augment=False, transform=None, mode='train'):
        self.file_names = file_names
        self.to_augment = to_augment
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        random.seed(random.randint(0, 666))
        np.random.seed(random.randint(0, 666))

        img_file_name = self.file_names[idx]
        image = load_image(img_file_name, self.mode)

        if self.mode == 'test':
            return self.test(image, img_file_name)

        mask = load_mask(img_file_name)
        data = {"image": image, "mask": mask}
        augmented = self.transform(**data)
        image, mask = augmented["image"], augmented["mask"]

        if self.mode == 'train':
            return add_depth_channels(img_to_tensor(image)), torch.from_numpy(np.expand_dims(mask, 0)).float()

    def test(self, image, img_file_name):
        data = {"image": image}
        augmented = self.transform(**data)
        image = augmented["image"]

        return add_depth_channels(img_to_tensor(image)), str(img_file_name)


def load_image(path, mode):
    img = cv2.imread(str('../data/' + mode + '/images/' + path + '.png'))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path):
    mask = cv2.imread(str('../data/train/masks/' + path + '.png'), 0)
    mask[mask > 0] = 1
    return mask.astype(np.uint8)
