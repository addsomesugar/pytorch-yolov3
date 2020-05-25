import glob
import os
import random
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from utils.augmentations import horizontal_flip


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    pad = [0, 0, pad1, pad2] if h <= w else [pad1, pad2, 0, 0]
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class YOLODataset(Dataset):
    def __init__(self, image_label_list_file, img_size=416, augment=True):
        self.image_files, self.label_files = np.loadtxt(image_label_list_file, dtype=str).T.tolist()
        self.img_size = img_size
        self.augment = augment

    def __getitem__(self, index):
        # ---------
        #  Image
        # ---------
        img_path = self.image_files[index % len(self.image_files)].rstrip()

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = h, w
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------
        label_path = self.label_files[index % len(self.image_files)].rstrip()

        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[0]
            y2 += pad[2]
            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes
        img = resize(img, self.img_size)
        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horizontal_flip(img, targets)

        return img_path, img, targets

    @staticmethod
    def collate_fn(batch):
        paths, imgs, targets = zip(*batch)
        targets = [boxes for boxes in targets if boxes is not None]
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        imgs = torch.stack(imgs)
        return paths, imgs, targets

    def __len__(self):
        return len(self.image_files)


class PascalDataset(Dataset):
    def __init__(self, image_label_list_file, names_file, img_size=416, augment=True):
        self.image_files, self.label_files = np.loadtxt(image_label_list_file, dtype=str).T.tolist()
        with open(names_file, 'r') as names_f:
            self.names = names_f.readlines()
        self.names = [name.strip() for name in self.names if name != '']
        print(self.names)
        self.img_size = img_size
        self.augment = augment

    def get_names(self):
        return self.names

    def __getitem__(self, index):
        # ----------
        # image
        # ----------
        img_path = self.image_files[index % len(self.image_files)].rstrip()
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape

        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape
        # ----------
        # label
        # ----------
        label_path = self.label_files[index % len(self.image_files)].rstrip()

        targets = None
        if os.path.exists(label_path):
            tree = ET.parse(label_path)
            width = tree.find('size').find('width').text
            height = tree.find('size').find('height').text
            assert w == int(width)
            assert h == int(height)
            boxes = []
            for object_iter in tree.findall("object"):
                bndbox = object_iter.find('bndbox')
                label = object_iter.find('name').text
                xmin = int(float(bndbox.find('xmin').text))
                ymin = int(float(bndbox.find('ymin').text))
                xmax = int(float(bndbox.find('xmax').text))
                ymax = int(float(bndbox.find('ymax').text))
                xmin += pad[0]
                ymin += pad[2]
                xmax += pad[1]
                ymax += pad[3]
                box = np.zeros(5, dtype=np.float)
                box[0] = self.names.index(object_iter.find('name').text)
                box[1] = ((xmin + xmax) / 2) / padded_w
                box[2] = ((ymin + ymax) / 2) / padded_h
                box[3] = (xmax - xmin) / padded_w
                box[4] = (ymax - ymin) / padded_h
                boxes.append(box.tolist())
            boxes = torch.tensor(boxes)
            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes
        img = resize(img, self.img_size)
        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horizontal_flip(img, targets)

        return img_path, img, targets

    @staticmethod
    def collate_fn(batch):
        paths, imgs, targets = zip(*batch)
        targets = [boxes for boxes in targets if boxes is not None]
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        imgs = torch.stack(imgs)
        return paths, imgs, targets

    def __len__(self):
        return len(self.image_files)





