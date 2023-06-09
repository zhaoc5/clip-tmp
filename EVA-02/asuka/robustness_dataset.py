# --------------------------------------------------------
# EVA-02: A Visual Representation for Neon Genesis
# Github source: https://github.com/baaivision/EVA/EVA02
# Copyright (c) 2023 Beijing Academy of Artificial Intelligence (BAAI)
# Licensed under The MIT License [see LICENSE for details]
# By Yuxin Fang
#
# Based on EVA: Exploring the Limits of Masked Visual Representation Learning at Scale (https://arxiv.org/abs/2211.07636)
# https://github.com/baaivision/EVA/tree/master/EVA-01
# --------------------------------------------------------'



import json
from torch.utils import data
from torchvision.datasets import ImageFolder
import torch
import os
from PIL import Image
import numpy as np
import argparse
from tqdm import tqdm
import multiprocessing
from multiprocessing import Process, Manager
import collections
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
import torchvision
import cv2

torch.manual_seed(0)


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

ImageItem = collections.namedtuple('ImageItem', ('image_name', 'tag'))
normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])

# normalize = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN,
#                                  std=IMAGENET_DEFAULT_STD)

transform = transforms.Compose([
    transforms.Resize(448, interpolation=3),
    transforms.CenterCrop(448),
    transforms.ToTensor(),
    normalize,
])

class RobustnessDataset(ImageFolder):
    def __init__(self, imagenet_path, imagenet_classes_path='imagenet_classes.json', isV2=True, isSI=False):
        self._isV2 = isV2
        self._isSI = isSI
        self._imagenet_path = imagenet_path
        with open(imagenet_classes_path, 'r') as f:
            self._imagenet_classes = json.load(f)
        self._tag_list = [tag for tag in os.listdir(self._imagenet_path)]
        self._all_images = []
        for tag in self._tag_list:
            base_dir = os.path.join(self._imagenet_path, tag)
            for i, file in enumerate(os.listdir(base_dir)):
                self._all_images.append(ImageItem(file, tag))


    def __getitem__(self, item):
        image_item = self._all_images[item]
        image_path = os.path.join(self._imagenet_path, image_item.tag, image_item.image_name)
        image = Image.open(image_path)
        image = image.convert('RGB')
        image = transform(image)

        if self._isV2:
            class_name = int(image_item.tag)
        elif self._isSI:
            class_name = int(label_str_to_imagenet_classes[image_item.tag])
        else:
            class_name = int(self._imagenet_classes[image_item.tag])

        return image, class_name

    def __len__(self):
        return len(self._all_images)