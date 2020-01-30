#!/usr/bin/env python
# encoding: utf-8

import os
import cv2
import torch
import numpy as np
from random import shuffle
import torch.utils.data as data
import torchvision.transforms as transforms


def img_loader(path):
    try:
        with open(path, 'rb') as f:
            img = cv2.imread(path, 1)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = cv2.resize(img, (299, 299))
            #img = cv2.resize(img, (112, 112))
            return img
    except IOError:
        print('Cannot load image ' + path)


class TrainDataFlow(data.Dataset):
    def __init__(self, root, file_list, transform=None, loader=img_loader):

        self.root = root
        self.transform = transform
        self.loader = loader

        image_list = []
        label_list = []
        with open(file_list) as f:
            img_label_list = f.read().splitlines()
        shuffle(img_label_list)
        shuffle(img_label_list)
        for info in img_label_list:
            image_path, label_name = info.split(' ')
            image_list.append(image_path)
            label_list.append(int(label_name))

        self.image_list = image_list
        self.label_list = label_list
        self.class_nums = len(np.unique(self.label_list))
        print("dataset size: ", len(self.image_list), '/', self.class_nums)

    def __getitem__(self, index):
        img_path = self.image_list[index]
        label = self.label_list[index]
        img = self.loader(os.path.join(self.root, img_path))
        return img.tobytes(), label

    def __len__(self):
        return len(self.image_list)

