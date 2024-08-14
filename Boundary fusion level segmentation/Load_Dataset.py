# -*- coding: utf-8 -*-
# @Time    : 2021/6/19 11:30 上午
# @Author  : Haonan Wang
# @File    : Load_Dataset.py
# @Software: PyCharm
import numpy as np
import torch
import random
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from typing import Callable
import os
import cv2
from scipy import ndimage

def random_rot_flip(image, label,segresult,boundresult):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    segresult = np.rot90(segresult, k)
    boundresult = np.rot90(boundresult, k)

    axis = np.random.randint(0, 2)

    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    segresult = np.flip(segresult, axis=axis).copy()
    boundresult = np.flip(boundresult, axis=axis).copy()
    return image, label,segresult,boundresult

def random_rotate(image, label,segresult,boundresult):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    segresult = ndimage.rotate(segresult, angle, order=0, reshape=False)
    boundresult = ndimage.rotate(boundresult, angle, order=0, reshape=False)
    return image, label,segresult,boundresult

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label ,segresult,boundresult= sample['image'], sample['label'],sample['segresult'],sample['boundresult']
        image, label ,segresult,boundresult= F.to_pil_image(image), F.to_pil_image(label), F.to_pil_image(segresult), F.to_pil_image(boundresult)
        x, y = image.size
        if random.random() > 0.5:
            image, label,segresult,boundresult = random_rot_flip(image, label,segresult,boundresult)
        elif random.random() < 0.5:
            image, label,segresult,boundresult = random_rotate(image, label,segresult,boundresult)

        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            segresult = zoom(segresult, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            boundresult = zoom(boundresult, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = F.to_tensor(image)
        label = to_long_tensor(label)
        segresult = to_long_tensor(segresult)
        boundresult = to_long_tensor(boundresult)
        sample = {'image': image, 'label': label,'segresult':segresult,'boundresult':boundresult}
        return sample

class ValGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label ,segresult,boundresult= sample['image'], sample['label'],sample['segresult'],sample['boundresult']
        image, label ,segresult,boundresult= F.to_pil_image(image), F.to_pil_image(label), F.to_pil_image(segresult), F.to_pil_image(boundresult)
        x, y = image.size
        if random.random() > 0.5:
            image, label,segresult,boundresult = random_rot_flip(image, label,segresult,boundresult)
        elif random.random() < 0.5:
            image, label,segresult,boundresult = random_rotate(image, label,segresult,boundresult)

        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            segresult = zoom(segresult, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            boundresult = zoom(boundresult, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = F.to_tensor(image)
        label = to_long_tensor(label)
        segresult = to_long_tensor(segresult)
        boundresult = to_long_tensor(boundresult)
        sample = {'image': image, 'label': label,'segresult':segresult,'boundresult':boundresult}
        return sample

def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()

def correct_dims(*images):
    corr_images = []
    # print(images)
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)

    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images

class ImageToImage2D(Dataset):
    """
    Reads the images and applies the augmentation transform on them.
    Usage:
        1. If used without the unet.model.Model wrapper, an instance of this object should be passed to
           torch.utils.data.DataLoader. Iterating through this returns the tuple of image, mask and image
           filename.
        2. With unet.model.Model wrapper, an instance of this object should be passed as train or validation
           datasets.

    Args:
        dataset_path: path to the dataset. Structure of the dataset should be:
            dataset_path
              |-- images
                  |-- img001.png
                  |-- img002.png
                  |-- ...
              |-- masks
                  |-- img001.png
                  |-- img002.png
                  |-- ...

        joint_transform: augmentation transform, an instance of JointTransform2D. If bool(joint_transform)
            evaluates to False, torchvision.transforms.ToTensor will be used on both image and mask.
        one_hot_mask: bool, if True, returns the mask in one-hot encoded form.
    """

    def __init__(self, dataset_path: str, joint_transform: Callable = None, one_hot_mask: int = False, image_size: int =512) -> None:
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.input_path = os.path.join(dataset_path, 'img')
        self.segresult_path = os.path.join(dataset_path, 'segresult')
        self.boundresult_path = os.path.join(dataset_path, 'boundresult')
        self.output_path = os.path.join(dataset_path, 'labelcol')
        self.images_list = os.listdir(self.input_path)
        self.one_hot_mask = one_hot_mask

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(os.listdir(self.input_path))

    def __getitem__(self, idx):

        image_filename = self.images_list[idx]

        image = cv2.imread(os.path.join(self.input_path, image_filename))
        image = cv2.resize(image,(self.image_size,self.image_size))

        mask = cv2.imread(os.path.join(self.output_path, image_filename[: -3] + "png"),0)
        mask = cv2.resize(mask,(self.image_size,self.image_size))
        mask[mask<=0] = 0
        mask[mask>0] = 1

        segresult= cv2.imread(os.path.join(self.segresult_path, image_filename[: -3] + "png"),0)
        segresult = cv2.resize(segresult,(self.image_size,self.image_size))
        segresult[segresult<=0] = 0
        segresult[segresult>0] = 1

        boundresult= cv2.imread(os.path.join(self.boundresult_path, image_filename[: -3] + "png"),0)
        boundresult = cv2.resize(boundresult,(self.image_size,self.image_size))
        boundresult[boundresult<=0] = 0
        boundresult[boundresult>0] = 1


        image, mask,segresult,boundresult = correct_dims(image, mask,segresult,boundresult)
        # image, mask = F.to_pil_image(image), F.to_pil_image(mask)
        # print("11",image.shape)
        # print("22",mask.shape)
        imgfiles=[]
        imgfiles.append(os.path.join(self.input_path, image_filename))
        sample = {'image': image, 'label': mask,'segresult':segresult,'boundresult': boundresult}

        if self.joint_transform:
            sample = self.joint_transform(sample)
        # sample = {'image': image, 'label': mask}
        # print("2222",np.max(mask), np.min(mask))

        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)
        # mask = np.swapaxes(mask,2,0)
        # print(image.shape)
        # print("mask",mask)
        # mask = np.transpose(mask,(2,0,1))
        # image = np.transpose(image,(2,0,1))
        # print(image.shape)
        # print(mask.shape)
        # print(sample['image'].shape)
        return sample, image_filename

