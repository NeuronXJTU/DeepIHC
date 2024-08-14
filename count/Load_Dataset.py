
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from typing import Callable
import os
import cv2


class ValGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image,segresult,boundresult= sample['image'],sample['segresult'],sample['boundresult']
        image,segresult,boundresult= F.to_pil_image(image), F.to_pil_image(segresult), F.to_pil_image(boundresult)
        image = F.to_tensor(image)
        segresult = to_long_tensor(segresult)
        boundresult = to_long_tensor(boundresult)
        sample = {'image': image,'segresult':segresult,'boundresult':boundresult}
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
    def __init__(self, dataset_path: str, joint_transform: Callable = None, one_hot_mask: int = False, image_size: int =512) -> None:
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.input_path = os.path.join(dataset_path, 'img')
        self.segresult_path = os.path.join(dataset_path, 'segresult')
        self.boundresult_path = os.path.join(dataset_path, 'boundresult')
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

        segresult= cv2.imread(os.path.join(self.segresult_path, image_filename[: -3] + "png"),0)
        segresult = cv2.resize(segresult,(self.image_size,self.image_size))
        segresult[segresult<=0] = 0
        segresult[segresult>0] = 1

        boundresult= cv2.imread(os.path.join(self.boundresult_path, image_filename[: -3] + "png"),0)
        boundresult = cv2.resize(boundresult,(self.image_size,self.image_size))
        boundresult[boundresult<=0] = 0
        boundresult[boundresult>0] = 1


        image,segresult,boundresult = correct_dims(image,segresult,boundresult)
        imgfiles=[]
        imgfiles.append(os.path.join(self.input_path, image_filename))
        sample = {'image': image, 'segresult':segresult,'boundresult': boundresult}

        if self.joint_transform:
            sample = self.joint_transform(sample)
        return sample, image_filename

