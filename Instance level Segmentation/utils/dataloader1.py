import os.path as osp

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

from utils.utils import cvtColor, preprocess_input


class COCODetection(data.Dataset):
    def __init__(self, image_path, coco,local_rank, COCO_LABEL_MAP={}, augmentation=None):
        self.image_path     = image_path

        self.coco           = coco
        self.ids            = list(self.coco.imgToAnns.keys())
        
        self.augmentation   = augmentation
        
        self.label_map      = COCO_LABEL_MAP
        self.length         = len(self.ids)

        self.local_rank         = local_rank

    def __getitem__(self, index):
        return self.pull_item(index)

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        local_rank=self.local_rank
        #------------------------------#
        #   载入coco序号
        #   根据coco序号载入目标信息
        #------------------------------#
        image_id    = self.ids[index]
        target      = self.coco.loadAnns(self.coco.getAnnIds(imgIds = image_id))

        #------------------------------#
        #   根据目标信息判断是否为
        #   iscrowd
        #------------------------------#
        target      = [x for x in target if not ('iscrowd' in x and x['iscrowd'])]
        crowd       = [x for x in target if ('iscrowd' in x and x['iscrowd'])]
        num_crowds  = len(crowd)
        #------------------------------#
        #   将不是iscrowd的目标
        #       是iscrowd的目标进行堆叠
        #------------------------------#
        target      += crowd

        image_path  = osp.join(self.image_path, self.coco.loadImgs(image_id)[0]['file_name'])
        #print(self.coco.loadImgs(image_id)[0]['file_name'])
        image       = Image.open(image_path)
        image       = cvtColor(image)
        image       = np.array(image, np.float32)
        height, width, _ = image.shape
        if len(target) > 0:
            masks = np.array([self.coco.annToMask(obj).reshape(-1) for obj in target], np.float32)
            masks = masks.reshape((-1, height, width))
            boundarys = np.zeros((len(target),height, width))
            boxes_classes = []
            j=0
            for obj in target:
                seg=obj['segmentation'][0]
                for i in range(0,len(seg),2):
                    temp_x=seg[i]
                    temp_y=seg[i+1]
                    boundarys[j,int(temp_y),int(temp_x)]=1
                bbox        = obj['bbox']
                final_box   = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3], self.label_map[obj['category_id']] - 1]
                boxes_classes.append(final_box)
                j=j+1
            boxes_classes = np.array(boxes_classes, np.float32)
            boxes_classes[:, [0, 2]] /= width
            boxes_classes[:, [1, 3]] /= height
        if self.augmentation is not None:
            if len(boxes_classes) > 0:
                #print(masks)
                image, masks,boundarys, boxes, labels = self.augmentation(image, masks,boundarys, boxes_classes[:, :4], {'num_crowds': num_crowds, 'labels': boxes_classes[:, 4]})
                num_crowds  = labels['num_crowds']
                labels      = labels['labels']
                boxes       = np.concatenate([boxes, np.expand_dims(labels, axis=1)], -1)
        image = preprocess_input(image)

        images = torch.from_numpy(np.array(np.transpose(image, [2, 0, 1]), np.float32)).cuda(local_rank)
        boxes = [ann.cuda(local_rank) for ann in boxes]
        masks = [mask.cuda(local_rank) for mask in masks]
        boundarys = [boundary.cuda(local_rank) for boundary in boundarys]
        return images, boxes, masks,boundarys, num_crowds

def dataset_collate(batch):
    images      = []
    targets     = []
    masks       = []
    boundarys       = []
    num_crowds  = []

    for sample in batch:
        images.append(sample[0])
        targets.append(torch.from_numpy(sample[1]))
        masks.append(torch.from_numpy(sample[2]))
        boundarys.append(torch.from_numpy(sample[3]))
        num_crowds.append(sample[4])

    return torch.from_numpy(np.array(images, np.float32)), targets, masks,boundarys, num_crowds
