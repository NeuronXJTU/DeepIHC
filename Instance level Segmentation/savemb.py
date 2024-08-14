import colorsys
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from nets.yolact import Yolact
from utils.anchors import get_anchors
from utils.utils import (cvtColor, get_classes, preprocess_input, resize_image,
                         show_config)
from utils.utils_bbox import BBoxUtility


# --------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和classes_path都需要修改！
# --------------------------------------------#
class YOLACT(object):
    _defaults = {
        # --------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        # --------------------------------------------------------------------------#
        "model_path": 'logs/best_epoch_weights.pth',
        "classes_path": 'model_data/shape_classes.txt',
        # ---------------------------------------------------------------------#
        #   输入图片的大小
        # ---------------------------------------------------------------------#
        "input_shape": [512, 512],
        # ---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        # ---------------------------------------------------------------------#
        "confidence": 0.5,
        # ---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        # ---------------------------------------------------------------------#
        "nms_iou": 0.3,
        # ---------------------------------------------------------------------#
        #   先验框的大小
        # ---------------------------------------------------------------------#
        "anchors_size": [24, 48, 96, 192, 384],
        # ---------------------------------------------------------------------#
        #   传统非极大抑制
        # ---------------------------------------------------------------------#
        "traditional_nms": False,
        # -------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        # -------------------------------#
        "cuda": True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化Yolact
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        # ---------------------------------------------------#
        #   计算总的类的数量
        # ---------------------------------------------------#
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.num_classes += 1
        self.anchors = torch.from_numpy(get_anchors(self.input_shape, self.anchors_size)).type(torch.FloatTensor)
        if self.cuda:
            self.anchors = self.anchors.cuda()
        if self.num_classes <= 81:
            self.colors = np.array([[0, 0, 0], [244, 67, 54], [233, 30, 99], [156, 39, 176], [103, 58, 183],
                                    [100, 30, 60], [63, 81, 181], [33, 150, 243], [3, 169, 244], [0, 188, 212],
                                    [20, 55, 200], [0, 150, 136], [76, 175, 80], [139, 195, 74], [205, 220, 57],
                                    [70, 25, 100], [255, 235, 59], [255, 193, 7], [255, 152, 0], [255, 87, 34],
                                    [90, 155, 50], [121, 85, 72], [158, 158, 158], [96, 125, 139], [15, 67, 34],
                                    [98, 55, 20], [21, 82, 172], [58, 128, 255], [196, 125, 39], [75, 27, 134],
                                    [90, 125, 120], [121, 82, 7], [158, 58, 8], [96, 25, 9], [115, 7, 234],
                                    [8, 155, 220], [221, 25, 72], [188, 58, 158], [56, 175, 19], [215, 67, 64],
                                    [198, 75, 20], [62, 185, 22], [108, 70, 58], [160, 225, 39], [95, 60, 144],
                                    [78, 155, 120], [101, 25, 142], [48, 198, 28], [96, 225, 200], [150, 167, 134],
                                    [18, 185, 90], [21, 145, 172], [98, 68, 78], [196, 105, 19], [215, 67, 84],
                                    [130, 115, 170], [255, 0, 255], [255, 255, 0], [196, 185, 10], [95, 167, 234],
                                    [18, 25, 190], [0, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255],
                                    [155, 0, 0], [0, 155, 0], [0, 0, 155], [46, 22, 130], [255, 0, 155],
                                    [155, 0, 255], [255, 155, 0], [155, 255, 0], [0, 155, 255], [0, 255, 155],
                                    [18, 5, 40], [120, 120, 255], [255, 58, 30], [60, 45, 60], [75, 27, 244], [128, 25, 70]], dtype='uint8')
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.bbox_util = BBoxUtility()
        self.generate()

        show_config(**self._defaults)

    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    def generate(self, onnx=False):
        self.net = Yolact(self.num_classes, train_mode=False)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))

        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()


    def save_image(self, img_name,imgpath,vis_mask_path,vis_boundary_path):
        image_path=os.path.join(imgpath,img_name)
        image = Image.open(image_path)
        image_shape = np.array(np.shape(image)[0:2])

        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        image_origin = np.array(image, np.uint8)
        # ---------------------------------------------------------#
        #   直接resize到指定大小
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        # ---------------------------------------------------------#
        #   添加上batch_size维度，图片预处理，归一化。
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)
        mask_8_path = os.path.join(vis_mask_path, img_name)
        boundary_8_path = os.path.join(vis_boundary_path, img_name)
        with torch.no_grad():
            image_data = torch.from_numpy(image_data).type(torch.FloatTensor)
            if self.cuda:
                image_data = image_data.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            outputs = self.net(image_data)
            # print("==============outputs",outputs)
            # ---------------------------------------------------------#
            #   解码并进行非极大抑制
            # ---------------------------------------------------------#
            results = self.bbox_util.decode_nms(outputs, self.anchors, self.confidence, self.nms_iou, image_shape,
                                                self.traditional_nms)
            # print("===============results",results)
            if results[0] is None:
                temp_out = np.zeros((image_shape[0], image_shape[1]))
                cv2.imwrite(mask_8_path, temp_out)
                cv2.imwrite(boundary_8_path, temp_out)
                return

            box_thre, class_thre, class_ids, masks_arg, masks_sigmoid, boundarys_arg= [
                x.cpu().numpy() for x in results]

        # ----------------------------------------------------------------------#
        #   masks_class [image_shape[0], image_shape[1]]
        #   根据每个像素点所属的实例和是否满足门限需求，判断每个像素点的种类
        # ----------------------------------------------------------------------#
        masks_class = masks_sigmoid * (class_ids[None, None, :] + 1)
        masks_class = np.reshape(masks_class, [-1, np.shape(masks_sigmoid)[-1]])
        masks_class = np.reshape(masks_class[np.arange(np.shape(masks_class)[0]), np.reshape(masks_arg, [-1])],
                                 [image_shape[0], image_shape[1]])



        color_masks = self.colors[masks_class].astype('uint8')
        cv2.imwrite("color_masks.png", color_masks)
        mask_8 = cv2.imread("color_masks.png", cv2.IMREAD_GRAYSCALE)
        mask_8_path = os.path.join(vis_mask_path, img_name)
        boundary_8_path = os.path.join(vis_boundary_path, img_name)
        mask_8[mask_8 > 0] = 255
        boundarys_arg[boundarys_arg > 0] = 255
        #print(boundarys_arg)
        cv2.imwrite(mask_8_path,mask_8)
        cv2.imwrite(boundary_8_path,boundarys_arg)
        return
yolact=YOLACT()
test_list=['train','test','val']
for test_name in test_list:
    if test_name=='train':
        imgpath="savemask/"+'Train_Folder/img/'
        vis_mask_path = "savemask/"+'Train_Folder/segresult/'
        vis_boundary_path = "savemask/"+'Train_Folder/boundresult/'
    elif test_name == 'test':
        imgpath = "savemask/" + 'Test_Folder/img/'
        vis_mask_path = "savemask/" + 'Test_Folder/segresult/'
        vis_boundary_path = "savemask/" + 'Test_Folder/boundresult/'
    elif test_name == 'val':
        imgpath = "savemask/" + 'Val_Folder/img/'
        vis_mask_path = "savemask/" + 'Val_Folder/segresult/'
        vis_boundary_path = "savemask/" + 'Val_Folder/boundresult/'
    if not os.path.exists(vis_mask_path):
        os.makedirs(vis_mask_path)
    if not os.path.exists(vis_boundary_path):
        os.makedirs(vis_boundary_path)
    imgs = os.listdir(imgpath)
    for img_name in imgs:
        yolact.save_image(img_name,imgpath,vis_mask_path,vis_boundary_path)


