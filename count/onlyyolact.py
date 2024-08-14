import shutil
import cv2
import numpy as np
from PIL import Image
import torch
from yolact import Yolact
from utils import get_anchors,decode_nms
import os
from utils import split_img,image_compose
from PIL import ImageDraw, ImageFont
from patch_dense_net import UNet
from Load_Dataset import ValGenerator, ImageToImage2D
from torch.utils.data import DataLoader
from utils import count_center
input_shape=[512,512]
colors = np.array([[0, 0, 0], [244, 67, 54] ],dtype='uint8')
num_classes=2
model_path_2="best_epoch/ki67-2-yolact.pth"
model_path_fusion_2 = "best_epoch/ki67-2-fusion.pth.tar"
model_path_1 = "best_epoch/ki67-1-yolact.pth"
model_path_fusion_1 = "best_epoch/ki67-1-fusion.pth.tar"
confidence=0.5
nms_iou=0.3
traditional_nms=False
anchors_size=[24, 48, 96, 192, 384]
input_modalities = 3
n_pred_labels=1
anchors = torch.from_numpy(get_anchors(input_shape, anchors_size)).type(torch.FloatTensor)



def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


def resize_image(image, size):
    w, h    = size
    new_image = image.resize((w, h), Image.BICUBIC)
    return new_image

def preprocess_input(image):
    mean    = (123.68, 116.78, 103.94)
    std     = (58.40, 57.12, 57.38)
    image   = (image - mean)/std
    return image
def save_image(net,img_name,imgpath,vis_mask_path,vis_boundary_path):
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
    image_data = resize_image(image, (input_shape[1], input_shape[0]))
    # ---------------------------------------------------------#
    #   添加上batch_size维度，图片预处理，归一化。
    # ---------------------------------------------------------#
    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)
    mask_8_path = os.path.join(vis_mask_path, img_name)
    boundary_8_path = os.path.join(vis_boundary_path, img_name)
    with torch.no_grad():
        image_data = torch.from_numpy(image_data).type(torch.FloatTensor)
        # ---------------------------------------------------------#
        #   将图像输入网络当中进行预测！
        # ---------------------------------------------------------#
        outputs = net(image_data)
        # print("==============outputs",outputs)
        # ---------------------------------------------------------#
        #   解码并进行非极大抑制
        # ---------------------------------------------------------#
        results = decode_nms(outputs, anchors, confidence, nms_iou, image_shape,
                                            traditional_nms)
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
    color_masks = colors[masks_class].astype('uint8')
    cv2.imwrite("color_masks.png", color_masks)
    mask_8 = cv2.imread("color_masks.png", cv2.IMREAD_GRAYSCALE)
    mask_8_path = os.path.join(vis_mask_path, img_name)
    boundary_8_path = os.path.join(vis_boundary_path, img_name)
    mask_8[mask_8 > 0] = 255
    boundarys_arg[boundarys_arg > 0] = 255
    #print(boundarys_arg)
    cv2.imwrite(mask_8_path,mask_8)
    cv2.imwrite(boundary_8_path,boundarys_arg)
    os.remove("color_masks.png")
    return


def get_centers(image,net_yolact,net_fusion,img_width, img_heigh,IMAGE_ROW, IMAGE_COLUMN,end_name,type):

    split_img(image, input_shape[0], input_shape[1])
    imgpath = "test/img/"
    vis_mask_path = "test/segresult/"
    vis_boundary_path = "test/boundresult/"
    if not os.path.exists(vis_mask_path):
        os.makedirs(vis_mask_path, mode=0o777)
    if not os.path.exists(vis_boundary_path):
        os.makedirs(vis_boundary_path, mode=0o777)
    imgs = os.listdir(imgpath)
    for img_name in imgs:
        save_image(net_yolact, img_name, imgpath, vis_mask_path, vis_boundary_path)
    if type=="1":
        image_compose(input_shape, IMAGE_ROW, IMAGE_COLUMN, 0, vis_mask_path, "yolact/1/"+end_name, [img_width, img_heigh])  # 调用函数
    elif type=="2":
        image_compose(input_shape, IMAGE_ROW, IMAGE_COLUMN, 0, vis_mask_path, "yolact/2/" + end_name,
                      [img_width, img_heigh])  # 调用函数


def main(img):
    end_name=img.split('/')[-1]
    net_ki67_1 = Yolact(num_classes, train_mode=False)
    net_ki67_1.load_state_dict(torch.load(model_path_1, map_location=torch.device('cpu')))
    net_ki67_1 = net_ki67_1.eval()

    net_ki67_2 = Yolact(num_classes, train_mode=False)
    net_ki67_2.load_state_dict(torch.load(model_path_2, map_location=torch.device('cpu')))
    net_ki67_2 = net_ki67_2.eval()

    checkpoint_fusion_ki671 = torch.load(model_path_fusion_1, map_location=torch.device('cpu'))
    fusion_ki671 = UNet(input_modalities, n_pred_labels * 2, n_pred_labels)
    fusion_ki671.load_state_dict(checkpoint_fusion_ki671['state_dict'])
    fusion_ki671 = fusion_ki671.eval()

    checkpoint_fusion_ki672 = torch.load(model_path_fusion_2, map_location=torch.device('cpu'))
    fusion_ki672 = UNet(input_modalities, n_pred_labels * 2, n_pred_labels)
    fusion_ki672.load_state_dict(checkpoint_fusion_ki672['state_dict'])
    fusion_ki672=fusion_ki672.eval()

    image = Image.open(img)
    img_width=image.size[0]
    img_heigh=image.size[1]
    font = ImageFont.truetype(font='simhei.ttf', size=np.floor(3e-2 * image.size[1] - 0.5).astype('int32'))
    IMAGE_ROW = int(img_heigh/input_shape[0])+1  # 图片间隔，也就是合并成一张图后，一共有几行
    IMAGE_COLUMN = int(img_width/input_shape[0])+1
    #
    get_centers(image, net_ki67_1, fusion_ki671, img_width, img_heigh, IMAGE_ROW, IMAGE_COLUMN,end_name,"1")
    get_centers(image,net_ki67_2,fusion_ki672,img_width, img_heigh,IMAGE_ROW, IMAGE_COLUMN,end_name,"2")


if __name__ == '__main__':
    files_img = os.listdir(r'D:/ki67/BCData/images/test/')
    for i in files_img:
        img_name = os.path.join('D:/ki67/BCData/images/test/' + i)
        main(img_name)

