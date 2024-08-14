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


def get_centers(image,net_yolact,net_fusion,img_width, img_heigh,IMAGE_ROW, IMAGE_COLUMN,end_name,type,out_dir):
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


    tf_test = ValGenerator(output_size=input_shape)
    test_dataset = ImageToImage2D("test/", tf_test, image_size=input_shape[0])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    for i, (sampled_batch, names) in enumerate(test_loader, 1):
        test_data, test_segresult, test_boundresult = sampled_batch['image'], sampled_batch['segresult'], sampled_batch[
            'boundresult']
        # test_data, test_label = test_data.cuda(), test_label.cuda()
        arr = test_data.numpy()
        arr = arr.astype(np.float32())
        input_img = torch.from_numpy(arr)
        test_segresult = test_segresult.unsqueeze(1)
        test_boundresult = test_boundresult.unsqueeze(1)

        output = net_fusion(input_img, torch.cat((test_segresult, test_boundresult), dim=1))
        pred_class = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output))
        predict_save = pred_class[0].cpu().data.numpy()
        predict_save = np.reshape(predict_save, (input_shape[0], input_shape[1]))
        for p in range(0, predict_save.shape[0]):
            if p < 10 or p > (predict_save.shape[0] - 10):
                for q in range(0, predict_save.shape[1]):
                    if q < 10 or q > (predict_save.shape[1] - 10):
                        predict_save[p, q] = 0
        if not os.path.exists("test/mask"):
            os.mkdir("test/mask/", mode=0o777)
        cv2.imwrite("test/mask/" + names[0], predict_save * 255)
    ki672_mask = 'test/mask/'
    image_compose(input_shape, IMAGE_ROW, IMAGE_COLUMN, 0, ki672_mask, "ki672.png", [img_width, img_heigh])  # 调用函数
    ki672 = cv2.imread('ki672.png', 0)
    ki672 = ki672[0:img_heigh, 0:img_width]
    if not os.path.exists(out_dir+"/1/"):
        os.mkdir(out_dir+"/1/")  # 如果不存在该文件夹，则创建，用于储存后续提取出来的文件
    if not os.path.exists(out_dir + "/2/"):
        os.mkdir(out_dir + "/2/")  # 如果不存在该文件夹，则创建，用于储存后续提取出来的文件
    if type=="1":
        cv2.imwrite(out_dir+"/1/"+end_name, ki672)
        ki672_count_center = count_center(out_dir+"/1/"+end_name)
    if type=="2":
        cv2.imwrite(out_dir+"/2/" + end_name, ki672)
        ki672_count_center = count_center(out_dir+"/2/" + end_name)
    # os.remove("ki672.png")
    return ki672_count_center

def main(img,out_dir):
    print(out_dir)
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
    if img_heigh % input_shape[0] == 0:
        IMAGE_ROW = int(img_heigh / input_shape[0])  # 图片间隔，也就是合并成一张图后，一共有几行
    else:
        IMAGE_ROW = int(img_heigh / input_shape[0]) + 1
    if img_width % input_shape[1] == 0:
        IMAGE_COLUMN = int(img_width / input_shape[0])
    else:
        IMAGE_COLUMN = int(img_width / input_shape[0]) + 1
    #
    ki671_count_center = get_centers(image, net_ki67_1, fusion_ki671, img_width, img_heigh, IMAGE_ROW, IMAGE_COLUMN,end_name,"1",out_dir)
    ki672_count_center=get_centers(image,net_ki67_2,fusion_ki672,img_width, img_heigh,IMAGE_ROW, IMAGE_COLUMN,end_name,"2",out_dir)

    shutil.rmtree("test")
    imageki672 = cvtColor(Image.open(img))
    drawki672 = ImageDraw.Draw(imageki672)
    big_title = "蓝色:%d 棕色:%d" % (len(ki671_count_center), len(ki672_count_center))
    drawki672.text((50, 20), big_title, fill=(0, 0, 0), font=font)
    del drawki672
    for i, c in list(enumerate(ki672_count_center)):
        draw = ImageDraw.Draw(imageki672)
        shape = [(ki672_count_center[i][0] - 5, ki672_count_center[i][1] - 5), (ki672_count_center[i][0] + 5, ki672_count_center[i][1] + 5)]
        draw.ellipse(shape, fill=(0,255,0))
        del draw
    for i, c in list(enumerate(ki671_count_center)):
        draw = ImageDraw.Draw(imageki672)
        shape = [(ki671_count_center[i][0] - 5, ki671_count_center[i][1] - 5), (ki671_count_center[i][0] + 5, ki671_count_center[i][1] + 5)]
        draw.ellipse(shape, fill=(255,0,0))
        del draw
    # with open(r'D:/ki67/ours_ki67/wsi_choose/count_all.txt', 'a') as f:
    #     f.write(end_name +'  '+str(len(ki671_count_center))+'  '+str(len(ki672_count_center))+'  蓝色：'+str(ki671_count_center)+'  棕色：'+str(ki672_count_center)+'\n')
    imageki672.save(out_dir+'/'+end_name)
    return len(ki671_count_center),len(ki672_count_center)

if __name__ == '__main__':
    # path="D:/ki67/xkz/20240124/"
    # path_1=os.listdir(path)
    # for name1 in path_1:
    #     name_1 = os.path.join(r'D:/ki67/xkz/20240124/'+name1)
    #     out_dir='D:/ki67/xkz/20240124_COUNT/'+name1
    #     if not os.path.exists(out_dir):
    #         os.mkdir(out_dir)  # 如果不存在该文件夹，则创建，用于储存后续提取出来的文件
    path_2 = os.listdir("D:/ki67/xkz/20240124/")
    for name2 in path_2:
        name_2 = os.path.join(r'D:/ki67/xkz/20240124/' +name2)
        out_dir = 'D:/ki67/xkz/20240124_COUNT1/'+ name2+'/'
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)  # 如果不存在该文件夹，则创建，用于储存后续提取出来的文件
        path_3 = os.listdir(name_2)
        ki671 = 0
        ki672 = 0
        for img in path_3:
            img_name=name_2+'/'+img
            ki671_count_center,ki672_count_center=main(img_name,out_dir)
        #     ki671=ki671+ki671_count_center
        #     ki672=ki672+ki672_count_center
        # with open(r'D:/ki67/choose/wsi_ki67_count/count_all.txt', 'a') as f:
        #     f.write(name + '  ' + str(ki671) + '  ' + str(ki672)+'\n')


4