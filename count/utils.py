from itertools import product
from math import sqrt
import torch
import torch.nn.functional as F
from torchvision.ops import nms
from PIL import Image
import shutil
from skimage import img_as_ubyte
from skimage.morphology import reconstruction, dilation, erosion, disk, diamond, square
import numpy as np
from skimage.measure import label
from skimage.segmentation import watershed
from scipy.ndimage.morphology import distance_transform_cdt
import os
import cv2
from skimage.io import imread, imsave
def make_anchors(conv_h, conv_w, scale, input_shape=[550, 550], aspect_ratios=[1, 1 / 2, 2]):
    prior_data = []
    for j, i in product(range(conv_h), range(conv_w)):
        x = (i + 0.5) / conv_w
        y = (j + 0.5) / conv_h

        for ar in aspect_ratios:
            ar = sqrt(ar)
            w = scale * ar / input_shape[1]
            h = scale / ar / input_shape[0]

            prior_data += [x, y, w, h]

    return prior_data


# ---------------------------------------------------#
#   用于计算共享特征层的大小
# ---------------------------------------------------#
def get_img_output_length(height, width):
    filter_sizes = [7, 3, 3, 3, 3, 3, 3]
    padding = [3, 1, 1, 1, 1, 1, 1]
    stride = [2, 2, 2, 2, 2, 2, 2]
    feature_heights = []
    feature_widths = []

    for i in range(len(filter_sizes)):
        height = (height + 2 * padding[i] - filter_sizes[i]) // stride[i] + 1
        width = (width + 2 * padding[i] - filter_sizes[i]) // stride[i] + 1
        feature_heights.append(height)
        feature_widths.append(width)
    return np.array(feature_heights)[-5:], np.array(feature_widths)[-5:]


def get_anchors(input_shape=[550, 550], anchors_size=[24, 48, 96, 192, 384]):
    feature_heights, feature_widths = get_img_output_length(input_shape[0], input_shape[1])

    all_anchors = []
    for i in range(len(feature_heights)):
        anchors = make_anchors(feature_heights[i], feature_widths[i], anchors_size[i], input_shape)
        all_anchors += anchors

    all_anchors = np.reshape(all_anchors, [-1, 4])
    return all_anchors

def decode_boxes( pred_box, anchors, variances = [0.1, 0.2]):
    #---------------------------------------------------------#
    #   anchors[:, :2] 先验框中心
    #   anchors[:, 2:] 先验框宽高
    #   对先验框的中心和宽高进行调整，获得预测框
    #---------------------------------------------------------#
    boxes = torch.cat((anchors[:, :2] + pred_box[:, :2] * variances[0] * anchors[:, 2:],
                    anchors[:, 2:] * torch.exp(pred_box[:, 2:] * variances[1])), 1)

    #---------------------------------------------------------#
    #   获得左上角和右下角
    #---------------------------------------------------------#
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def jaccard( box_a, box_b, iscrowd: bool = False):
    use_batch = True
    if box_a.dim() == 2:
        use_batch = False
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]

    n = box_a.size(0)
    A = box_a.size(1)
    B = box_b.size(1)

    max_xy = torch.min(box_a[:, :, 2:].unsqueeze(2).expand(n, A, B, 2), box_b[:, :, 2:].unsqueeze(1).expand(n, A, B, 2))
    min_xy = torch.max(box_a[:, :, :2].unsqueeze(2).expand(n, A, B, 2), box_b[:, :, :2].unsqueeze(1).expand(n, A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    inter = inter[:, :, :, 0] * inter[:, :, :, 1]

    area_a = ((box_a[:, :, 2] - box_a[:, :, 0]) * (box_a[:, :, 3] - box_a[:, :, 1])).unsqueeze(2).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, :, 2] - box_b[:, :, 0]) * (box_b[:, :, 3] - box_b[:, :, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter

    out = inter / area_a if iscrowd else inter / union
    return out if use_batch else out.squeeze(0)
def fast_non_max_suppression( box_thre, class_thre, mask_thre, nms_iou=0.5, top_k=200, max_detections=100):
    # ---------------------------------------------------------#
    #   先进行tranpose，方便后面的处理
    #   [80, num_of_kept_boxes]
    # ---------------------------------------------------------#
    class_thre = class_thre.transpose(1, 0).contiguous()
    # ---------------------------------------------------------#
    #   [80, num_of_kept_boxes]
    #   每一行坐标为该种类所有的框的得分，
    #   对每一个种类单独进行排序
    # ---------------------------------------------------------#
    class_thre, idx = class_thre.sort(1, descending=True)

    idx = idx[:, :top_k].contiguous()
    class_thre = class_thre[:, :top_k]

    num_classes, num_dets = idx.size()
    # ---------------------------------------------------------#
    #   将num_classes作为第一维度，对每一个类进行非极大抑制
    #   [80, num_of_kept_boxes, 4]
    #   [80, num_of_kept_boxes, 32]
    # ---------------------------------------------------------#
    box_thre = box_thre[idx.view(-1), :].view(num_classes, num_dets, 4)
    mask_thre = mask_thre[idx.view(-1), :].view(num_classes, num_dets, -1)

    iou = jaccard(box_thre, box_thre)
    # ---------------------------------------------------------#
    #   [80, num_of_kept_boxes, num_of_kept_boxes]
    #   取矩阵的上三角部分
    # ---------------------------------------------------------#
    iou.triu_(diagonal=1)
    iou_max, _ = iou.max(dim=1)

    # ---------------------------------------------------------#
    #   获取和高得分重合程度比较低的预测结果
    # ---------------------------------------------------------#
    keep = (iou_max <= nms_iou)
    class_ids = torch.arange(num_classes, device=box_thre.device)[:, None].expand_as(keep)

    box_nms = box_thre[keep]
    class_nms = class_thre[keep]
    class_ids = class_ids[keep]
    mask_nms = mask_thre[keep]

    _, idx = class_nms.sort(0, descending=True)
    idx = idx[:max_detections]
    box_nms = box_nms[idx]
    class_nms = class_nms[idx]
    class_ids = class_ids[idx]
    mask_nms = mask_nms[idx]
    return box_nms, class_nms, class_ids, mask_nms

def traditional_non_max_suppression(box_thre, class_thre, mask_thre, pred_class_max, nms_iou, max_detections):
    num_classes = class_thre.size()[1]
    pred_class_arg = torch.argmax(class_thre, dim=-1)

    box_nms, class_nms, class_ids, mask_nms = [], [], [], []
    for c in range(num_classes):
        # --------------------------------#
        #   取出属于该类的所有框的置信度
        #   判断是否大于门限
        # --------------------------------#
        c_confs_m = pred_class_arg == c
        if len(c_confs_m) > 0:
            # -----------------------------------------#
            #   取出得分高于confidence的框
            # -----------------------------------------#
            boxes_to_process = box_thre[c_confs_m]
            confs_to_process = pred_class_max[c_confs_m]
            masks_to_process = mask_thre[c_confs_m]
            # -----------------------------------------#
            #   进行iou的非极大抑制
            # -----------------------------------------#
            idx = nms(boxes_to_process, confs_to_process, nms_iou)
            # -----------------------------------------#
            #   取出在非极大抑制中效果较好的内容
            # -----------------------------------------#
            good_boxes = boxes_to_process[idx]
            confs = confs_to_process[idx]
            labels = c * torch.ones((len(idx))).long()
            good_masks = masks_to_process[idx]
            box_nms.append(good_boxes)
            class_nms.append(confs)
            class_ids.append(labels)
            mask_nms.append(good_masks)
    box_nms, class_nms, class_ids, mask_nms = torch.cat(box_nms, dim=0), torch.cat(class_nms, dim=0), \
                                              torch.cat(class_ids, dim=0), torch.cat(mask_nms, dim=0)

    idx = torch.argsort(class_nms, 0, descending=True)[:max_detections]
    box_nms, class_nms, class_ids, mask_nms = box_nms[idx], class_nms[idx], class_ids[idx], mask_nms[idx]
    return box_nms, class_nms, class_ids, mask_nms

def yolact_correct_boxes( boxes, image_shape):
    image_size = np.array(image_shape)[::-1]
    image_size = torch.tensor([*image_size]).type(boxes.dtype).cuda() if boxes.is_cuda else torch.tensor(
        [*image_size]).type(boxes.dtype)

    scales = torch.cat([image_size, image_size], dim=-1)
    boxes = boxes * scales
    boxes[:, [0, 1]] = torch.min(boxes[:, [0, 1]], boxes[:, [2, 3]])
    boxes[:, [2, 3]] = torch.max(boxes[:, [0, 1]], boxes[:, [2, 3]])
    boxes[:, [0, 1]] = torch.max(boxes[:, [0, 1]], torch.zeros_like(boxes[:, [0, 1]]))
    boxes[:, [2, 3]] = torch.min(boxes[:, [2, 3]], torch.unsqueeze(image_size, 0).expand([boxes.size()[0], 2]))
    return boxes

def crop(masks, boxes):
    h, w, n = masks.size()
    x1, x2 = boxes[:, 0], boxes[:, 2]
    y1, y2 = boxes[:, 1], boxes[:, 3]

    rows = torch.arange(w, device=masks.device, dtype=x1.dtype).view(1, -1, 1).expand(h, w, n)
    cols = torch.arange(h, device=masks.device, dtype=x1.dtype).view(-1, 1, 1).expand(h, w, n)

    masks_left = rows >= x1.view(1, 1, -1)
    masks_right = rows < x2.view(1, 1, -1)
    masks_up = cols >= y1.view(1, 1, -1)
    masks_down = cols < y2.view(1, 1, -1)

    crop_mask = masks_left * masks_right * masks_up * masks_down
    return masks * crop_mask.float()

def decode_nms(outputs, anchors, confidence, nms_iou, image_shape, traditional_nms=False, max_detections=100):
    # ---------------------------------------------------------#
    #   pred_box    [18525, 4]  对应每个先验框的调整情况
    #   pred_class  [18525, 81] 对应每个先验框的种类
    #   pred_mask   [18525, 32] 对应每个先验框的语义分割情况
    #   pred_proto  [128, 128, 32]  需要和结合pred_mask使用
    # ---------------------------------------------------------#
    pred_box = outputs[0].squeeze()
    pred_class = outputs[1].squeeze()
    pred_masks = outputs[2].squeeze()
    pred_boundarys = outputs[3].squeeze()
    pred_proto = outputs[4].squeeze()

    # ---------------------------------------------------------#
    #   将先验框调整获得预测框，
    #   [18525, 4] boxes是左上角、右下角的形式。
    # ---------------------------------------------------------#
    boxes = decode_boxes(pred_box, anchors)

    # ---------------------------------------------------------#
    #   除去背景的部分，并获得最大的得分
    #   [18525, 80]
    #   [18525]
    # ---------------------------------------------------------#
    pred_class = pred_class[:, 1:]
    pred_class_max, _ = torch.max(pred_class, dim=1)
    keep = (pred_class_max > confidence)

    # ---------------------------------------------------------#
    #   保留满足得分的框，如果没有框保留，则返回None
    # ---------------------------------------------------------#
    box_thre = boxes[keep, :]
    class_thre = pred_class[keep, :]
    mask_thre = pred_masks[keep, :]
    if class_thre.size()[0] == 0:
        return None, None, None, None, None

    if not traditional_nms:
        box_thre, class_thre, class_ids, mask_thre = fast_non_max_suppression(box_thre, class_thre, mask_thre,
                                                                                   nms_iou)
        keep = class_thre > confidence
        box_thre = box_thre[keep]
        class_thre = class_thre[keep]
        class_ids = class_ids[keep]
        mask_thre = mask_thre[keep]
    else:
        box_thre, class_thre, class_ids, mask_thre = traditional_non_max_suppression(box_thre, class_thre,
                                                                                          mask_thre,
                                                                                          pred_class_max[keep],
                                                                                          nms_iou, max_detections)

    box_thre = yolact_correct_boxes(box_thre, image_shape)

    # ---------------------------------------------------------#
    #   pred_proto      [128, 128, 32]
    #   mask_thre       [num_of_kept_boxes, 32]
    #   masks_sigmoid   [128, 128, num_of_kept_boxes]
    # ---------------------------------------------------------#
    masks_sigmoid = torch.sigmoid(torch.matmul(pred_proto, mask_thre.t()))
    # ----------------------------------------------------------------------#
    #   masks_sigmoid   [image_shape[0], image_shape[1], num_of_kept_boxes]
    # ----------------------------------------------------------------------#
    masks_sigmoid = masks_sigmoid.permute(2, 0, 1).contiguous()
    masks_sigmoid = F.interpolate(masks_sigmoid.unsqueeze(0), (image_shape[0], image_shape[1]), mode='bilinear',
                                  align_corners=False).squeeze(0)
    masks_sigmoid = masks_sigmoid.permute(1, 2, 0).contiguous()
    masks_sigmoid = crop(masks_sigmoid, box_thre)

    # ----------------------------------------------------------------------#
    #   masks_arg   [image_shape[0], image_shape[1]]
    #   获得每个像素点所属的实例
    # ----------------------------------------------------------------------#
    masks_arg = torch.argmax(masks_sigmoid, dim=-1)
    # ----------------------------------------------------------------------#
    #   masks_arg   [image_shape[0], image_shape[1], num_of_kept_boxes]
    #   判断每个像素点是否满足门限需求
    # ----------------------------------------------------------------------#
    masks_sigmoid = masks_sigmoid > 0.5
    pred_boundarys[pred_boundarys > 0.8] = 1
    pred_boundarys[pred_boundarys <= 0.8] = 0
    return box_thre, class_thre, class_ids, masks_arg, masks_sigmoid, pred_boundarys

#==========================================图像裁剪合成=======================================================#

# 判断是否需要进行图像填充
def judge(img, wi, he):
    width, height = img.size
    # 默认新图像尺寸初始化为原图像
    new_width, new_height = img.size
    if width % wi != 0:
        new_width = (width//wi + 1) * wi
    if height % he != 0:
        new_height = (height//he + 1) * he
    # 新建一张新尺寸的全黑图像
    new_image = Image.new('RGB', (new_width, new_height), (0, 0, 0))
    # 将原图像粘贴在new_image上，默认为左上角坐标对应
    new_image.paste(img, box=None, mask=None)
    return new_image
# 按照指定尺寸进行图片裁剪
def crop_image(image, patch_w, patch_h):
    width, height = image.size
    # 补丁计数
    cnt = 0
    if not os.path.exists("test"):
        os.mkdir("test",mode=0o777)
    else:
        shutil.rmtree("test")
        os.mkdir("test",mode=0o777)
    if not os.path.exists("test/img"):
        os.mkdir("test/img",mode=0o777)
    else:
        shutil.rmtree("test/img")
        os.mkdir("test/img",mode=0o777)
    for w in range(0, width, patch_w):
        for h in range(0, height, patch_h):
            cnt += 1
            # 指定原图片的左、上、右、下
            img = image.crop((w, h, w+patch_w, h+patch_h))
            img.save("test/img/%d.png" % cnt)
    print("图片补丁裁剪结束，共有{}张补丁".format(cnt))

def split_img(img,wi,he):
    new_image = judge(img, wi, he)
    # 图片补丁裁剪
    crop_image(new_image, wi, he)

def image_compose(IMAGE_SIZE, IMAGE_ROW, IMAGE_COLUMN, padding, IMAGES_PATH, IMAGE_SAVE_PATH,row_shape):
    IMAGES_FORMAT = ['.bmp', '.jpg', '.tif', '.png']  # 图片格式
    # 获取图片集地址下的所有图片名称
    image_names = [name for name in os.listdir(IMAGES_PATH) for item in IMAGES_FORMAT if
                   os.path.splitext(name)[1] == item]

    # 排序，这里需要根据自己的图片名称切割，得到数字
    image_names.sort(key=lambda x: int(x.split(("."), 2)[0]))
    # 简单的对于参数的设定和实际图片集的大小进行数量判断
    if len(image_names) != IMAGE_ROW * IMAGE_COLUMN:
        raise ValueError("合成图片的参数和要求的数量不能匹配！")

    to_image = Image.new('RGB', (
    IMAGE_COLUMN * IMAGE_SIZE[0] + padding * (IMAGE_COLUMN - 1), IMAGE_ROW * IMAGE_SIZE[1] + padding * (IMAGE_ROW - 1)),
                         'white')  # 创建一个新图,颜色为白色
    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    i=0
    for y in range(1, IMAGE_COLUMN + 1):
        for x in range(1, IMAGE_ROW + 1):
            from_image = Image.open(IMAGES_PATH + image_names[i])
            to_image.paste(from_image, (
                (y - 1) * IMAGE_SIZE[0] + padding * (y - 1), (x - 1) * IMAGE_SIZE[1] + padding * (x - 1)))
            i=i+1
    to_image.save(IMAGE_SAVE_PATH)
    img=cv2.imread(IMAGE_SAVE_PATH,0)
    img=img[0:row_shape[1],0:row_shape[0]]
    return cv2.imwrite(IMAGE_SAVE_PATH,img)  # 保存新图

#===========================================================================================================#

#====================================计算中心点===============================================================#

def PrepareProb(img, convertuint8=True, inverse=True):
    """
    Prepares the prob image for post-processing, it can convert from
    float -> to uint8 and it can inverse it if needed.
    """
    if convertuint8:
        img = img_as_ubyte(img)
    if inverse:
        img = 255 - img
    return img

def HreconstructionErosion(prob_img, h):
    """
    Performs a H minimma reconstruction via an erosion method.
    """

    def making_top_mask(x, lamb=h):
       return min(255, x + lamb)

    f = np.vectorize(making_top_mask)
    shift_prob_img = f(prob_img)

    seed = shift_prob_img
    mask = prob_img
    recons = reconstruction(
        seed, mask, method='erosion').astype(np.dtype('ubyte'))
    return recons

def find_maxima(img, convertuint8=False, inverse=False, mask=None):
    """
    Finds all local maxima from 2D image.
    """
    img = PrepareProb(img, convertuint8=convertuint8, inverse=inverse)
    recons = HreconstructionErosion(img, 1)
    if mask is None:
        return recons - img
    else:
        res = recons - img
        res[mask==0] = 0
        return res
def generate_wsl(ws):
    """
    Generates watershed line that correspond to areas of
    touching objects.
    """
    se = square(3)
    ero = ws.copy()
    ero[ero == 0] = ero.max() + 1
    ero = erosion(ero, se)
    ero[ws == 0] = 0

    grad = dilation(ws, se) - ero
    grad[ws == 0] = 0
    grad[grad > 0] = 255
    grad = grad.astype(np.uint8)
    return grad
def DynamicWatershedAlias(p_img, lamb, p_thresh=0.5):
    """
    Applies our dynamic watershed to 2D prob/dist image.
    """
    b_img = (p_img > p_thresh) + 0
    Probs_inv = PrepareProb(p_img)

    Hrecons = HreconstructionErosion(Probs_inv, lamb)
    markers_Probs_inv = find_maxima(Hrecons, mask=b_img)
    markers_Probs_inv = label(markers_Probs_inv)
    ws_labels = watershed(Hrecons, markers_Probs_inv, mask=b_img)
    arrange_label = ArrangeLabel(ws_labels)
    wsl = generate_wsl(arrange_label)
    arrange_label[wsl > 0] = 0

    return arrange_label

def PostProcess(prob_image, param=7, thresh = 0.5):
    """
    Perform DynamicWatershedAlias with some default parameters.
    """
    segmentation_mask = DynamicWatershedAlias(prob_image, param, thresh)
    return segmentation_mask

def ArrangeLabel(mat):
    """
    Arrange label image as to effectively put background to 0.
    """
    val, counts = np.unique(mat, return_counts=True)
    background_val = val[np.argmax(counts)]
    mat = label(mat, background = background_val)
    if np.min(mat) < 0:
        mat += np.min(mat)
        mat = ArrangeLabel(mat)
    return mat
def DistanceWithoutNormalise(bin_image):
    res = np.zeros_like(bin_image)
    for j in range(1, bin_image.max() + 1):
        one_cell = np.zeros_like(bin_image)
        one_cell[bin_image == j] = 1
        one_cell = distance_transform_cdt(one_cell)
        res[bin_image == j] = one_cell[bin_image == j]
    res = res.astype('uint8')
    return res

def count_center(path):
    img=cv2.imread(path,0)
    img[img>240]=255
    img[img<=240]=0
    img = DistanceWithoutNormalise(img)
    x=PostProcess(img,1,0)
    x[x>0]=255
    cv2.imwrite(path,x)
    count_center=[]
    groundtruth = cv2.imread(path)[:, :, 0]
    groundtruth[groundtruth>240]=255
    groundtruth[groundtruth<=240]=0
    contours, cnt = cv2.findContours(groundtruth.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        M = cv2.moments(contours[i])  # 计算第一条轮廓的各阶矩,字典形式
        if M["m00"]==0:
            continue
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        # cv2.drawContours(image, contours, 0, 255, -1)#绘制轮廓，填充
        count_center.append([center_x,center_y])
    return count_center

#===========================================================================================================#
