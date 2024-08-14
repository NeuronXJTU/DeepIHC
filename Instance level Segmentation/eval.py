import os
import os.path as osp

from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
import numpy as np
from utils.utils import get_classes, get_coco_label_map
from utils.utils_map import Make_json, prep_metrics
from utils.utils import cvtColor, preprocess_input
from yolact import YOLACT
from sklearn.metrics import roc_auc_score,jaccard_score

import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import recall_score, precision_score, confusion_matrix
from skimage.measure import label
from sklearn.metrics import f1_score
yolact = YOLACT()


def AJI_fast(G, S):
    """
    AJI as described in the paper, but a much faster implementation.
    """
    G = label(G, background=0)
    S = label(S, background=0)
    if S.sum() == 0:
        return 0.
    C = 0
    U = 0
    USED = np.zeros(S.max())

    G_flat = G.flatten()
    S_flat = S.flatten()
    G_max = np.max(G_flat)
    S_max = np.max(S_flat)
    m_labels = max(G_max, S_max) + 1
    cm = confusion_matrix(G_flat, S_flat, labels=range(m_labels)).astype(np.float)
    LIGNE_J = np.zeros(S_max)
    for j in range(1, S_max + 1):
        LIGNE_J[j - 1] = cm[:, j].sum()

    for i in range(1, G_max + 1):
        LIGNE_I_sum = cm[i, :].sum()

        def h(indice):
            LIGNE_J_sum = LIGNE_J[indice - 1]
            inter = cm[i, indice]

            union = LIGNE_I_sum + LIGNE_J_sum - inter
            return inter / union

        JI_ligne = map(h, range(1, S_max + 1))
        best_indice = np.argmax(JI_ligne) + 1
        C += cm[i, best_indice]
        U += LIGNE_J[best_indice - 1] + LIGNE_I_sum - cm[i, best_indice]
        USED[best_indice - 1] = 1

    U_sum = ((1 - USED) * LIGNE_J).sum()
    U += U_sum
    return float(C) / float(U)


def show_image_with_dice(predict_save, labs):

    tmp_lbl = (labs).astype(np.float32)
    tmp_3dunet = (predict_save).astype(np.float32)
    dice_pred = 2 * np.sum(tmp_lbl * tmp_3dunet) / (np.sum(tmp_lbl) + np.sum(tmp_3dunet) + 1e-5)
    iou_pred = jaccard_score(tmp_lbl.reshape(-1),tmp_3dunet.reshape(-1))
    precision = precision_score(tmp_lbl.reshape(-1), tmp_3dunet.reshape(-1))
    recall = recall_score(tmp_lbl.reshape(-1), tmp_3dunet.reshape(-1))
    acc = accuracy_score(tmp_lbl.reshape(-1), tmp_3dunet.reshape(-1))
    roc=0.6
    try:
        roc = roc_auc_score(tmp_lbl.reshape(-1), tmp_3dunet.reshape(-1))
    except ValueError:
        pass
    f1 = f1_score(tmp_lbl.reshape(-1), tmp_3dunet.reshape(-1))
    aji = AJI_fast(tmp_lbl.reshape(-1), tmp_3dunet.reshape(-1))
    return dice_pred, iou_pred,precision,recall,acc,roc,f1,aji


def dataloader1(image_path, coco, COCO_LABEL_MAP,index):

    image_id = ids[index]
    target     = coco.loadAnns(coco.getAnnIds(imgIds = image_id))
    target      = [x for x in target if not ('iscrowd' in x and x['iscrowd'])]
    crowd       = [x for x in target if ('iscrowd' in x and x['iscrowd'])]
    num_crowds  = len(crowd)
    target      += crowd
    image_path  = osp.join(image_path, coco.loadImgs(image_id)[0]['file_name'])
    image       = Image.open(image_path)
    mask_8, boundary_8 = yolact.eval_image(image)

    image       = cvtColor(image)
    image       = np.array(image, np.float32)
    height, width, _ = image.shape
    if len(target) > 0:
        masks = np.array([coco.annToMask(obj).reshape(-1) for obj in target], np.float32)
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
            final_box   = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3], label_map[obj['category_id']] - 1]
            boxes_classes.append(final_box)
            j=j+1
        boxes_classes = np.array(boxes_classes, np.float32)
        boxes_classes[:, [0, 2]] /= width
        boxes_classes[:, [1, 3]] /= height
    image = preprocess_input(image)
    masks=masks.sum(axis=0)
    masks[masks>0]=1
    print(boundarys.shape)
    boundarys=boundarys.sum(axis=0)
    boundarys[boundarys>0]=255
    import cv2
    cv2.imwrite("boundarys.png",boundarys)
    dice_pred, iou_pred, precision, recall, acc, roc, f1, aji=show_image_with_dice(mask_8,masks)

    return dice_pred, iou_pred, precision, recall, acc, roc, f1, aji


if __name__ == '__main__':
    classes_path    = 'model_data/shape_classes.txt'
    test_image_path        = "datasets/coco/JPEGImages"
    test_annotation_path   = "datasets/coco/Jsons/test_annotations.json"
    test_coco = COCO(test_annotation_path)
    class_names, num_classes = get_classes(classes_path)
    COCO_LABEL_MAP = get_coco_label_map(test_coco, class_names)

    ids = list(test_coco.imgToAnns.keys())
    label_map = COCO_LABEL_MAP
    length = len(ids)
    dice_pred = 0.0
    iou_pred = 0.0
    dice_ens = 0.0
    precision_pred = 0.0
    recall_pred = 0.0
    acc_pred = 0.0
    roc_pred = 0.0
    f1_pred = 0.0
    aji_pred = 0.0
    for index in range(length):
        dice_pred_t,iou_pred_t,precision_pred_t,recall_pred_t,acc_pred_t,roc_pred_t,f1_pred_t,aji_pred_t = dataloader1(test_image_path, test_coco, COCO_LABEL_MAP,index)
        dice_pred += dice_pred_t
        iou_pred += iou_pred_t
        precision_pred += precision_pred_t
        recall_pred += recall_pred_t
        acc_pred += acc_pred_t
        roc_pred += roc_pred_t
        f1_pred += f1_pred_t
        aji_pred += aji_pred_t
        
    print ("dice_pred",dice_pred/length)
    print ("iou_pred",iou_pred/length)
    print ("precision_pred",precision_pred/length)
    print ("recall_pred",recall_pred/length)
    print ("acc_pred",acc_pred/length)
    print ("roc_pred",roc_pred/length)
    print ("f1_pred",f1_pred/length)
    print ("aji_pred",aji_pred/length)
