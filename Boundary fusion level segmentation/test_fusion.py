import torch.optim
from Load_Dataset import ValGenerator, ImageToImage2D
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")
import Config as config
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from utils import *
import cv2
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import recall_score, precision_score, confusion_matrix
from skimage.measure import label
from sklearn.metrics import f1_score
#from nets.UNet import UNet as UNet_distbound
from nets.UNet import UNet as UNet_distbound
from nets.patch_dense_net import UNet
from postpro import PostProcess
from nets.neighbor.UNet import UNet as UNet_neighbor
input_modalities = 3
n_pred_labels=1

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


def show_image_with_dice(predict_save, labs, save_path):

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
    #print("===========predict_save0",predict_save)
    if config.task_name is "MoNuSeg":
        #predict_save = cv2.pyrUp(predict_save,(448,448))
        #print("=======predict_saveup",predict_save)
        # predict_save = cv2.resize(predict_save,(2000,2000))
        # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #定义一个核
        # predict_save = cv2.filter2D(predict_save, -1, kernel=kernel)
        cv2.imwrite(save_path,predict_save * 255)
    else:
        cv2.imwrite(save_path,predict_save * 255)
    return dice_pred, iou_pred,precision,recall,acc,roc,f1,aji

def vis_and_save_heatmap(model,input_img,test_segresult,test_boundresult,img_RGB, labs, vis_save_path, dice_pred, dice_ens):


    test_segresult, test_boundresult = test_segresult.cuda(), test_boundresult.cuda()
    test_segresult=test_segresult.unsqueeze(1)
    test_boundresult=test_boundresult.unsqueeze(1)
    print(test_boundresult.shape)
    print(input_img.shape)
    model.eval()
    output = model(input_img.cuda(), torch.cat((test_segresult, test_boundresult), dim=1))
    #output=output-neighbors
    pred_class = torch.where(output>0.5,torch.ones_like(output),torch.zeros_like(output))
    predict_save = pred_class[0].cpu().data.numpy()
    predict_save = np.reshape(predict_save, (config.img_size, config.img_size))
    dice_pred_tmp, iou_tmp,precision,recall,acc,roc,f1,aji = show_image_with_dice(predict_save, labs, save_path=vis_save_path+'_predict'+model_type+'.jpg')
    return dice_pred_tmp, iou_tmp,precision,recall,acc,roc,f1,aji



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    test_session = config.test_session
    if config.task_name is "GlaS":
        test_num = 80
        model_type = config.model_name
        model_path = "./GlaS/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name is "MoNuSeg":
        test_num = 65
        model_type = config.model_name
        model_path = "./MoNuSeg/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"
    #model_path ="./MoNuSeg_best/models/best_model-UNet.pth.tar"

    save_path  = config.task_name +'/'+ model_type +'/' + test_session + '/'
    vis_path = "./" + config.task_name + '_visualize_test/'
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)

    checkpoint = torch.load(model_path, map_location='cuda')

    config_vit = config.get_CTranS_config()
    #model = UNet(config_vit,n_channels=config.n_channels,n_classes=config.n_labels)
    model=UNet(input_modalities, n_pred_labels*2, n_pred_labels)
    model = model.cuda()

    if torch.cuda.device_count() > 1:
        print ("Let's use {0} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=[0,1,2,3])
    model.load_state_dict(checkpoint['state_dict'])
    print('Model loaded !')
    tf_test = ValGenerator(output_size=[config.img_size, config.img_size])
    test_dataset = ImageToImage2D(config.test_dataset, tf_test,image_size=config.img_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    test_num=len(test_loader)
    dice_pred = 0.0
    iou_pred = 0.0
    dice_ens = 0.0
    precision_pred = 0.0
    recall_pred = 0.0
    acc_pred = 0.0
    roc_pred = 0.0
    f1_pred = 0.0
    aji_pred = 0.0
    with tqdm(total=test_num, desc='Test visualize', unit='img', ncols=70, leave=True) as pbar:
        for i, (sampled_batch, names) in enumerate(test_loader, 1):
            test_data, test_label,test_segresult,test_boundresult = sampled_batch['image'], sampled_batch['label'], sampled_batch['segresult'], sampled_batch['boundresult']
            #test_data, test_label = test_data.cuda(), test_label.cuda()
            arr=test_data.numpy()
            arr = arr.astype(np.float32())

            lab=test_label.data.numpy()

            img_lab = np.reshape(lab, (lab.shape[1], lab.shape[2])) * 255
            fig, ax = plt.subplots()
            plt.imshow(img_lab, cmap='gray')
            plt.axis("off")
            height, width = config.img_size, config.img_size
            fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.savefig(vis_path+names[0]+"_lab.jpg", dpi=300)
            plt.close()

            input_img = torch.from_numpy(arr)
            dice_pred_t,iou_pred_t,precision_pred_t,recall_pred_t,acc_pred_t,roc_pred_t,f1_pred_t,aji_pred_t = vis_and_save_heatmap(model,input_img, test_segresult,test_boundresult,None, lab,
                                                          vis_path+names[0],
                                               dice_pred=dice_pred, dice_ens=dice_ens)
            dice_pred+=dice_pred_t
            iou_pred+=iou_pred_t
            precision_pred += precision_pred_t
            recall_pred += recall_pred_t
            acc_pred += acc_pred_t
            roc_pred += roc_pred_t
            f1_pred += f1_pred_t
            aji_pred += aji_pred_t
            torch.cuda.empty_cache()
            pbar.update()
    print ("dice_pred",dice_pred/test_num)
    print ("iou_pred",iou_pred/test_num)
    print ("precision_pred",precision_pred/test_num)
    print ("recall_pred",recall_pred/test_num)
    print ("acc_pred",acc_pred/test_num)
    print ("roc_pred",roc_pred/test_num)
    print ("f1_pred",f1_pred/test_num)
    print ("aji_pred",aji_pred/test_num)




