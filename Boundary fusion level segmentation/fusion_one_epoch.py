# -*- coding: utf-8 -*-
# @Time    : 2021/6/19 2:14 下午
# @Author  : Haonan Wang
# @File    : Train_one_epoch.py
# @Software: PyCharm
import numpy as np
import torch
import torch.optim
import time
from utils import *
import Config as config
import warnings

from skimage.measure import label

import os

warnings.filterwarnings("ignore")




def nms(boxes, scores, nms_thresh=0.5, top_k=200):
    # boxes shape[-1, 4]
    # scores shape [-1,]
    scores = scores
    boxes = boxes
    keep = scores.new(scores.size(0)).zero_().long()# create a new tensor
    if boxes.numel() == 0: # return the total elements number in boxes
        return keep
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    area = torch.mul(y2-y1,x2-x1)
    v, idx = scores.sort(0)
    idx = idx[-top_k:]
    yy1 = boxes.new()  # create a new tensor of the same type
    xx1 = boxes.new()
    yy2 = boxes.new()
    xx2 = boxes.new()
    h = boxes.new()
    w = boxes.new()
    count = 0
    while idx.numel()>0:
        i = idx[-1]
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]

        # doing about remains...
        # select the remaining boxes
        torch.index_select(y1, dim=0, index=idx, out=yy1)
        torch.index_select(x1, dim=0, index=idx, out=xx1)
        torch.index_select(y2, dim=0, index=idx, out=yy2)
        torch.index_select(x2, dim=0, index=idx, out=xx2)

        # calculate the inter boxes clamp with box i
        yy1 = torch.clamp(yy1, min=y1[i])
        xx1 = torch.clamp(xx1, min=x1[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        xx2 = torch.clamp(xx2, max=x2[i])

        w.resize_as_(xx2)
        h.resize_as_(yy2)

        w = xx2-xx1
        h = yy2-yy1

        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)

        inter = w*h

        rem_areas = torch.index_select(area, dim=0, index=idx)
        union = (rem_areas-inter)+area[i]
        IoU = inter/union
        idx = idx[IoU.le(nms_thresh)]
    return keep, count

def diceCoeff(pred, gt, smooth=1, activation='sigmoid'):
    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    loss = (2 * intersection + smooth) / (unionset + smooth)

    return loss.sum() / N


def print_summary(epoch, i, nb_batch, loss, loss_name, batch_time,
                  average_loss, average_time, iou, average_iou,
                  dice, average_dice, acc, average_acc, mode, lr, logger):
    '''
        mode = Train or Test
    '''
    summary = '   [' + str(mode) + '] Epoch: [{0}][{1}/{2}]  '.format(
        epoch, i, nb_batch)
    string = ''
    string += 'Loss:{:.3f} '.format(loss)
    string += '(Avg {:.4f}) '.format(average_loss)
    # string += 'IoU:{:.3f} '.format(iou)
    # string += '(Avg {:.4f}) '.format(average_iou)
    string += 'Dice:{:.4f} '.format(dice)
    string += '(Avg {:.4f}) '.format(average_dice)
    # string += 'Acc:{:.3f} '.format(acc)
    # string += '(Avg {:.4f}) '.format(average_acc)
    if mode == 'Train':
        string += 'LR {:.2e}   '.format(lr)
    # string += 'Time {:.1f} '.format(batch_time)
    string += '(AvgTime {:.1f})   '.format(average_time)
    summary += string
    logger.info(summary)
    # print summary

def load_dec_weights(dec_model, dec_weights):
    dec_dict = torch.load(dec_weights,map_location=torch.device('cpu'))
    dec_dict_update = {}
    for k in dec_dict:
        if k.startswith('module') and not k.startswith('module_list'):
            dec_dict_update[k[7:]] = dec_dict[k]
        else:
            dec_dict_update[k] = dec_dict[k]
    dec_model.load_state_dict(dec_dict_update, strict=True)
    return dec_model
##################################################################################
# =================================================================================
#          Train One Epoch
# =================================================================================
##################################################################################
def fusion_one_epoch(loader,model, criterion, optimizer, writer, epoch, lr_scheduler, model_type, logger):
    logging_mode = 'Train' if model.training else 'Val'


    end = time.time()
    time_sum, loss_sum = 0, 0
    dice_sum, iou_sum, acc_sum = 0.0, 0.0, 0.0

    dices = []
    for www, (sampled_batch, names) in enumerate(loader, 1):

        try:
            loss_name = criterion._get_name()
        except AttributeError:
            loss_name = criterion.__name__

        images, masks,segresults,boundresults = sampled_batch['image'], sampled_batch['label'],sampled_batch['segresult'],sampled_batch['boundresult']

        images, masks,segresults,boundresults = images.cuda(), masks.cuda(),segresults.cuda(),boundresults.cuda()
        segresults=segresults.unsqueeze(1)
        boundresults=boundresults.unsqueeze(1)
        # -----------------load  model -------------------------
        preds = model(images.cuda(), torch.cat((segresults.cuda(), boundresults.cuda()), dim=1))
        out_loss=criterion(preds,masks.float())

        if model.training:
            optimizer.zero_grad()
            out_loss.backward()
            optimizer.step()

        masks=masks.cpu().detach().numpy()
        preds=preds.cpu().detach().numpy()
        masks = label(masks)


        train_iou = iou_on_batch(torch.tensor(masks), torch.tensor(preds))
        train_dice = criterion._show_dice(torch.tensor(preds), torch.tensor(masks))

        batch_time = time.time() - end
        # train_acc = acc_on_batch(masks,preds)
        if epoch % config.vis_frequency == 0 and logging_mode is 'Val':
            vis_path = config.visualize_path + str(epoch) + '/'
            if not os.path.isdir(vis_path):
                os.makedirs(vis_path)
            save_on_batch(images, torch.tensor(masks), torch.tensor(preds), names, vis_path)
        dices.append(train_dice)

        time_sum += len(images) * batch_time
        loss_sum += len(images) * out_loss
        iou_sum += len(images) * train_iou
        # acc_sum += len(images) * train_acc
        dice_sum += len(images) * train_dice
        if www == len(loader):
            average_loss = loss_sum / (config.batch_size * (www - 1) + len(images))
            average_time = time_sum / (config.batch_size * (www - 1) + len(images))
            train_iou_average = iou_sum / (config.batch_size * (www - 1) + len(images))
            # train_acc_average = acc_sum / (config.batch_size*(i-1) + len(images))
            train_dice_avg = dice_sum / (config.batch_size * (www - 1) + len(images))
        else:
            average_loss = loss_sum / (www * config.batch_size)
            average_time = time_sum / (www * config.batch_size)
            train_iou_average = iou_sum / (www * config.batch_size)
            # train_acc_average = acc_sum / (i * config.batch_size)
            train_dice_avg = dice_sum / (www * config.batch_size)

        end = time.time()
        torch.cuda.empty_cache()

        if www % config.print_frequency == 0:
            print_summary(epoch + 1, www, len(loader), out_loss, loss_name, batch_time,
                          average_loss, average_time, train_iou, train_iou_average,
                          train_dice, train_dice_avg, 0, 0, logging_mode,
                          lr=min(g["lr"] for g in optimizer.param_groups), logger=logger)

        if config.tensorboard:
            step = epoch * len(loader) + www
            writer.add_scalar(logging_mode + '_' + loss_name, out_loss.item(), step)

            # plot metrics in tensorboard
            writer.add_scalar(logging_mode + '_iou', train_iou, step)
            # writer.add_scalar(logging_mode + '_acc', train_acc, step)
            writer.add_scalar(logging_mode + '_dice', train_dice, step)

        torch.cuda.empty_cache()


    if lr_scheduler is not None:
        lr_scheduler.step()
# if epoch + 1 > 10: # Plateau
#     if lr_scheduler is not None:
#         lr_scheduler.step(train_dice_avg)
    return average_loss, train_dice_avg
