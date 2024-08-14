import os

import torch
from tqdm import tqdm

from utils.utils import get_lr


def fit_one_epoch(model_train, model, multi_loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir,height, local_rank=0):
    total_loss  = 0
    val_loss    = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    #print("model_train.train()")
    model_train.train()
    #print("model_train.train() finish")
    for iteration, batch in enumerate(gen):
        #print("iteration")
        if iteration >= epoch_step:
            break
        images, targets, masks_gt,boundarys_gt, num_crowds = batch[0], batch[1], batch[2], batch[3],batch[4]
        with torch.no_grad():
            if cuda:
                #print("cuda true")
                images      = images.cuda(local_rank)
                targets     = [ann.cuda(local_rank) for ann in targets]
                masks_gt    = [mask.cuda(local_rank) for mask in masks_gt]
                boundarys_gt    = [boundary.cuda(local_rank) for boundary in boundarys_gt]
        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        if not fp16:
            #----------------------#
            #   前向传播
            #----------------------#
            #print("===========model_train(images)")
            outputs = model_train(images)
            #print("===========model_train(images)")
            #----------------------#
            #   计算损失
            #----------------------#
            losses  = multi_loss(outputs, targets, masks_gt,boundarys_gt, num_crowds,height)
            #print("losses1================",losses)
            losses  = {k: v.mean() for k, v in losses.items()}  # Mean here because Dataparallel.
            #print("losses2================",losses)
            loss    = sum([losses[k] for k in losses])
            #print("losses3================",loss)

            #----------------------#
            #   反向传播
            #----------------------#
            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                #----------------------#
                #   前向传播
                #----------------------#
                outputs = model_train(images)
                #----------------------#
                #   计算损失
                #----------------------#
                losses  = multi_loss(outputs, targets, masks_gt,boundarys_gt, num_crowds)
                losses  = {k: v.mean() for k, v in losses.items()}  # Mean here because Dataparallel.
                loss    = sum([losses[k] for k in losses])

            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        total_loss += loss.item()
        
        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, targets, masks_gt,boundarys_gt, num_crowds = batch[0], batch[1], batch[2], batch[3],batch[4]
        with torch.no_grad():
            if cuda:
                images      = images.cuda(local_rank)
                targets     = [ann.cuda(local_rank) for ann in targets]
                masks_gt    = [mask.cuda(local_rank) for mask in masks_gt]
                boundarys_gt    = [boundary.cuda(local_rank) for boundary in boundarys_gt]

            optimizer.zero_grad()
            #----------------------#
            #   前向传播
            #----------------------#
            outputs = model_train(images)
            #----------------------#
            #   计算损失
            #----------------------#
            losses      = multi_loss(outputs, targets, masks_gt,boundarys_gt, num_crowds,height)
            losses      = {k: v.mean() for k, v in losses.items()}
            loss        = sum([losses[k] for k in losses])

            val_loss += loss.item()
            
            if local_rank == 0:
                pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1), 
                                    'lr'        : get_lr(optimizer)})
                pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))
            
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
