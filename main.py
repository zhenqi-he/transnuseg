import copy
import logging
import math

from os.path import join as pjoin
import cv2
import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from torchvision import transforms
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from models_4outputs import SwinTransformerSys_modified_4branches
import os

from torch.utils.data import Dataset,DataLoader,TensorDataset
from torch.nn.modules.loss import CrossEntropyLoss
import torch.optim as optim
from torchvision import transforms
import torch.utils.data as data
import scipy.io as sio
import matplotlib.pyplot as plt
import random

import time
import logging
import sys
from datetime import datetime
import argparse

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import torch.optim as optim
from dataset import Histology,MyDataset,MoNuSeg
# from CA import *
from model_sharedAtt import SwinTransformerSys_modified_shared_MLP
from model_MLP import SwinTransformerSys_modified_MLP
from utils import *
from model import *
from modified1 import *
from modified2 import *
from swin import *
from modified3 import *
from modified4 import *
from model_sharedAttention import SharedSwinTransformerSys_modified
from model_sharedAttention2 import SharedSwinTransformerSys_modified2
from model_sharedAttention3 import SharedSwinTransformerSys_modified3
from ND_Crossentropy import *
from lovasz_loss import *
from hausdorff import *
from focal_loss import *
from boundary_loss import *
from Dice_loss import *

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

num_classes = 2
IMG_SIZE = 512
PATCH_SIZE = 4
IN_CHANS = 1
EMBED_DIM = 96
DEPTHS = [2, 2, 2, 2]
NUM_HEADS = [3, 6, 12, 24]
WINDOW_SIZE = 8 #original 7 --> 8
MLP_RATIO = 4
QKV_BIAS = True
QK_SCALE = None
DROP_RATE = 0.0
DROP_PATH_RATE = 0.1
APE = False
PATCH_NORM = True 
USE_CHECKPOINT = False
PRETRAIN_CKPT = None


base_lr = 0.0001
WARMUP_LR = 5e-7
MIN_LR = 5e-6
batch_size = 1
num_epoch = 100


def main():


    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default="swin-unet-modified1") ### options : swin-unet, swin-unet-modified1,swin-unet-modified2
    parser.add_argument("--alpha",default=0.3)
    parser.add_argument("--beta",default=0.3)
    parser.add_argument("--gamma",default=0.3)
    parser.add_argument("--sharing_ratio",default=0.5)
    parser.add_argument("--dataset",default="Histology")
    parser.add_argument("--model_path",default="")

    args = parser.parse_args()
    
    model_type = args.model_type
    dataset = args.dataset

    alpha = float(args.alpha)
    beta = float(args.beta)
    gamma = float(args.gamma)
    sharing_ratio = float(args.sharing_ratio)

    if dataset == "Radiology":
        channel = 1
    elif dataset == "Histology":
        channel = 3
        IMG_SIZE = 512
    else:
        logging.info("Wrong Dataset type")
        return 0
    
    
    
    
    model = TransNucSeg(img_size=IMG_SIZE,num_classes=num_classes,in_chans = channel,num_heads=NUM_HEADS,depths=DEPTHS,shared_ratio=sharing_ratio,window_size=8)
  
 
    model.to(device)

    now = datetime.now()
    logging.basicConfig(filename='/root/Swin_unet/loggings/log_{}.txt'.format(str(now)), level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("DATASET type {}".format(dataset))
    logging.info("Batch size : {} , epoch num: {}, alph: {}, beta : {}, gamma: {}, sharing_ratio = {}".format(batch_size,num_epoch,alpha,beta,gamma,sharing_ratio))

    
    if dataset == "Radiology":
        total_data = MyDataset()
        train_set_size = int(len(total_data) * 0.8)
        test_set_size = len(total_data) - train_set_size

        train_set, test_set = data.random_split(total_data, [train_set_size, test_set_size],generator=torch.Generator().manual_seed(666))
    elif dataset == "Histology":
        data_path = "/root/Swin_unet/dataset/histology/histology_train"
        train_set = Histology(dir_path = os.path.join(data_path,"train"),transform = None)
        test_set = Histology(dir_path = os.path.join(data_path,"test"),transform = None)
        # logging.info("train size {} test size {}".format(train_set_size,test_set_size))

        # train_set, test_set = data.random_split(total_data, [train_set_size, test_set_size],generator=torch.Generator().manual_seed(21))
        
    else:
        logging.info("Wrong Dataset type")
        return 0

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
 
    dataloaders = {"train":trainloader,"test":testloader}
    dataset_sizes = {"train":len(trainloader),"test":len(testloader)}
    logging.info("size train : {}, size test {} ".format(dataset_sizes["train"],dataset_sizes["test"]))
        
    test_loss = []
    train_loss = []
    lr_lists = []
        
    
    
    ce_loss1 = CrossEntropyLoss()
    dice_loss1 = DiceLoss(num_classes)
    ce_loss2 = CrossEntropyLoss()
    dice_loss2 = DiceLoss(num_classes)
    ce_loss3 = CrossEntropyLoss()
    dice_loss3 = DiceLoss(num_classes)


  

    optimizer = optim.Adam(model.parameters(), lr=base_lr)
  

    best_loss = 100
    best_epoch = 0


    for epoch in range(num_epoch):
        if epoch > best_epoch + 50:
            break
        for phase in ['train','test']:
            running_loss = 0
            s = time.time()
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   

            for i, d in enumerate(dataloaders[phase]):
              
                img, instance_seg_mask, semantic_seg_mask,normal_edge_mask,cluster_edge_mask = d
             
                img = img.float()    
                img = img.to(device)
                instance_seg_mask = instance_seg_mask.to(device)
                semantic_seg_mask = semantic_seg_mask.to(device)
                cluster_edge_mask = cluster_edge_mask.to(device)

                output1,output2,output3 = model(img)
                loss_seg = 0.4*ce_loss1(output1, semantic_seg_mask[0].long()) + 0.6*dice_loss1(output1, semantic_seg_mask[0].float(), softmax=True)
                loss_nor = 0.4*ce_loss2(output2, normal_edge_mask[0].long()) + 0.6*dice_loss2(output2, normal_edge_mask[0].float(), softmax=True)
                loss_clu = 0.4*ce_loss3(output3, cluster_edge_mask[0].long()) + 0.6*dice_loss3(output3, cluster_edge_mask[0].float(), softmax=True)
                
                if epoch < 10:
                    ratio_d = 1
                elif epoch < 20:
                    ratio_d = 0.7
                elif epoch < 30:
                    ratio_d = 0.3
                elif epoch < 40:
                    ratio_d = 0.1
                else:
                    ratio_d = 0
                
                m = torch.argmax(torch.softmax(output1, dim=1), dim=1).squeeze(0)
                m = m.cpu().detach().numpy()
                b = torch.argmax(torch.softmax(output2, dim=1), dim=1)
                c = torch.argmax(torch.softmax(output3, dim=1), dim=1)
                pred_edge_1 = edge_detection(m.copy(),channel)
                pred_edge_1 = torch.tensor(pred_edge_1).to(device)
                pred_edge_2 = output2-output3
                pred_edge_2[pred_edge_2<0] = 0
                
                dis_loss = dice_loss_dis(pred_edge_2,pred_edge_1.unsqueeze(0).float())
                
                loss = alpha*loss_seg + beta*loss_nor + gamma*loss_clu + ratio_d*dis_loss

                running_loss+=loss.item()
            if phase == 'train':
                with torch.autograd.set_detect_anomaly(True):
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
            e = time.time()
            epoch_loss = running_loss / dataset_sizes[phase]
            logging.info('Epoch {},: loss {}, {},time {}'.format(epoch+1,  epoch_loss,phase,e-s))

            if phase == 'train':
                train_loss.append(epoch_loss)
            else:
                test_loss.append(epoch_loss)

            if phase == 'test' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_epoch = epoch+1
                best_model_wts = copy.deepcopy(model.state_dict())
                logging.info("Best val loss {} save at epoch {}".format(best_loss,epoch+1))

    draw_loss(train_loss,val_loss,str(now))
        
    torch.save(best_model_wts, '/root/Swin_unet/models/model_spoch:{}_valloss:{}_{}.pt'.format(best_epoch,best_loss,str(now)))
    logging.info('Model saved. at {}'.format('/root/Swin_unet/models/model_spoch:{}_valloss:{}_{}.pt'.format(best_epoch,best_loss,str(now))))


    
    model.load_state_dict(best_model_wts)
    model.eval()

    dice_acc_test = 0
    dice_loss_test = DiceLoss(num_classes)
    dice_acc_test2 = 0
    dice_loss_test2 = DiceLoss(num_classes)
    F1 = 0
    F1_2 = 0
    acc = 0
    acc2 = 0
    Iou = 0
    aji = 0
    aji_smooth = 0
    ajip_smooth = 0
    aji2 = 0
    ajip = 0
    iou2 = 0
    pq1 = 0
    pq2 = 0
    pq3 = 0
    with torch.no_grad():
        for i, d in enumerate(testloader, 0):
            img, instance_seg_mask, semantic_seg_mask,normal_edge_mask,cluster_edge_mask = d
            # img = img.unsqueeze(0)
            img = img.float()    
            img = img.to(device)

            # semantic_seg_mask = semantic_seg_mask.unsqueeze(0).float()
            
            if model_type == "swin-unet":
                output1 = model(img)
                d_l = dice_loss_test(output1, semantic_seg_mask[0].float(), softmax=True)
                dice_acc_test += 1- d_l.item()
            elif model_type == "swin-unet-4":
                output1,output2,output3,output4 = model(img)
                d_l = dice_loss_test(output1, semantic_seg_mask[0].float(), softmax=True)
                dice_acc_test += 1- d_l.item()
            else:
                output1,output2,output3 = model(img)
                
                d_l = dice_loss_test(output1, semantic_seg_mask[0].float(), softmax=True)
                dice_acc_test += 1- d_l.item()
                
#                 sum_output = output1.clone()-output2.clone()-output3.clone()
#                 d_l2 = dice_loss_test2(sum_output,semantic_seg_mask[0].float(), softmax=True)
                
#                 dice_acc_test2 += 1- d_l2.item()

            # semantic_seg_mask = semantic_seg_mask.squeeze(0).detach().cpu().numpy()
            # instance_seg_mask = instance_seg_mask.squeeze(0).detach().cpu().numpy()
            semantic_seg_mask = semantic_seg_mask.squeeze(0).squeeze(0).detach().cpu().numpy()
            instance_seg_mask = instance_seg_mask.squeeze(0).squeeze(0).detach().cpu().numpy()

            if model_type == "swin-unet":
                m = torch.argmax(torch.softmax(output1, dim=1), dim=1).squeeze(0)
                m = m.cpu().detach().numpy()
                ins_predict = sem2ins(m.copy(),m.copy(),m.copy(),3)
                
                result = m.copy()
                
                F1 += calculate_F1_score(result,semantic_seg_mask)
                acc += calculate_acc(result,semantic_seg_mask)
                Iou += calculate_IoU(result,semantic_seg_mask)
                aji += AJI(instance_seg_mask,ins_predict)
                iou2 += get_iou(result,semantic_seg_mask)
                
                
              
            else:
                m = torch.argmax(torch.softmax(output1, dim=1), dim=1).squeeze(0)
                m = m.cpu().detach().numpy()

                b = torch.argmax(torch.softmax(output2, dim=1), dim=1).squeeze(0)
                b = b.cpu().detach().numpy()

                c = torch.argmax(torch.softmax(output3, dim=1), dim=1).squeeze(0)
                c = c.cpu().detach().numpy()


                result = m.copy() + b.copy() + c.copy()
                ins_predict_unsharpen = sem2ins(m.copy(),b.copy(),c.copy(),0)
                ins_predict_smooth = sem2ins_smooth(m.copy(),b.copy(),c.copy())
                # ins_predict_sharpen = sem2ins(m.copy(),b.copy(),c.copy(),1)
                Iou += float(calculate_IoU(m.copy(),semantic_seg_mask))
                logging.info("{}th iou {}, iou_sum {}".format(i,calculate_IoU(result,semantic_seg_mask),Iou))
                F1 += calculate_F1_score(result,semantic_seg_mask)
                F1_2 += calculate_F1_score(m.copy(),semantic_seg_mask)
                acc += calculate_acc(result,semantic_seg_mask)
                acc2 += calculate_acc(m.copy(),semantic_seg_mask)
                pq_stat =  get_fast_pq(instance_seg_mask,ins_predict_smooth)[0]
                pg1 += pq_stat[0]
                pg2 += pq_stat[1]
                pg3 += pq_stat[2]
                
                aji += get_fast_aji(instance_seg_mask,ins_predict_unsharpen)
                aji_smooth += get_fast_aji(instance_seg_mask,ins_predict_smooth)
                ajip += get_fast_aji_plus(instance_seg_mask,ins_predict_smooth)
                # aji2 += AJI(instance_seg_mask,ins_predict_sharpen)
                iou2 += float(get_iou(result,semantic_seg_mask))


            # cv2.imwrite("/root/Swin_unet/outputs/"+str(i)+".png",255*ins_predict)
            # cv2.imwrite("/root/Swin_unet/dataset/instance_masks/"+str(i)+".png",instance_seg_mask)

            


    # print(dice_acc_test)
    if model_type == 'swin-unet':
        logging.info("dice_loss {}".format(dice_acc_test/dataset_sizes['test']))

        logging.info("F1 {}".format(F1/dataset_sizes['test']))
        logging.info("acc {}".format(acc/dataset_sizes['test']))
        logging.info("IOU values {}, dataset len {}".format(IOU,dataset_sizes['test']))
        logging.info("Iou ".format(Iou/dataset_sizes['test']))
        logging.info("Iou2 ".format(iou2/dataset_sizes['test']))
        logging.info("AJI {}".format(aji/dataset_sizes['test']))

    else:
        logging.info("dice_acc {}".format(dice_acc_test/dataset_sizes['test']))
        logging.info("dice_acc2 {}".format(dice_acc_test2/dataset_sizes['test']))
        logging.info("F1 {}".format(F1/dataset_sizes['test']))
        logging.info("F1_2 {}".format(F1_2/dataset_sizes['test']))
        logging.info("acc {}".format(acc/dataset_sizes['test']))
        logging.info("acc2 {}".format(acc2/dataset_sizes['test']))
        logging.info("IOU values {}, dataset len {}".format(IOU,dataset_sizes['test']))
        logging.info("Iou ".format(Iou/dataset_sizes['test']))
        logging.info("IOU {} dateset len {}".format(Iou,dataset_sizes['test']))
        logging.info("Iou2 ".format(iou2/dataset_sizes['test']))
        logging.info("AJI unshrpen {}".format(aji/dataset_sizes['test']))
        logging.info("AJI plus {}".format(ajip/dataset_sizes['test']))
        logging.info("PQ1 {}".format(pq1/dataset_sizes['test']))
        logging.info("PQ2  {}".format(pq2/dataset_sizes['test']))
        logging.info("PQ3  {}".format(pq3/dataset_sizes['test']))
  
if __name__=='__main__':
    main()
