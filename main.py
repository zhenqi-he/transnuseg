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
from model_sharedAttention_MLP import SharedSwinTransformerSys_MLP_modified
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
num_epoch = 400

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default="swin-unet-modified1") ### options : swin-unet, swin-unet-modified1,swin-unet-modified2
    parser.add_argument("--alpha",default=0.3)
    parser.add_argument("--beta",default=0.3)
    parser.add_argument("--gamma",default=0.3)
    parser.add_argument("--sharing_ratio",default=0.5)
    parser.add_argument("--dataset",default="Histology")
    parser.add_argument("--model_path",default="")
 
    parser.add_argument("--save_model_path",default="./")
    args = parser.parse_args()
    
    model_type = args.model_type
    dataset = args.dataset

    print("loss_type ",loss_type)   
    model_path = "/root/Swin_unet/models/model_spoch:90_valloss:0.21124622246173963_2022-09-30 23:36:14.063703.pt"
    alpha = float(args.alpha)
    beta = float(args.beta)
    gamma = float(args.gamma)
    sharing_ratio = float(args.sharing_ratio)
    save_model_path = args.save_model_path

    now = datetime.now()
    logging.basicConfig(filename='/root/Swin_unet/loggings/log_{}.txt'.format(str(now)), level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("Model type {}".format(model_type))
    logging.info("DATASET type {}".format(dataset))
    logging.info("Batch size : {} , epoch num: {}, alph: {}, beta : {}, gamma: {}, sharing_ratio = {}".format(batch_size,num_epoch,alpha,beta,gamma,sharing_ratio))
    logging.info("loss_type {}".format(loss_type))


    if dataset == "Radiology":
        channel = 1
        total_data = MyDataset()
        train_set_size = int(len(total_data) * 0.8)
        test_set_size = len(total_data) - train_set_size

        train_set, test_set = data.random_split(total_data, [train_set_size, test_set_size],generator=torch.Generator().manual_seed(666))
    elif dataset == "Histology":
        channel = 3
        data_path = "/root/Swin_unet/dataset/histology/histology_train"
        train_set = Histology(dir_path = os.path.join(data_path,"train"),transform = None)
        test_set = Histology(dir_path = os.path.join(data_path,"test"),transform = None)
        # logging.info("train size {} test size {}".format(train_set_size,test_set_size))

        # train_set, test_set = data.random_split(total_data, [train_set_size, test_set_size],generator=torch.Generator().manual_seed(21))
        
    elif dataset == "MoNuSeg":
        channel = 3
        train_set = MoNuSeg(dir_path = "/root/Swin_unet/dataset/MoNuSeg/MoNuSeg/Training")
        test_set = MoNuSeg(dir_path = "/root/Swin_unet/dataset/MoNuSeg/MoNuSeg/Test")
    else:
        logging.info("Wrong Dataset type")
        return 0
    
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    dataloaders = {"train":trainloader,"test":testloader}
    dataset_sizes = {"train":len(trainloader),"test":len(testloader)}
    logging.info("size train : {}, size test {} ".format(dataset_sizes["train"],dataset_sizes["test"]))

    model = load_model_by_name(model_type,channel,sharing_ratio)
    model.to(device)

    one_output_model_lists = ["swin-unet","unet++","transunet"]




    val_loss = []
    train_loss = []
    


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
                model.train()  
            else:
                model.eval()  
            
            for i, d in enumerate(dataloaders[phase]):
                img, instance_seg_mask, semantic_seg_mask,normal_edge_mask,cluster_edge_mask = d
                img = img.float()    
                img = img.to(device)
                instance_seg_mask = instance_seg_mask.to(device)
                semantic_seg_mask = semantic_seg_mask.to(device)
                cluster_edge_mask = cluster_edge_mask.to(device)

                if model_type in one_output_model_lists:
                    output1 = model(img)
                    loss_seg = 0.4*ce_loss1(output1, semantic_seg_mask[0].long()) + 0.6*dice_loss1(output1, semantic_seg_mask[0].float(), softmax=True)
                    loss = loss_seg
                else:
                    output1,output2,output3 = model(img)

                    loss_seg = 0.4*ce_loss1(output1, semantic_seg_mask[0].long()) + 0.6*dice_loss1(output1, semantic_seg_mask[0].float(), softmax=True)
                    loss_nor = 0.4*ce_loss2(output2, normal_edge_mask[0].long()) + 0.6*dice_loss2(output2, normal_edge_mask[0].float(), softmax=True)
                    loss_clu = 0.4*ce_loss3(output3, cluster_edge_mask[0].long()) + 0.6*dice_loss3(output3, cluster_edge_mask[0].float(), softmax=True)
                    loss = alpha*loss_seg + beta*loss_nor + gamma*loss_clu
           

                
                    
                running_loss += loss.item()
                if phase == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            e = time.time()
            epoch_loss = running_loss / dataset_sizes[phase]
            logging.info('Epoch {},: loss {}, {},time {}'.format(epoch+1,  epoch_loss,phase,e-s))

            if phase == 'train':
                train_loss.append(epoch_loss)
            else:
                val_loss.append(epoch_loss)
            
            if phase == 'test' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_epoch = epoch+1
                best_model_wts = copy.deepcopy(model.state_dict())
                logging.info("Best val loss {} save at epoch {}".format(best_loss,epoch+1))
    draw_loss(train_loss,val_loss)
    model_save_path = os.path.join(save_model_path,"{}_{}_epoch:{}_loss:{}_{}.pt".format(model_type,dataset,best_epoch,best_loss,str(now)))

    torch.save(best_model_wts,model_save_path)
    logging.info("Model saved at {}".format(model_save_path))

    dice_acc,acc,Iou,F1,aji,aji_2,ajip,ajip_2,pq,pq_2 = test(testloader,model_type,model_save_path,channel,device,sharing_ratio)
    logging.info("Dice acc {}".format(dice_acc/len(testloader)))
    logging.info("acc {}".format(acc/len(testloader)))
    logging.info("IOU {}".format(Iou/len(testloader)))
    logging.info("IOU sum {}".format(Iou))
    logging.info("F1 {}".format(F1/len(testloader)))
    logging.info("Aji {}".format(aji/len(testloader)))
    logging.info("Aji2 {}".format(aji_2/len(testloader)))
    logging.info("Ajip {}".format(ajip/len(testloader)))
    logging.info("Ajip_2 {}".format(ajip_2/len(testloader)))
    logging.info("pq {}".format(pq/len(testloader)))
    logging.info("pq2 {}".format(pq_2/len(testloader)))


