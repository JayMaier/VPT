# https://stackoverflow.com/questions/50805634/how-to-create-mask-images-from-coco-dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt
import wandb
import torch.optim.lr_scheduler as schedule

import os
import numpy as np
from PIL import Image
import torch.utils
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pprint

from videogpt import VideoData, VideoGPT, load_videogpt
from videogpt.utils import save_video_grid
import argparse
import torchvision
import ipdb
import pycocotools
from datasets import *

import segmentation_models_pytorch.losses as Loss
import segmentation_models_pytorch as smp


class seghead(nn.Module):
    def __init__(self, n_classes, dropout):
        super(seghead, self).__init__()

        self.latent = [1, 4, 32, 32]
        self.classes = n_classes
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.decode1 = nn.Sequential(
            nn.ConvTranspose2d(self.latent[1], 8, 2, 2, padding = 0),
            # nn.BatchNorm2d(self.classes),
            nn.ReLU(inplace= True)
        )

        self.decode2 = nn.Sequential(
            nn.Conv2d(8, self.classes, 3, 1, padding = 1),
            # nn.BatchNorm2d(self.classes),
            nn.ReLU(inplace= True)
        )

        self.decode3 = nn.Sequential(
            nn.ConvTranspose2d(self.classes, self.classes, 1, 1, padding = 0),
            # nn.BatchNorm2d(self.classes),
            nn.ReLU(inplace= True)
        )

        self.decode4 = nn.Sequential(
            nn.ConvTranspose2d(self.classes, self.classes, 3, 1, padding = 1), 
            # nn.BatchNorm2d(self.classes),
            nn.ReLU(inplace= True)
        )
        
        # self.squash = nn.Sequential(
        #     nn.Conv2d(1, )
        # )
        self.fc = nn.Linear(4*32*32, 4*32*32)
        self.relu = nn.ReLU()
        

    def forward(self, x):
        #or squueze at 1
        # ipdb.set_trace()
        #x = x.squeeze(0)
        x = x.squeeze(1)
        bs= x.shape[0]
        x = x.type(torch.cuda.FloatTensor).cuda()
        
        # ipdb.set_trace()
        # thing = x.flatten()
        # newthing = self.fc(thing)
        # x4 = newthing.view(1, 2, 64, 64)
        x = self.dropout1(self.fc(x.view(bs, -1)))
        x = x.view(bs, 4, 32, 32)
        x = self.relu(x)
        
        x1 = self.decode1(x)
        x2 = self.dropout2(self.decode2(x1)) #+ x1
        x3 = self.decode3(x2) + x2
        x4 = self.dropout3(self.decode4(x3)) #+ x3
        # x4 = F.sigmoid(x4)

        return x2


def dice(pred, gt):
    dce = Loss.DiceLoss(mode = "binary")
    return dce(pred, gt)

def bce(pred, gt):
    if torch.mean(gt) > 0.0001:
        weight = 1 / (torch.mean(gt))
        pos_wei = torch.ones_like(gt) * weight
        BCE_fun = nn.BCEWithLogitsLoss(pos_weight=pos_wei)
    else:
        BCE_fun = nn.BCEWithLogitsLoss()
    
    loss = BCE_fun(pred.squeeze(1), gt) 
    return loss

def jacard(pred, gt):
    jac = Loss.JaccardLoss(mode = "binary")
    return jac(pred, gt)



def loss(pred, gt, mode):
    if mode == "b":
        return bce(pred, gt)
    elif mode == "d":
        return dice(pred, gt)
    elif mode == "j":
        return jacard(pred, gt)
    elif mode == "bd":
        return bce(pred, gt) + dice(pred, gt)
    elif mode == "bj":
        return bce(pred, gt) + jacard(pred, gt)
    elif mode == "dj":
        return dice(pred, gt) + jacard(pred, gt)
    elif mode == "bdj":
        return bce(pred, gt) + jacard(pred, gt) + dice(pred, gt)
    else:
        assert 1, "enter a mode for loss"



def train(
        decoder,   
        device,
        wandb_freq = None,
        epochs = None,
        batch_size = 1,
        learning_rate = None,
        save_checkpoint_every = 10000,
        weight_decay: float = 9,
        gradient_clipping: float = 1.0,
        train_images_dir=None,
        train_mask_dir=None,
        val_images_dir=None,
        show_mask_every = 100, 
        val_mask_dir=None,
        dir_checkpoint="data/decoder_model.pt",
        WDB = None,
        loss_mode = None,
        lr_sced = None,
        opt = "ADAM"):
    
    coco_set = Coco_Dataset_Embeddings(img_dir='data/small/train',
                           anno_file='data/small/train/_annotations.coco.json')
    # ipdb.set_trace()
    

    train_size = int(0.8 * len(coco_set))
    val_size = int(len(coco_set) - train_size)
    train_coco, val_coco = torch.utils.data.random_split(coco_set, [train_size, val_size])
    
    train_data_loader = DataLoader(train_coco, batch_size= batch_size)
    val_data_loader = DataLoader(val_coco, batch_size=1)
    if opt == "ADAM":
        optimizer = optim.Adam(decoder.parameters(), lr = learning_rate)
    if opt == "ADAM_W":
        optimizer = optim.AdamW(decoder.parameters(), lr = learning_rate)
    if opt == "SGD":
        optimizer = optim.SGD(decoder.parameters(), lr = learning_rate)

    if lr_sced:
        lr_scedule = schedule.ExponentialLR(optimizer= optimizer, gamma= lr_sced)

    step, step_val = 0, 0

    iou_scores = []
    for epo in range(epochs):
        decoder.train()
        train_loss = 0
        for image, mask in train_data_loader:
            step += 1
            image = image.to(device)
            mask = mask.to(device)
            enc_out = image
            dec_out = decoder(enc_out)
            loss_val = loss(pred= dec_out, gt = mask, mode= loss_mode)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            train_loss += loss_val.cpu().detach().numpy()/len(train_data_loader)
            if epo == epochs - 1:
                        print('hi')

        decoder.eval()
        iou_scores = 0
        for image, mask in val_data_loader:
            image = image.to(device)
            mask = mask.to(device)
            enc_out = image
            dec_out = decoder(enc_out)           
            tp, fp , fn , tn = smp.metrics.get_stats(output= dec_out.squeeze(1), target= mask.round().int(), mode = "binary", threshold = 0.5)
            iou_scores += smp.metrics.iou_score(tp, fp, fn, tn, reduction= "micro")/len(val_data_loader)
            if epo % show_mask_every == 0:
                image_np = torch.sigmoid(dec_out).cpu().detach().numpy()
                #ipdb.set_trace()
                if len(image_np.shape) == 4:
                    image_np = image_np[0, 0, :, :]
                if len(image_np.shape) == 3:
                    image_np = image_np[0, :, :]
                channel_1 = Image.fromarray(mask[0].cpu().numpy()*255)
                channel_2 = Image.fromarray(image_np*255)
                name1 = "mask " + str(step_val)
                name2 = "pred_mask" + str(step_val)
                actual_masks = wandb.Image(channel_1.convert("RGB"), caption = name2)
                pred_masks = wandb.Image(channel_2.convert("RGB"), caption = name1)

                wandb.log({"train actual masks ": actual_masks,
                        "train pred masks ": pred_masks})# ipdb.set_trace()
        wandb.log({"iou score val": iou_scores, "train loss":train_loss})
        print("loss at epo ", epo, " : ",  train_loss)

        if lr_sced:
            lr_scedule.step()
        # ipdb.set_trace()
        
             
    return iou_scores

def obj(config):
    decoder = seghead(n_classes= 1, dropout=config.dropout).cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ipdb.set_trace()
    val_loss = train(
        decoder = decoder,   
        device = device,
        # epochs = config.epochs,
        # batch_size = config.batch_size,
        learning_rate = config.learning_rate,
        # WDB=run,
        # loss_mode = config.loss_mode,
        # lr_sced= config.lr_sced,
        # opt = config.opt
        )
    return val_loss      
def test():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = wandb.init(project="VPT")
    # wandb.init(wandb.config)

    score = obj(wandb.config)
    wandb.log({'score':score})


if __name__ == "__main__":
     

    project = "VPT"

    sweep_config = {
        "method" : "random"
    }
    
    metric = {
        "name" : "loss",
        "goal" : "minimize"
    }

    sweep_config["metric"] = metric

    parameters_dict ={
        "opt":{
            'values': ["ADAM"]
        },
        "loss_mode":{
            'values': ["b", "d", "j"]
        },
        "lr_sced":{
            "values": [0.999]

        },
        'dropout':{
            'values': [0.25]
        },
        'epochs':{
            "values": [3000]
        },
        "batch_size":{
            "values": [1, 2, 8]
        }
    }

    # parameters_dict ={
    #     "opt":{
    #         'values': ["ADAM", "ADAM_W", "SGD"]
    #     },
    #     "loss_mode":{
    #         'values': ["b", "d", "j", "bd", "bj", "dj", "bdj"]
    #     },
    #     "lr_sced":{
    #         "values": [0.999, 0.99, 0.9]

    #     },
    #     'dropout':{
    #         'values': [0.0, 0.25, 0.5]
    #     },
    #     'epochs':{
    #         "values": [300]
    #     },
    #     "batch_size":{
    #         "values": [1, 2, 8, 32]
    #     }
    # }
    parameters_dict.update({
        "learning_rate":{
            "distribution": "uniform",
            "min" : 1e-6,
            "max": 1e-4
        }
    })

    sweep_config["parameters"] = parameters_dict

    
    pprint.pprint(sweep_config)

    sweep_id = wandb.sweep(sweep_config, project = project)

    wandb.agent(sweep_id, function=test, count=100)
 


    