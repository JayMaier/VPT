# https://stackoverflow.com/questions/50805634/how-to-create-mask-images-from-coco-dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt
import wandb
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from videogpt import VideoData, VideoGPT, load_videogpt
from videogpt.utils import save_video_grid
import argparse
import torchvision
import ipdb
import pycocotools
from datasets import *
from unet import UNet

import segmentation_models_pytorch.losses as Loss
# import segmentation_models_pytorch.losses as Loss
import segmentation_models_pytorch as smp

class magic(nn.Module):
    def __init__(self):
        super(magic, self).__init__()
        self.mlp = nn.Linear(4*32*32, 1*64*64)
        self.unet = UNet(in_channels=1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = x.squeeze(0)
        x = x.type(torch.cuda.FloatTensor).cuda()
        
        x = self.mlp(x.flatten()).view(1, 1, 64, 64)
        x = self.relu(x)
        out = self.unet(x)
        return out
        
        

def loss(pred, gt):

   
    weight = 1 / (torch.mean(gt))
    pos_wei = torch.ones_like(gt) * weight
    BCE_fun = nn.BCEWithLogitsLoss(pos_weight=pos_wei)
    # ipdb.set_trace()
    loss = BCE_fun(pred[0], gt) #+ dice_fun(pred, gt)
    return loss


def train(
        decoder,   
        device,
        epochs = 1000,
        batch_size = 1,
        learning_rate = 1e-5,
        val_frequency =  10,
        save_checkpoint_every = 10000,
        weight_decay: float = 9,
        gradient_clipping: float = 1.0,
        train_images_dir=None,
        train_mask_dir=None,
        val_images_dir=None,
        show_mask_every = 1, 
        val_mask_dir=None,
        dir_checkpoint=None,):
    
    # ipdb.set_trace()
    coco_set = Coco_Dataset_Embeddings(img_dir='data/small/train',
                           anno_file='data/small/train/_annotations.coco.json')
    # ipdb.set_trace()
    
    # data_loader = DataLoader(coco_set)
    
    train_size = int(0.8 * len(coco_set))
    val_size = int(len(coco_set) - train_size)
    train_coco, val_coco = torch.utils.data.random_split(coco_set, [train_size, val_size])
    
    train_data_loader = DataLoader(train_coco, batch_size= batch_size)
    val_data_loader = DataLoader(val_coco, batch_size=1)
    
    # ipdb.set_trace()

    # enc = segenc(n_classes=2).cuda()
    # opt_params = 
    optimizer = optim.Adam(list(decoder.parameters()), lr = learning_rate)
    
    step = 0

    
    step, step_val = 0, 0

    for epo in range(epochs):
        train_loss = 0
        for image, mask in train_data_loader:
            step += 1
            optimizer.zero_grad()
            # ipdb.set_trace()
            step += 1
            image = image.cuda()
            mask = mask.cuda()
            # enc_out = encoder.sample_frame(batch_size, image)
            # enc_out = enc(image)
            dec_out = decoder(image)
            #change the image mask
            loss_val = loss(pred= dec_out, gt = mask)

            

            loss_val.backward()

            optimizer.step()
            train_loss += loss_val.cpu().detach().numpy()/len(train_data_loader)
            # print(loss_val)
            if step % val_frequency == 0:
                print("loss at step ", step, " :" , loss_val.cpu().detach().numpy())

            # #save model 
            # # if step % save_checkpoint_every == 0:
            # #     torch.save(encoder.state_dict(), dir_checkpoint)

            #display results
            # if step % show_mask_every == 0:
            #     thresh = 0.5
            #     # ipdb.set_trace()
            #     image_np = F.sigmoid(dec_out).cpu().detach().numpy()
            #     out_mask = image_np[0, 0]
            #     # out_mask = np.zeros((64, 64))
            #     # out_mask[F.sigmoid(dec_out).cpu()[0, 0] > thresh] = 1
                
            #     # image_np = torch.where(dec_out).cpu().detach().numpy()

            #     # # If your tensor has a batch dimension, remove it
            #     # if len(image_np.shape) == 4:
            #     #     image_np = image_np.squeeze(0)

            #     # If your image is in channel-first format, transpose it to channel-last format (optional)
            #     #if image_np.shape[0] == 3:
            #     #    image_np = image_np.transpose(1, 2, 0)
            #     # ipdb.set_trace()
            #     # ipdb.set_trace()
            #     channel_1 = Image.fromarray(mask[0, 0].cpu().numpy()*255)
            #     channel_2 = Image.fromarray(out_mask*255)
            #     name1 = "channel_1_" + str(step) + ".png"
            #     name2 = "channel_2_" + str(step) +".png"
            #     channel_1.convert("RGB").save(name1)
            #     channel_2.convert("RGB").save(name2)
        iou_scores= 0
        for image, mask in val_data_loader:
            step_val +=1 
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

                wandb.log({"val actual masks ": actual_masks,
                        "val pred masks ": pred_masks})# ipdb.set_trace()
        wandb.log({"iou score val": iou_scores, "train loss":train_loss})
            
def obj(config):
    decoder = magic().cuda()
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
     
    decoder = magic().cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
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
            'values': ["unet from latent"]
        },
    }



    
    parameters_dict.update({
        "learning_rate":{
            "distribution": "uniform",
            "min" : 1e-6,
            "max": 1e-4
        }
    })

    sweep_config["parameters"] = parameters_dict

    
    # pprint.pprint(sweep_config)

    sweep_id = wandb.sweep(sweep_config, project = project)

    wandb.agent(sweep_id, function=test, count=100)

