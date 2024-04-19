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
import torch.utils
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from videogpt import VideoData, VideoGPT, load_videogpt
from videogpt.utils import save_video_grid
import argparse
import torchvision
import ipdb
import pycocotools
from datasets import *

import segmentation_models_pytorch.losses as Loss


class seghead(nn.Module):
    def __init__(self, n_classes):
        super(seghead, self).__init__()

        self.latent = [1, 4, 32, 32]
        self.classes = n_classes

        self.decode1 = nn.Sequential(
            nn.ConvTranspose2d(self.latent[1], 128, 2, 2, padding = 0),
            # nn.BatchNorm2d(self.classes),
            nn.ReLU(inplace= True)
        )

        self.decode2 = nn.Sequential(
            nn.Conv2d(128, self.classes, 3, 1, padding = 1),
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
        x = x.squeeze(0)
        x = x.type(torch.cuda.FloatTensor).cuda()
        
        # ipdb.set_trace()
        # thing = x.flatten()
        # newthing = self.fc(thing)
        # x4 = newthing.view(1, 2, 64, 64)
        x = self.fc(x.flatten()).view(1, 4, 32, 32)
        x = self.relu(x)
        
        x1 = self.decode1(x)
        # self.fc(x)
        
        x2 = self.decode2(x1) #+ x1
        # x3 = self.decode3(x2) #+ x2
        # x4 = self.decode4(x3) #+ x3
        # x4 = F.sigmoid(x4)
        return x2


def dice():
    Loss.DiceLoss(mode = "binary")

def bce():

def jacard():

def bce_dice():

def bce_jacard():

def dice_jacard():

def bce_jacard_dice():









def loss(pred, gt):
    weight = 1 / (torch.mean(gt))
    pos_wei = torch.ones_like(gt) * weight
    BCE_fun = nn.BCEWithLogitsLoss()
    loss = BCE_fun(pred[0, 0], gt[0]) #+ dice_fun(pred, gt)
    return loss








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
        show_mask_every = 999, 
        val_mask_dir=None,
        dir_checkpoint="data/decoder_model.pt",
        WDB = None,):
    
    coco_set = Coco_Dataset_Embeddings(img_dir='data/mini/train',
                           anno_file='data/mini/train/_annotations.coco.json')
    

    train_size = int(0.8 * len(coco_set))
    val_size = int(len(coco_set) - train_size)
    train_coco, val_coco = torch.utils.data.random_split(coco_set, [train_size, val_size])
    
    train_data_loader = DataLoader(train_coco, batch_size= batch_size)
    val_data_loader = DataLoader(val_coco, batch_size= batch_size)


    optimizer = optim.Adam(decoder.parameters(), lr = learning_rate)
    step, step_val = 0, 0



    for epo in range(epochs):
        decoder.train()
        train_loss = 0
        for image, mask in train_data_loader:
            step += 1
            image = image.to(device)
            mask = mask.to(device)
            enc_out = image
            dec_out = decoder(enc_out)
            #change the image mask
            loss_val = loss(pred= dec_out, gt = mask)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            train_loss += loss_val.cpu().detach().numpy()
            if step % save_checkpoint_every == 0:
                torch.save(decoder.state_dict(), dir_checkpoint + str(step))

        print("loss at epo ", epo, " : ",  train_loss)

        if WDB:
            WDB.log({"train_loss" : train_loss})
            if epo % show_mask_every == 0:
                image_np = torch.sigmoid(dec_out).cpu().detach().numpy()
                if len(image_np.shape) == 4:
                    image_np = image_np.squeeze(0)
                channel_1 = Image.fromarray(mask[0].cpu().numpy()*255)
                channel_2 = Image.fromarray(image_np[0]*255)
                name1 = "mask " + str(step_val)
                name2 = "pred_mask" + str(step_val)
                actual_masks = wandb.Image(channel_1.convert("RGB"), caption = name2)
                pred_masks = wandb.Image(channel_2.convert("RGB"), caption = name1)

                WDB.log({"train actual masks ": actual_masks,
                        "train pred masks ": pred_masks})


        if WDB and epo % wandb_freq == 0 or epo % show_mask_every == 0:
            decoder.eval()
            loss_epo = 0 
            for image, mask in val_data_loader:
                step_val += 1
                image = image.cuda()
                mask = mask.cuda()
                enc_out = image
                dec_out = decoder(enc_out)
                loss_val = loss(pred= dec_out, gt = mask)
                loss_epo += loss_val.item()

            image_np = torch.sigmoid(dec_out).cpu().detach().numpy()
            # If your tensor has a batch dimension, remove it
            if len(image_np.shape) == 4:
                image_np = image_np.squeeze(0)
            channel_1 = Image.fromarray(mask[0].cpu().numpy()*255)
            channel_2 = Image.fromarray(image_np[0]*255)
            name1 = "mask " + str(step_val)
            name2 = "pred_mask" + str(step_val)
            actual_masks = wandb.Image(channel_1.convert("RGB"), caption = name2)
            pred_masks = wandb.Image(channel_2.convert("RGB"), caption = name1)

            WDB.log({"validation actual masks ": actual_masks,
                     "validation pred masks ": pred_masks,
                     "validation loss": loss_epo})


if __name__ == "__main__":
     
    decoder = seghead(n_classes= 1).cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='bair_gpt')
    parser.add_argument('--n', type=int, default=1)
    args = parser.parse_args()
    #n = args.
    epochs = 5000
    batch_size = 1
    learning_rate = 1e-3

    run = wandb.init(
                project = "VPT",
                config={
                    "epochs" : epochs,
                    "batch size" : batch_size,
                    "learning rate" : learning_rate
                }
            )
        
    train(
        decoder,   
        device,
        epochs = epochs,
        batch_size = batch_size,
        learning_rate = learning_rate,
        wandb_freq = 100,
        WDB = run,
        )
    



    


