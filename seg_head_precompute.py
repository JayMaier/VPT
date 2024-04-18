# https://stackoverflow.com/questions/50805634/how-to-create-mask-images-from-coco-dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt

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
        x = x.squeeze(1)
        x = x.type(torch.cuda.FloatTensor).cuda()
        
        x = self.fc(x.flatten()).view(1, 4, 32, 32)
        x = self.relu(x)
        
        x1 = self.decode1(x)
        
        x2 = self.decode2(x1) 

        return x2

def loss(pred, gt):

   
    weight = 1 / (torch.mean(gt))
    pos_wei = torch.ones_like(gt) * weight
    BCE_fun = nn.BCEWithLogitsLoss()
    
    loss = BCE_fun(pred[0, 0], gt[0])
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
        show_mask_every = 999, 
        val_mask_dir=None,
        dir_checkpoint=None,):
    
    
    coco_set = Coco_Dataset_Embeddings(img_dir='data/mini/train',
                           anno_file='data/mini/train/_annotations.coco.json')
    
    data_loader = DataLoader(coco_set)

    optimizer = optim.Adam(decoder.parameters(), lr = learning_rate)
    
    step = 0

    


    for epo in range(epochs):
        for image, mask in data_loader:

            # ipdb.set_trace()
            step += 1
            image = image.cuda()
            mask = mask.cuda()

            enc_out = image
            dec_out = decoder(enc_out)
            #change the image mask
            loss_val = loss(pred= dec_out, gt = mask)

            optimizer.zero_grad()

            loss_val.backward()

            optimizer.step()

            if step % val_frequency == 0:
                print("loss at step ", step, " :" , loss_val.cpu().detach().numpy())



if __name__ == "__main__":
     
    decoder = seghead(n_classes= 1).cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='bair_gpt')
    parser.add_argument('--n', type=int, default=1)
    args = parser.parse_args()
    n = args.n


    
    train(
        decoder,   
        device,
        epochs = 5000,
        batch_size = 1,
        learning_rate = 1e-5,
        )
    



    


