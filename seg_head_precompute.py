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
        x = x.squeeze(1)
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
        # x3 = self.decode3(x2) 
        # x4 = self.decode4(x3) #+ x3
        # x4 = F.sigmoid(x4)
        return x2
def loss(pred, gt):

    if torch.mean(gt) > 0.0001:
        weight = 1 / (torch.mean(gt))
        pos_wei = torch.ones_like(gt) * weight
        # print('stuff')
            
            
        BCE_fun = nn.BCEWithLogitsLoss(pos_weight=pos_wei)
    else:
        BCE_fun = nn.BCEWithLogitsLoss()
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
        show_mask_every = 999, 
        val_mask_dir=None,
        dir_checkpoint=None,):
    
    # ipdb.set_trace()
    coco_set = Coco_Dataset_Embeddings(img_dir='data/small/train',
                           anno_file='data/small/train/_annotations.coco.json')
    # ipdb.set_trace()
    
    data_loader = DataLoader(coco_set)
    
    # ipdb.set_trace()


    optimizer = optim.Adam(decoder.parameters(), lr = learning_rate)
    
    step = 0

    


    for epo in range(epochs):
        for image, mask in data_loader:
            optimizer.zero_grad()
            # ipdb.set_trace()
            step += 1
            image = image.cuda()
            mask = mask.cuda()
            # ipdb.set_trace()
            # enc_out = encoder.sample_frame(batch_size, image)
            enc_out = image
            dec_out = decoder(enc_out)
            #change the image mask
            loss_val = loss(pred= dec_out, gt = mask)

            

            loss_val.backward()

            optimizer.step()

            if step % val_frequency == 0:
                print("loss at step ", step, " :" , loss_val.cpu().detach().numpy())

            #save model 
            # if step % save_checkpoint_every == 0:
            #     torch.save(encoder.state_dict(), dir_checkpoint)

            #display results
            if step % show_mask_every == 0:
                thresh = 0.5
                # ipdb.set_trace()
                image_np = F.sigmoid(dec_out).cpu().detach().numpy()
                out_mask = image_np[0, 0]
                # out_mask = np.zeros((64, 64))
                # out_mask[F.sigmoid(dec_out).cpu()[0, 0] > thresh] = 1
                
                # image_np = torch.where(dec_out).cpu().detach().numpy()

                # # If your tensor has a batch dimension, remove it
                # if len(image_np.shape) == 4:
                #     image_np = image_np.squeeze(0)

                # If your image is in channel-first format, transpose it to channel-last format (optional)
                #if image_np.shape[0] == 3:
                #    image_np = image_np.transpose(1, 2, 0)
                # ipdb.set_trace()
                # ipdb.set_trace()
                channel_1 = Image.fromarray(mask[0].cpu().numpy()*255)
                channel_2 = Image.fromarray(out_mask*255)
                name1 = "channel_1_" + str(step) + ".png"
                name2 = "channel_2_" + str(step) +".png"
                channel_1.convert("RGB").save(name1)
                channel_2.convert("RGB").save(name2)
            
                


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
    



    


