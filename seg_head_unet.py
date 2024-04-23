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
from unet import UNet

import segmentation_models_pytorch.losses as Loss




def loss(pred, gt):

   
    weight = 1 / (torch.mean(gt))
    pos_wei = torch.ones_like(gt) * weight
    BCE_fun = nn.BCEWithLogitsLoss(pos_weight=pos_wei)
    # ipdb.set_trace()
    loss = BCE_fun(pred, gt) #+ dice_fun(pred, gt)
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
    coco_set = Coco_Dataset(img_dir='data/small/train',
                           anno_file='data/small/train/_annotations.coco.json')
    # ipdb.set_trace()
    
    data_loader = DataLoader(coco_set)
    
    # ipdb.set_trace()

    # enc = segenc(n_classes=2).cuda()
    # opt_params = 
    optimizer = optim.Adam(list(decoder.parameters()), lr = learning_rate)
    
    step = 0

    


    for epo in range(epochs):
        for image, mask in data_loader:
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
            # print(loss_val)
            if step % val_frequency == 0:
                print("loss at step ", step, " :" , loss_val.cpu().detach().numpy())

            # #save model 
            # # if step % save_checkpoint_every == 0:
            # #     torch.save(encoder.state_dict(), dir_checkpoint)

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
                channel_1 = Image.fromarray(mask[0, 0].cpu().numpy()*255)
                channel_2 = Image.fromarray(out_mask*255)
                name1 = "channel_1_" + str(step) + ".png"
                name2 = "channel_2_" + str(step) +".png"
                channel_1.convert("RGB").save(name1)
                channel_2.convert("RGB").save(name2)
            
                


if __name__ == "__main__":
     
    decoder = UNet().cuda()
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
    



    


