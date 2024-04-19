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





class segenc(nn.Module):
    def __init__(self, n_classes):
        super(segenc, self).__init__()
        
        

        self.latent = [1, 4, 32, 32]
        self.classes = n_classes
        
        # self.conv_block = nn.Sequential(
        #     nn.Conv2d(3, 8, 3, stride=2, padding=1),
        #     nn.ReLU(inplace=True)
        # )
        
        self.b1 = self.get_conv_block(in_chan = 3, out_chan = 32, stride=1, pool_size=3, pool_stride=2)  
        self.b2 = self.get_conv_block(in_chan = 32, out_chan = 64, stride=1, pool_size=3, pool_stride=1)        
        self.b3 = self.get_conv_block(in_chan = 64, out_chan = 128, stride=1, pool_size=3, pool_stride=1)        
        self.b4 = self.get_conv_block(in_chan = 128, out_chan = 256, stride=1, pool_size=3, pool_stride=1)        
        self.b5 = self.get_conv_block(in_chan = 256, out_chan = 256, stride=1, pool_size=3, pool_stride=1)
        
        
        self.downsamp1 = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.resblock1 = nn.Sequential(
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.create_embedding = nn.Sequential(
            nn.Conv2d(256, 4, 3, padding=1),
            nn.ReLU(inplace=True)
        )

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
        
    def get_conv_block(self, in_chan, out_chan, stride, pool_size, pool_stride):
            block = nn.Sequential(
                nn.Conv2d(in_chan, out_chan, 3, stride=stride, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_chan, out_chan, 3,  stride=stride, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(pool_size, stride=pool_stride, padding=1)
            )
            return block    

    def forward(self, x):
        #or squueze at 1
        # # ipdb.set_trace()
        # x = x.squeeze(1)
        # x = x.type(torch.cuda.FloatTensor).cuda()
        
        # # ipdb.set_trace()
        # # thing = x.flatten()
        # # newthing = self.fc(thing)
        # # x4 = newthing.view(1, 2, 64, 64)
        
        # x = self.downsamp1(x)
        
        # x1 = self.resblock1(x) + x
        # x2 = self.resblock1(x1) + x1
        # x2 = self.create_embeding(x2)
        # ipdb.set_trace()
        x = self.b1(x)
        x = self.b2(x) 
        x = self.b3(x) 
        x = self.b4(x) 
        # x = self.b5(x) 
        x = self.create_embedding(x)
        
        
        
        # x = self.fc(x.flatten()).view(1, 4, 32, 32)
        # x = self.relu(x)
        
        # x1 = self.decode1(x)
        # # self.fc(x)
        
        # x2 = self.decode2(x1) #+ x1
        # x3 = self.decode3(x2) 
        # x4 = self.decode4(x3) #+ x3
        # x4 = F.sigmoid(x4)
        return x

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

    enc = segenc(n_classes=2).cuda()
    # opt_params = 
    optimizer = optim.Adam(list(decoder.parameters()) + list(enc.parameters()), lr = learning_rate)
    
    step = 0

    


    for epo in range(epochs):
        for image, mask in data_loader:
            optimizer.zero_grad()
            # ipdb.set_trace()
            step += 1
            image = image.cuda()
            mask = mask.cuda()
            # enc_out = encoder.sample_frame(batch_size, image)
            enc_out = enc(image)
            dec_out = decoder(enc_out)
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
    



    


