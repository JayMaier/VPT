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

class seghead(nn.Module):
    def __init__(self, n_classes):
        super(seghead, self).__init__()

        self.latent = [1, 4, 32, 32]
        self.classes = n_classes

        self.decode1 = nn.Sequential(
            nn.ConvTranspose2d(self.latent[1], self.classes, 2, 2, padding = 0),
            nn.BatchNorm2d(self.classes),
            nn.ReLU(inplace= True)
        )

        self.decode2 = nn.Sequential(
            nn.ConvTranspose2d(4, 4, 3, 1, padding = 1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace= True)
        )

        self.decode3 = nn.Sequential(
            nn.ConvTranspose2d(4, 4, 1, 1, padding = 0),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace= True)
        )

        self.decode4 = nn.Sequential(
            nn.ConvTranspose2d(4, 4, 3, 1, padding = 1), 
            nn.BatchNorm2d(self.classes),
            nn.ReLU(inplace= True)
        )

    def forward(self, x):
        x = x.type(torch.cuda.FloatTensor).cuda()
        y = self.decode1(x)
        #y = self.decode2(y)
        #y = self.decode3(y)
        #y = self.decode4(y)
        return y

def dice_coeff(input, target, reduce_batch_first= False, epsilon= 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input, target, reduce_batch_first = False, epsilon = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input, target, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)
    
def loss(pred, gt, n_classes):

    # if n_classes == 1:
    #     crit_2 = dice_loss(F.sigmoid(pred.squeeze(1)), gt.float(), multiclass=False)
    # else:
    #     crit_2 = dice_loss(
    #         F.softmax(pred, dim=1).float(),
    #         F.one_hot(gt, n_classes).permute(0, 3, 1, 2).float(),
    #         multiclass=True
    #     )
    weight = 1 / (torch.mean(gt))
    pos_weights = torch.full((n_classes,), weight)
    #ipdb.set_trace()
    crit_1 = nn.BCEWithLogitsLoss(pos_weight= pos_weights)
    loss = crit_1(pred, gt.cuda())# + crit_2
    return loss


# class Coco_Dataset(Dataset):
#     def __init__(self, img_dir, anno_file, transform=None):
#         self.img_dir = img_dir
#         self.anno_file = anno_file
#         self.coco = pycocotools.coco.COCO(anno_file)
#         self.cat_ids = self.coco.getCatIds()
#         self.im2float = torchvision.transforms.ConvertImageDtype(torch.float32)
        
#     def __len__(self):
#         return len(self.coco.imgs)
    
#     def __getitem__(self, index):
#         index = 0
#         img = self.coco.imgs[index]
#         # image = torch.tensor(np.array(Image.open(os.path.join(self.img_dir, img['file_name']))))
#         image = torchvision.io.read_image(os.path.join(self.img_dir, img['file_name']))
#         image = self.im2float(image)
        
#         anns_ids_1 = self.coco.getAnnIds(imgIds=img['id'], catIds=1, iscrowd=None)
#         anns_1 = self.coco.loadAnns(anns_ids_1)
        

#         anns_ids_2 = self.coco.getAnnIds(imgIds=img['id'], catIds=2, iscrowd=None)
#         anns_2 = self.coco.loadAnns(anns_ids_2)
#         ### TODO: DANGER!!! do we get every class every time ? ###
#         mask = np.zeros((len(self.cat_ids)-1, image.shape[1], image.shape[2]))
#         # mask = self.coco.annToMask(anns[0])
#         # print('cat ids', self.cat_ids)
#         # ipdb.set_trace()
        
#         for i in range(0, len(anns_1)):
#             mask[0] += self.coco.annToMask(anns_1[i])

#         for i in range(0, len(anns_2)):
#             mask[1] += self.coco.annToMask(anns_2[i])
        
#         mask = torch.tensor(mask)
#         mask = mask.clamp(0, 1)
#         return image, mask

        
        


transform = transforms.Compose([
    transforms.CenterCrop((64, 64)),
    transforms.Resize((64, 64)),  
    transforms.ToTensor(),           
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
])

def train(encoder,
        decoder,   
        device,
        n_classes,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_frequency: float = 0.5,
        save_checkpoint_every = 10000,
        weight_decay: float = 9,
        gradient_clipping: float = 1.0,
        train_images_dir=None,
        train_mask_dir=None,
        val_images_dir=None,
        show_mask_every = 1, 
        val_mask_dir=None,
        dir_checkpoint=None,
        ):
    
    
    # dataset = SegmentationDataset(root_dir='path/to/dataset', transform=transform)
    
    train_set = torchvision.datasets.CocoDetection(root = 'data/mini/train',
                                                   annFile= 'data/mini/train/_annotations.coco.json')
    # ipdb.set_trace()
    coco_set = Coco_Dataset_Embeddings(img_dir='data/mini/train',
                           anno_file='data/mini/train/_annotations.coco.json')
    # ipdb.set_trace()
    
    data_loader = DataLoader(coco_set, batch_size=1, shuffle=True, num_workers=4)
    
    # ipdb.set_trace()

    for params in encoder.parameters():
        params.requires_grad = False

    optimizer = optim.Adam(decoder.parameters(), lr = learning_rate, weight_decay= weight_decay)
    batches = len(data_loader)
    step = 0

    for epo in range(epochs):
        for image, mask in data_loader:
            step += 1
            image = image.cuda()
            mask = mask.cuda()
            enc_out = encoder.sample_frame(batch_size, image)
            dec_out = decoder(enc_out)
            #change the image mask
            loss_val = loss(dec_out, mask, n_classes)

            optimizer.zero_grad()

            loss_val.backward()

            optimizer.step()

            if step % val_frequency == 0:
                print("loss at step ", step, " :" , loss_val.cpu().detach().numpy())

            #save model 
            if step % save_checkpoint_every == 0:
                torch.save(encoder.state_dict(), dir_checkpoint)

            #display results
            if step % show_mask_every == 0:
                image_np = dec_out.cpu().detach().numpy()

                # If your tensor has a batch dimension, remove it
                if len(image_np.shape) == 4:
                    image_np = image_np.squeeze(0)

                # If your image is in channel-first format, transpose it to channel-last format (optional)
                #if image_np.shape[0] == 3:
                #    image_np = image_np.transpose(1, 2, 0)
                #ipdb.set_trace()
                channel_1 = Image.fromarray(image_np[0])
                channel_2 = Image.fromarray(image_np[1])
                name1 = "channel_1_" + str(step) + ".png"
                name2 = "channel_2_" + str(step) +".png"
                channel_1.convert("RGB").save(name1)
                channel_2.convert("RGB").save(name2)
                


if __name__ == "__main__":
    n_classes = 2
    decoder = seghead(n_classes= n_classes).cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='bair_gpt')
    parser.add_argument('--n', type=int, default=1)
    args = parser.parse_args()
    n = args.n

    if not os.path.exists(args.ckpt):
        gpt = load_videogpt(args.ckpt)
    else:
        gpt = VideoGPT.load_from_checkpoint(args.ckpt)
    gpt = gpt.cuda()
    gpt.eval()
    
    train(gpt,
        decoder,   
        device,
        n_classes = n_classes
        epochs = 5,
        batch_size = 1,
        learning_rate = 1e-5,
        val_frequency = 0.5,
        )
    



    


