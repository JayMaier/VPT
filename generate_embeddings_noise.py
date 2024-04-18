import os
import argparse
import torch
import torchvision

from videogpt import VideoData, VideoGPT, load_videogpt
from videogpt.utils import save_video_grid
import ipdb
from datasets import *
from torch.utils.data import Dataset, DataLoader



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

img_dir='data/small/train'

coco_set = Coco_Dataset_embedder(img_dir=img_dir,
                           anno_file='data/small/train/_annotations.coco.json')
    # ipdb.set_trace()
    
data_loader = DataLoader(coco_set, batch_size=1, shuffle=True, num_workers=4)

for img, msk, pth in data_loader:

    # enc_out = gpt.sample_frame(1, img)
    # ipdb.set_trace()
    enc_out = torch.rand((1, 4, 32, 32))
    # ipdb.set_trace()
    emb_pth = os.path.join(img_dir, 'emb_noise', pth[0])
    torch.save(enc_out, emb_pth)
    print("saved embedding to: ", emb_pth)









