from torchvision.io import read_video
from videogpt import load_vqvae
from videogpt.data import preprocess
import torch
import ipdb

video_filename = '/home/jay/VPT/data/SegTrackv2/mp4s/girl.mp4'
sequence_length = 16
resolution = 64
device = torch.device('cuda')

vqvae = load_vqvae('bair_stride4x2x2')
video = read_video(video_filename, pts_unit='sec')[0]
ipdb.set_trace()
video = preprocess(video.cuda(), resolution, sequence_length).unsqueeze(0).to(device)
ipdb.set_trace()

encodings = vqvae.encode(video.cpu())
ipdb.set_trace()
video_recon = vqvae.decode(encodings)
ipdb.set_trace()