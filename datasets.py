import torch
from torch.utils.data import Dataset
import pycocotools
import torchvision
import numpy as np
import os


class Coco_Dataset(Dataset):
    def __init__(self, img_dir, anno_file, transform=None):
        self.img_dir = img_dir
        self.anno_file = anno_file
        self.coco = pycocotools.coco.COCO(anno_file)
        self.cat_ids = self.coco.getCatIds()
        self.im2float = torchvision.transforms.ConvertImageDtype(torch.float32)
        
    def __len__(self):
        return len(self.coco.imgs)
    
    def __getitem__(self, index):
        # index = 0
        img = self.coco.imgs[index]
        # image = torch.tensor(np.array(Image.open(os.path.join(self.img_dir, img['file_name']))))
        image = torchvision.io.read_image(os.path.join(self.img_dir, img['file_name']))
        image = self.im2float(image)
        
        anns_ids_1 = self.coco.getAnnIds(imgIds=img['id'], catIds=1, iscrowd=None)
        anns_1 = self.coco.loadAnns(anns_ids_1)
        

        anns_ids_2 = self.coco.getAnnIds(imgIds=img['id'], catIds=2, iscrowd=None)
        anns_2 = self.coco.loadAnns(anns_ids_2)
        ### TODO: DANGER!!! do we get every class every time ? ###
        mask = np.zeros((len(self.cat_ids)-1, image.shape[1], image.shape[2]))
        # mask = self.coco.annToMask(anns[0])
        # print('cat ids', self.cat_ids)
        # ipdb.set_trace()
        
        for i in range(0, len(anns_1)):
            mask[0] += self.coco.annToMask(anns_1[i])

        for i in range(0, len(anns_2)):
            mask[1] += self.coco.annToMask(anns_2[i])
        
        mask = torch.tensor(mask)
        mask = mask.clamp(0, 1)
        return image, mask
    
class Coco_Dataset_embedder(Dataset):
    def __init__(self, img_dir, anno_file, transform=None):
        self.img_dir = img_dir
        self.anno_file = anno_file
        self.coco = pycocotools.coco.COCO(anno_file)
        self.cat_ids = self.coco.getCatIds()
        self.im2float = torchvision.transforms.ConvertImageDtype(torch.float32)
        
    def __len__(self):
        return len(self.coco.imgs)
    
    def __getitem__(self, index):
        # index = 0
        img = self.coco.imgs[index]
        # image = torch.tensor(np.array(Image.open(os.path.join(self.img_dir, img['file_name']))))
        pth = os.path.join(self.img_dir, img['file_name'])
        image = torchvision.io.read_image(pth)
        image = self.im2float(image)
        
        anns_ids_1 = self.coco.getAnnIds(imgIds=img['id'], catIds=1, iscrowd=None)
        anns_1 = self.coco.loadAnns(anns_ids_1)
        

        anns_ids_2 = self.coco.getAnnIds(imgIds=img['id'], catIds=2, iscrowd=None)
        anns_2 = self.coco.loadAnns(anns_ids_2)
        ### TODO: DANGER!!! do we get every class every time ? ###
        mask = np.zeros((len(self.cat_ids)-1, image.shape[1], image.shape[2]))
        # mask = self.coco.annToMask(anns[0])
        # print('cat ids', self.cat_ids)
        # ipdb.set_trace()
        
        for i in range(0, len(anns_1)):
            mask[0] += self.coco.annToMask(anns_1[i])

        for i in range(0, len(anns_2)):
            mask[1] += self.coco.annToMask(anns_2[i])
        
        mask = torch.tensor(mask)
        mask = mask.clamp(0, 1)
        return image, mask, img['file_name']
    

    
class Coco_Dataset_Embeddings(Dataset):
    def __init__(self, img_dir, anno_file, transform=None):
        self.img_dir = img_dir
        self.embedding_dir = os.path.join(self.img_dir, 'emb')
        self.anno_file = anno_file
        self.coco = pycocotools.coco.COCO(anno_file)
        self.cat_ids = self.coco.getCatIds()

    def __len__(self):
        return len(self.coco.imgs)
    
    def __getitem__(self, index):
        # index = 1
        img = self.coco.imgs[index]
        # image = torch.tensor(np.array(Image.open(os.path.join(self.img_dir, img['file_name']))))
        image = torch.load(os.path.join(self.embedding_dir, img['file_name']))
        # print(img['file_name'])
        
        # anns_ids_1 = self.coco.getAnnIds(imgIds=img['id'], catIds=1, iscrowd=None)
        # anns_1 = self.coco.loadAnns(anns_ids_1)
        

        anns_ids_2 = self.coco.getAnnIds(imgIds=img['id'], catIds=1, iscrowd=None)
        anns_2 = self.coco.loadAnns(anns_ids_2)


        ### TODO: DANGER!!! do we get every class every time ? ###
        mask = np.zeros((64, 64))

        # for i in range(0, len(anns_1)):
        #     mask[0] += self.coco.annToMask(anns_1[i])

        for i in range(0, len(anns_2)):
            mask += self.coco.annToMask(anns_2[i])
        
        mask = torch.tensor(mask)
        mask = mask.clamp(0, 1)
        return image, mask
    
    
    
    
class Coco_Dataset_Embeddings_Noise(Dataset):
    def __init__(self, img_dir, anno_file, transform=None):
        self.img_dir = img_dir
        self.embedding_dir = os.path.join(self.img_dir, 'emb_noise')
        self.anno_file = anno_file
        self.coco = pycocotools.coco.COCO(anno_file)
        self.cat_ids = self.coco.getCatIds()

    def __len__(self):
        return len(self.coco.imgs)
    
    def __getitem__(self, index):
        # index = 1
        img = self.coco.imgs[index]
        # image = torch.tensor(np.array(Image.open(os.path.join(self.img_dir, img['file_name']))))
        image = torch.load(os.path.join(self.embedding_dir, img['file_name']))
        # print(img['file_name'])
        
        # anns_ids_1 = self.coco.getAnnIds(imgIds=img['id'], catIds=1, iscrowd=None)
        # anns_1 = self.coco.loadAnns(anns_ids_1)
        

        anns_ids_2 = self.coco.getAnnIds(imgIds=img['id'], catIds=2, iscrowd=None)
        anns_2 = self.coco.loadAnns(anns_ids_2)


        ### TODO: DANGER!!! do we get every class every time ? ###
        mask = np.zeros((64, 64))

        # for i in range(0, len(anns_1)):
        #     mask[0] += self.coco.annToMask(anns_1[i])

        for i in range(0, len(anns_2)):
            mask += self.coco.annToMask(anns_2[i])
        
        mask = torch.tensor(mask)
        mask = mask.clamp(0, 1)
        return image, mask