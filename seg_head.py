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

class seghead(nn.Module):
    def __init__(self, n_classes):
        super(seghead, self).__init__()

        self.latent = [8, 2, 32]
        self.classes = n_classes

        self.decode1 = nn.Sequential(
            nn.ConvTranspose2d(self.latent[0], 4, 2, 2, padding = 0),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace= True)
        )

        self.decode2 = nn.Sequential(
            nn.ConvTranspose2d(4, self.classes, 2, 2, padding = 0), 
            nn.BatchNorm2d(self.classes),
            nn.ReLU(inplace= True)
        )

    def forward(self, x):
        y = self.decode1(x)
        out = self.decode2(y)
        return out

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
    
def loss(pred, gt, n_classes = 2):

    if n_classes == 1:
        crit_2 = dice_loss(F.sigmoid(pred.squeeze(1)), gt.float(), multiclass=False)
    else:
        crit_2 = dice_loss(
            F.softmax(pred, dim=1).float(),
            F.one_hot(gt, n_classes).permute(0, 3, 1, 2).float(),
            multiclass=True
        )

    crit_1 = nn.BCEWithLogitsLoss(pos_weight= 10)
    loss = crit_1(pred, gt) + crit_2
    return loss

class SegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images_dir = os.path.join(root_dir, 'images')
        self.masks_dir = os.path.join(root_dir, 'masks')
        self.image_filenames = os.listdir(self.images_dir)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_dir, self.image_filenames[idx])
        mask_name = os.path.join(self.masks_dir, self.image_filenames[idx].replace('.jpg', '_mask.png'))

        image = Image.open(img_name).convert('RGB')
        mask = Image.open(mask_name)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

            #fix this to get after the COCO dataset

        return image, mask


transform = transforms.Compose([
    transforms.CenterCrop((64, 64)),
    transforms.Resize((64, 64)),  
    transforms.ToTensor(),           
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
])

def train(encoder,
        decoder,   
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_frequency: float = 0.5,
        save_checkpoint_every = 10000,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        train_images_dir=None,
        train_mask_dir=None,
        val_images_dir=None,
        show_mask_every = 100, 
        val_mask_dir=None,
        dir_checkpoint=None,):
    

    dataset = SegmentationDataset(root_dir='path/to/dataset', transform=transform)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    for params in encoder.parameters():
        params.requires_grad = False

    optimizer = optim.Adam(decoder.parameters(), lr = learning_rate, weight_decay= weight_decay)
    batches = len(data_loader)
    step = 0

    for epo in range(epochs):
        step += 1
        for image, mask in data_loader:
            enc_out = encoder(image)
            dec_out = decoder(enc_out)
            #change the image mask
            loss_val = loss(dec_out, mask)

            optimizer.zero_grad()

            loss_val.backward()

            optimizer.step()

            if step % val_frequency == 0:
                print("loss at step ", step, " :" , loss_val.detach().numpy())

            #save model 
            if step % save_checkpoint_every == 0:
                torch.save(encoder.state_dict(), dir_checkpoint)

            #display results
            if step % show_mask_every == 0:
                image_np = dec_out.numpy()

                # If your tensor has a batch dimension, remove it
                if len(image_np.shape) == 4:
                    image_np = image_np.squeeze(0)

                # If your image is in channel-first format, transpose it to channel-last format (optional)
                if image_np.shape[0] == 3:
                    image_np = image_np.transpose(1, 2, 0)

                # Plot the image
                plt.imshow(image_np)
                plt.axis('off')  # Turn off axis

                # Save the image
                plt.savefig('output_image.png', bbox_inches='tight', pad_inches=0)

                # Show the image (optional)
                plt.show()



if __name__ == "main":
    encoder = 
    decoder = seghead(n_classes= 2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(encoder,
        decoder,   
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_frequency: float = 0.5,
        save_checkpoint: bool = True,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        train_images_dir=None,
        train_mask_dir=None,
        val_images_dir=None,
        val_mask_dir=None,
        dir_checkpoint=None,)
    



    


