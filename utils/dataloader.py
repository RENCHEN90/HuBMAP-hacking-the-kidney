import os
import numpy as np
import cv2
import pandas as pd

import gc

from albumentations import *
from sklearn.model_selection import KFold
# import matplotlib.pyplot as plt
import torch
import torch.utils.data as data

import torchvision

from PIL import Image


# bs = 64
# nfolds = 4
# fold = 0
# SEED = 2020
# TRAIN = '../input/hubmap-256x256/train/'
# MASKS = '../input/hubmap-256x256/masks/'
# LABELS = '../input/hubmap-kidney-segmentation/train.csv'
# NUM_WORKERS = 4


###########
# https://www.kaggle.com/iafoss/256x256-images
mean = np.array([0.65459856, 0.48386562, 0.69428385])
std = np.array([0.15167958, 0.23584107, 0.13146145])

def img2tensor(img,dtype:np.dtype=np.float32):
    if img.ndim==2 : img = np.expand_dims(img,2)
    img = np.transpose(img,(2,0,1))
    return torch.from_numpy(img.astype(dtype, copy=False))

# class HuBMAPDataset(Dataset):
#     def __init__(self, fold=fold, train=True, tfms=None):
#         ids = pd.read_csv(LABELS).id.values
#         kf = KFold(n_splits=nfolds,random_state=SEED,shuffle=True)
#         ids = set(ids[list(kf.split(ids))[fold][0 if train else 1]])
#         self.fnames = [fname for fname in os.listdir(TRAIN) if fname.split('_')[0] in ids]
#         self.train = train
#         self.tfms = tfms
        
#     def __len__(self):
#         return len(self.fnames)
    
#     def __getitem__(self, idx):
#         fname = self.fnames[idx]
#         img = cv2.cvtColor(cv2.imread(os.path.join(TRAIN,fname)), cv2.COLOR_BGR2RGB)
#         mask = cv2.imread(os.path.join(MASKS,fname),cv2.IMREAD_GRAYSCALE)
#         if self.tfms is not None:
#             augmented = self.tfms(image=img,mask=mask)
#             img,mask = augmented['image'],augmented['mask']
#         return img2tensor((img/255.0 - mean)/std),img2tensor(mask)
    
# def get_aug(p=1.0):
#     return Compose([
#         HorizontalFlip(),
#         VerticalFlip(),
#         RandomRotate90(),
#         ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.9, 
#                          border_mode=cv2.BORDER_REFLECT),
#         OneOf([
#             OpticalDistortion(p=0.3),
#             GridDistortion(p=.1),
#             IAAPiecewiseAffine(p=0.3),
#         ], p=0.3),
#         OneOf([
#             HueSaturationValue(10,15,10),
#             CLAHE(clip_limit=2),
#             RandomBrightnessContrast(),            
#         ], p=0.3),
#     ], p=p)




class HuBMAPDataset(data.Dataset):
    def __init__(self, data_csv, transform=None):
        self.trainsize = 512
        self.images = []
        self.gts = []
        with open(data_csv, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.split()[0]
            if line:
                img_path,gt_path = line.split(',', -1)[:2]
                if img_path and gt_path:
                    self.images.append(img_path)
                    self.gts.append(gt_path)
        

        # self.images =self.images[:20]
        # self.gts = self.gts[:20]

        self.size = len(self.images)


        self.img_transform =torchvision.transforms.Compose([torchvision.transforms.Resize((self.trainsize, self.trainsize)), torchvision.transforms.ToTensor()])
        self.gt_transform = torchvision.transforms.Compose([torchvision.transforms.Resize((self.trainsize, self.trainsize)),torchvision.transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        
        image = self.img_transform(image)
        # image = image/255.0
        gt = self.gt_transform(gt)
        return image, gt
    
    def rgb_loader(self, path):
        # img = cv2.imread(path,1)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        # img = cv2.imread(path,0)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('1')


    def __len__(self):
        return self.size


def get_loader(data_list_txt, batchsize=4, shuffle=True, num_workers=4, pin_memory=False):
    dataset = HuBMAPDataset(data_list_txt)
    data_loader =data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=False)
    return data_loader



if __name__ == "__main__":
    #example of train images with masks
    ds = HuBMAPDataset(tfms=get_aug())
    dl = DataLoader(ds,batch_size=64,shuffle=False,num_workers=NUM_WORKERS)
    imgs,masks = next(iter(dl))
    plt.figure(figsize=(16,16))
    for i,(img,mask) in enumerate(zip(imgs,masks)):
        img = ((img.permute(1,2,0)*std + mean)*255.0).numpy().astype(np.uint8)
        plt.subplot(8,8,i+1)
        plt.imshow(img,vmin=0,vmax=255)
        plt.imshow(mask.squeeze().numpy(), alpha=0.2)
        plt.axis('off')
        plt.subplots_adjust(wspace=None, hspace=None)
    plt.savefig('trainwithmask_demo.png')   
    del ds,dl,imgs,masks