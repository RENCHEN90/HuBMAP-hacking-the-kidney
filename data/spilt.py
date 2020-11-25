import numpy as np
from sklearn.model_selection import train_test_split
import glob
import torch
import cv2
import os

image_root = '/home/casxm/zhangqin/Data/crop_data_1024_512/mask/*'
img_list = glob.glob(image_root)
for img_path in img_list:
    img = cv2.imread(img_path, 0)
    img = img * 255
    cv2.imwrite(img_path,img)
print('done')





image_root = '/home/casxm/zhangqin/Data/crop_data_1024_512/image/*'

img_list = glob.glob(image_root)

train_list, val_list = train_test_split(img_list, test_size=0.1, random_state=20201120)
print(len(train_list), len(val_list))

mask_train_list = [png_path.replace('image','mask') for png_path in train_list]
mask_val_list = [png_path.replace('image', 'mask') for png_path in val_list]

train= np.column_stack((train_list,mask_train_list))
np.savetxt('./train_set.csv', train, delimiter=',',fmt ='%s')

val = np.column_stack((val_list, mask_val_list))
np.savetxt('./val_set.csv', val, delimiter = ',',fmt ='%s')