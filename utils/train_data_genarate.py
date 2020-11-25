import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# from PIL import Image
import tifffile as tiff
import cv2
import os
from tqdm import tqdm
# import zipfile
import gc

sz = 512   #the size of tiles
reduce_time = 2  #reduce the original images by 4 times 
s_th = 40  #saturation blancking threshold
p_th = 200 * sz // 256  #threshold for the minimum number of pixels
over_lap = 128
MASKS = '/home/casxm/zhangqin/Data/train.csv'
DATA = '/home/casxm/zhangqin/Data/train'
OUT_TRAIN = '/home/casxm/zhangqin/Data/crop_data_1024_512/image'
OUT_MASKS = '/home/casxm/zhangqin/Data/crop_data_1024_512/mask'

os.makedirs(OUT_TRAIN, exist_ok=True)
os.makedirs(OUT_MASKS, exist_ok=True)


def enc2mask(encs, shape):
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for m, enc in enumerate(encs):
        if isinstance(enc, np.float) and np.isnan(enc):
            continue
        s = enc.split()
        for i in range(len(s) // 2):
            start = int(s[2 * i]) - 1
            length = int(s[2 * i+1])            
            img[start:(start + length)] = 1 + m
    return img.reshape(shape).T

def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

# New version
def rle_encode_less_memory(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    This simplified method requires first and last pixel to be zero
    '''
    pixels = img.T.flatten()
    
    # This simplified method requires first and last pixel to be zero
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    
    return ' '.join(str(x) for x in runs)



def get_fixed_windows(mask_size, wind_size, overlap_size):
    '''
    This function can generate overlapped windows given various image size
    params:
        mask_size (w, h): the image width and height
        wind_size (w, h): the window width and height
        overlap (overlap_w, overlap_h): the overlap size contains x-axis and y-axis
    return:
        rects [(xmin, ymin, xmax, ymax)]: the windows in a list of rectangles
    '''
    rects = set()
    if mask_size[0] < wind_size[0] or mask_size[1] < wind_size[1]:
        return []
    assert overlap_size[0] < wind_size[0]
    assert overlap_size[1] < wind_size[1]
    im_w = wind_size[0] if mask_size[0] < wind_size[0] else mask_size[0]
    im_h = wind_size[1] if mask_size[1] < wind_size[1] else mask_size[1]
    stride_w = wind_size[0] - overlap_size[0]
    stride_h = wind_size[1] - overlap_size[1]
    # area_th = wind_size[1] * wind_size[0] * 0.9
    c_x = im_w // 2
    c_y = im_h //2
    # delta_x = (c_x - (wind_size[0] - 2 * overlap_size[0])) % stride_w    
    # delta_y = (c_y - (wind_size[1] - 2 * overlap_size[1])) % stride_h
    for j in range(wind_size[1]-1, im_h + stride_h, stride_h):
        for i in range(wind_size[0]-1, im_w + stride_w, stride_w):
            right, down = i+1, j+1
            right = right if right < im_w else im_w
            down  =  down if down < im_h  else im_h
            left = right - wind_size[0]
            up = down - wind_size[1]            
            rects.add((left, up , right, down ))
             
    return list(rects)


# img = np.zeros((1028, 1028, 3), np.uint8)
# recs = get_fixed_windows((1028, 1028), (512, 512), (128, 128))
# print(len(recs))
# for  i,rec in enumerate(recs):
#     x1, y1, x2, y2 = rec    
#     color = (50+i*10,0,255-i*5)
#     cv2.rectangle(img, (x1,y1), (x2,y2),color, 2)
# cv2.imwrite('test.png',img)


df_masks = pd.read_csv(MASKS).set_index('id')
print(df_masks.head())

x_tot,x2_tot = [],[]
rec_num = 0
for index, encs in df_masks.iterrows():
    #read image and generate the mask
    ori_img = tiff.imread(os.path.join(DATA,index+'.tiff'))
    if len(ori_img.shape) == 5: ori_img = np.transpose(ori_img.squeeze(), (1, 2, 0))   

    h,w = ori_img.shape[:2]
    ori_mask = enc2mask(encs,(w,h))

    #split image and mask into tiles using the reshape+transpose trick
    # img = cv2.resize(ori_img,(ori_img.shape[1]//reduce_time,ori_img.shape[0]//reduce_time),
    #                     interpolation = cv2.INTER_AREA) 
    # mask = cv2.resize(ori_mask,(ori_img.shape[1]//reduce_time,ori_img.shape[0]//reduce_time),
    #                     interpolation = cv2.INTER_NEAREST)
    recs = get_fixed_windows((w,h),(sz*reduce_time,sz*reduce_time),(over_lap,over_lap))

    #write data
    print(index, len(recs),ori_img.shape,)
    

    # print(recs)
    for i, rec in enumerate(recs):
        #remove black or gray images based on saturation check
        x1, y1, x2, y2 = rec
        # print(rec)
        im = ori_img[y1:y2, x1:x2]        
        mm = ori_mask[y1:y2, x1:x2]
        # print(im.shape,mm.shape)

        im = cv2.resize(im, (512, 512), interpolation=cv2.INTER_AREA)
        mm = cv2.resize(mm, (512, 512), interpolation=cv2.INTER_NEAREST)

        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        if (s>s_th).sum() <= p_th or im.sum() <= p_th: continue 
        num = str(rec_num).zfill(6)
        im_path = os.path.join(OUT_TRAIN, f'{num}_{index}_{x1}_{y1}.png')
        mm_path = os.path.join(OUT_MASKS, f'{num}_{index}_{x1}_{y1}.png')
        cv2.imwrite(im_path, im)
        cv2.imwrite(mm_path, mm)
        rec_num += 1
        x_tot.append((im/255.0).reshape(-1,3).mean(0))
        x2_tot.append(((im/255.0)**2).reshape(-1,3).mean(0))       
        # del hsv, x1, y1, x2, y2
        # gc.collect()
    print(index, len(recs),rec_num)
        

#image stats
img_avr =  np.array(x_tot).mean(0)
img_std =  np.sqrt(np.array(x2_tot).mean(0) - img_avr**2)
print('mean:',img_avr, ', std:', img_std)
        
