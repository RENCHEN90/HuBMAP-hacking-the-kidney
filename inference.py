import os
import numpy as np
import cv2
import argparse

import torch
import tiffile as tiff

from utils import process,tissue_seg
from utils.process import get_fixed_windows
from utils.tissue_seg import locate_tissue


# a = [1, 2, 3, 4, 5, 6, 7]
# batch_size = 3
# a = [a[i:i + batch_size] for i in range(0, len(a), batch_size)]
# print(a)

def get_miou(result, reference):
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))
    intersection = np.count_nonzero(result & reference)    
    size_i1 = np.count_nonzero(result)
    size_i2 = np.count_nonzero(reference)
    try:
        miou = intersection / float(size_i1 + size_i2 - intersection)        
    except ZeroDivisionError:
        miou = 0.0    
    return miou

def get_dice(result, reference):
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))
    intersection = np.count_nonzero(result & reference)    
    size_i1 = np.count_nonzero(result)
    size_i2 = np.count_nonzero(reference)    
    try:
        dc = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc = 0.0    
    return dc


def get_model(opt):
    model = PraNet()
    model = torch.nn.DataParallel(model)
    model.cuda()
    model.load_state_dict(torch.load(opt.model_path))
    model.eval()
    return model

def predict(tiff_path,model,opt):
    img = tiff.imread(tiff_path)
    if len(img.shape) == 5:
        img = (np.transpose(img.squeeze(), (1, 2, 0))).astype(np.uint8)        
        # print(img.shape)
    h, w = img.shape[:2]
    t_cnts, tissue = get_tissue(img)
    boxes = []
    for i, cnt in enumerate(t_cnts):
        x, y, ww, hh = cv2.boundingRect(cnt)
        recs = get_fixed_windows((ww, hh), (opt.aim_size,opt.aim_size),(opt.overlap_size,opt.overlap_size))
        boxes.extend(recs)
    boxes_batchs = [boxes[i:i + opt.batch_size] for i in range(0, len(boxes), opt.batch_size)]
    
    img_temp = np.zeros((h, w), dtype=np.uint8)
    res = np.zeros((h, w))
    for box_batchs in boxes_batchs:
        img_batch = np.zeros(len(box_batchs), 3,opt.img_size, opt.img_size,)        
        for jj, rec in enumerate(box_batchs):            
            x1, y1, x2, y2 = rec            
            img_box = img[y + y1:y + y2, x + x1:x + x2]  # 512,512,3
            img_box = cv2.resize(img_box,(opt.img_size,opt.img_size),interpolation=cv2.INTER_AREA)
            img_box = np.transpose(img_box, (2, 0, 1)) # 3,512,512
            img_batch[jj, ...] = img_box
            # img_temp 
            img_temp[y + y1:y + y2, x + x1:x + x2] = img_temp[y + y1:y + y2, x + x1:x + x2]+1

        # 预测模型        
        res5, res4, res3, res2 = model(image)
        res = res2
        res = F.upsample(res, size=(opt.aim_size,opt.aim_size), mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        img_batch_res = res  # n,1,1024,1024

        # 结果返回原图
        for jj, rec in enumerate(box_batchs):            
            x1, y1, x2, y2 = rec
            res[y + y1:y + y2, x + x1:x + x2] = res[y + y1:y + y2, x + x1:x + x2] + img_batch_res[jj, ...]
    res = res / img_temp
    res = res>0.5
    return res.astype(np.uint8)


def main(opt):
    ####   strat predict   ####
    model = get_model(opt)
    with open(opt.test_list, 'r') as f:
        tiff_paths = f.readlines()

    for tiff_path in tiff_paths:
       res = predict(tiff_path, model, opt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_list', type=str,
                        default='/home/casxm/zhangqin/HuBMAP-hacking-the-kidney/data/test_set.txt', help='ori_image_dir')
    parser.add_argument('--aim_size',type=int,default=1024,help='stage 1 aim size')
    parser.add_argument('--overlap_size', type=int, default=256, help='stage 2 aim size')    
    parser.add_argument('--img_size', type=int, default=512)    
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--model_path', type=str, default='')
    opt = parser.parse_args()
    main(opt)