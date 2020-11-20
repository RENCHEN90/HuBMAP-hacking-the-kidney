import os
import argparse

import numpy as np
import cv2
import pandas as pd
import tiffile as tiff
import tqdm


from tissue_seg import locate_tissue

# def csv_to_mask(csv_data):
#     '''
#     orignal anotation is csv
#     need change to binaray mask
#     '''

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


def mask2enc(mask, shape, n=1):
    pixels = mask.T.flatten
    encs = []
    for i in range(1, n + 1):
        p = (pixels == i).astype(np.uint8)
        if p.sum() == 0:
            encs.append(np.nan)
        else:
            p = np.concatenate([[0], p, [0]])
            runs = np.where(p[1:] != p[:-1])[0] + 1
            runs[1::2] -= runs[::2]
            encs.append(' '.join(str(x) for x in runs))
    return encs

def img_add_mask(img, mask, color=(0, 200, 0), thick=1):    
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    res = cv2.drawContours(img, cnts, -1, color, thick)
    return res

def resize_img(img, dst_shape=(512,512), isMask=False):
    if isMask:
        interpolation = cv2.INTER_NEAREST
    else:
        interpolation = cv2.INTER_AREA
    res = cv2.resize(img, dst_shape, interpolation=interpolation)
    return res


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
            rects.add((left, up, right,down)) 
    return list(rects)


def get_tissue(img):
    h,w = img.shape[:2]
    fx = 1024/w
    res_img = cv2.resize(img, (0, 0), fx=fx, fy=fx, interpolation=cv2.INTER_AREA)
    black_mask = cv2.inRange(res_img, (0, 0, 0), (1, 1, 1))
    black_mask = black_mask / 255 * 220
    black_border = cv2.merge([black_mask,black_mask,black_mask])
    dst = (black_border + res_img).astype(np.uint8) 
    cnts, tissue = locate_tissue(dst)
    # dst = cv2.drawContours(res_img, cnts, -1, (0, 255, 0), 2)
    tissue = cv2.resize(tissue, (0, 0), fx=1 / fx, fy=1 / fx, interpolation=cv2.INTER_NEAREST)
    cnts,_ = cv2.findContours(tissue,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return cnts,tissue





def crop_img_check(opt):
    tumor_dir = os.path.join(opt.crop_check_dir+'/tumor')
    tumor_non_dir = os.path.join(opt.crop_check_dir + '/non_tumor')
    # tissue_non_dir = os.path.join(opt.crop_check_dir + '/non_tissue')
    os.makedirs(tumor_dir,exist_ok=True)
    os.makedirs(tumor_non_dir, exist_ok=True)
    
    train_csv_path = opt.train_csv
    df_data = pd.read_csv(train_csv_path).set_index('id')
    rec_num = 2955
    for index, encs in df_data.iterrows():
        if index in ['aaa6a05cc', 'cb2d976f4', '0486052bb', '2f6ecfcdf']:
            continue
        img = tiff.imread(os.path.join(opt.train_dir, index + '.tiff'))
        
        if len(img.shape) == 5:
            img = (np.transpose(img.squeeze(), (1, 2, 0))).astype(np.uint8)
        
        # print(img.shape)
        h, w = img.shape[:2]
        mask = enc2mask(encs, (w, h))
        # print(mask.shape)        
        # fx = 1024/w
        # res_mask = cv2.resize(mask, (0, 0), fx=fx, fy=fx, interpolation=cv2.INTER_NEAREST)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        print(index, 'cnt number:', len(cnts))
        t_cnts, tissue = get_tissue(img)
        print(type(img))
        draw_img = cv2.drawContours(img.copy(), cnts, -1, (0, 255, 0), 2)      
        for i,cnt in enumerate(t_cnts):        
            x, y, ww, hh = cv2.boundingRect(cnt)
            recs = get_fixed_windows((ww, hh), (opt.aim_size,opt.aim_size),(opt.overlap_size,opt.overlap_size))
            for j, rec in enumerate(recs):
                # isWhite = False  
                x1, y1, x2, y2 = rec            
                dst = draw_img[y + y1:y + y2, x + x1:x + x2]
                num = np.sum(mask[y + y1:y + y2, x + x1:x + x2])
                tt = tissue>0
                tissue_num = np.sum(tt[y + y1:y + y2, x + x1:x + x2])
                if tissue_num < 100 and num < 10:
                    continue
                if num < 10:
                    dst_dir = tumor_non_dir
                else:
                    dst_dir = tumor_dir
                dst_path = os.path.join(dst_dir,'{:0>6d}_{}_{}_{}.png'.format(rec_num,index,x1,y1))
                cv2.imwrite(dst_path, dst)
                rec_num +=1
                print(dst_path)         
    print('all done:',rec_num)
            
        




def check_data(opt):
    train_csv_path = opt.train_csv
    df_data = pd.read_csv(train_csv_path).set_index('id')
    for index, encs in df_data.iterrows():
        img = tiff.imread(os.path.join(opt.train_dir, index + '.tiff'))
        if len(img.shape) == 5:
            img = np.transpose(img.squeeze(), (1,2,0))
        print(img.shape)
        h, w = img.shape[:2]
        mask = enc2mask(encs, (w, h))
        print(mask.shape)
        fx = 0.2
        fy = 0.2
        res_img = cv2.resize(img, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
        res_mask = cv2.resize(mask, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_NEAREST)
        res = img_add_mask(res_img,res_mask)
        cv2.imwrite(os.path.join(opt.check_dir,index+'.png'),res)

        
def get_tissue_mask(opt):
    train_csv_path = opt.train_csv
    df_data = pd.read_csv(train_csv_path).set_index('id')
    for index, encs in df_data.iterrows():
        img = tiff.imread(os.path.join(opt.train_dir, index + '.tiff'))
        if len(img.shape) == 5:
            img = np.transpose(img.squeeze(), (1,2,0))
        print(img.shape)
        h, w = img.shape[:2]
        fx = 1024/w     
        res_img = cv2.resize(img, (0, 0), fx=fx, fy=fx, interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(opt.tissue_check_dir,index+'_img_resize.png') ,res_img)        
        black_mask = cv2.inRange(res_img, (0, 0, 0), (1, 1, 1))
        black_mask = black_mask / 255 * 220
        cv2.imwrite(os.path.join(opt.tissue_check_dir,index+'_black_mask.png') ,black_mask)
        black_border = cv2.merge([black_mask,black_mask,black_mask])
        dst = (black_border + res_img).astype(np.uint8)
        cv2.imwrite(os.path.join(opt.tissue_check_dir,index+'_mid.png') ,dst)  
        cnts, tissue = locate_tissue(dst)
        dst = cv2.drawContours(res_img, cnts, -1, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(opt.tissue_check_dir,index+'_tissuse_cnt.png') ,dst)        

def crop_img(opt):
    tumor_dir = os.path.join(opt.crop_dir,'tumor/image')
    tumor_non_dir = os.path.join(opt.crop_dir,'non_tumor/image')

    tumor_dir_mask = os.path.join(opt.crop_dir,'tumor/mask')
    tumor_non_dir_mask = os.path.join(opt.crop_dir ,'non_tumor/mask')

    # tissue_non_dir = os.path.join(opt.crop_check_dir + '/non_tissue')
    os.makedirs(tumor_dir,exist_ok=True)
    os.makedirs(tumor_non_dir, exist_ok=True)
    os.makedirs(tumor_dir_mask,exist_ok=True)
    os.makedirs(tumor_non_dir_mask, exist_ok=True)

    train_csv_path = opt.train_csv
    df_data = pd.read_csv(train_csv_path).set_index('id')
    rec_num = 0
    for index, encs in df_data.iterrows():
        if index in ['aaa6a05cc', 'cb2d976f4', '0486052bb', '2f6ecfcdf']:
            continue
        img = tiff.imread(os.path.join(opt.train_dir, index + '.tiff'))        
        if len(img.shape) == 5:
            img = (np.transpose(img.squeeze(), (1, 2, 0))).astype(np.uint8)        
        # print(img.shape)
        h, w = img.shape[:2]
        mask = enc2mask(encs, (w, h))
        # print(mask.shape)        
        # fx = 1024/w
        # res_mask = cv2.resize(mask, (0, 0), fx=fx, fy=fx, interpolation=cv2.INTER_NEAREST)
        # cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # print(index, 'cnt number:', len(cnts))
        t_cnts, tissue = get_tissue(img)        
        # draw_img = cv2.drawContours(img.copy(), cnts, -1, (0, 255, 0), 2) 
             
        for i,cnt in enumerate(t_cnts):        
            x, y, ww, hh = cv2.boundingRect(cnt)
            recs = get_fixed_windows((ww, hh), (opt.aim_size,opt.aim_size),(opt.overlap_size,opt.overlap_size))
            for j, rec in enumerate(recs):
                # isWhite = False  
                x1, y1, x2, y2 = rec            
                dst = img[y + y1:y + y2, x + x1:x + x2]
                dst_mask = mask[y + y1:y + y2, x + x1:x + x2]
                num = np.sum(mask[y + y1:y + y2, x + x1:x + x2])
                tt = tissue>0
                tissue_num = np.sum(tt[y + y1:y + y2, x + x1:x + x2])
                if tissue_num < 100 and num < 10:
                    continue
                if num < 10:
                    dst_dir = tumor_non_dir
                    dst_dir_mask = tumor_non_dir_mask
                else:
                    dst_dir = tumor_dir
                    dst_dir_mask = tumor_dir_mask
                dst = cv2.resize(dst, (512, 512), cv2.INTER_AREA)
                dst_mask = cv2.resize(dst_mask, (512, 512), cv2.INTER_NEAREST)
                dst_path = os.path.join(dst_dir, '{:0>6d}_{}_{}_{}.png'.format(rec_num, index, x1, y1))
                dst_path_mask = os.path.join(dst_dir_mask, '{:0>6d}_{}_{}_{}.png'.format(rec_num, index, x1, y1))
                cv2.imwrite(dst_path, dst)
                cv2.imwrite(dst_path_mask,dst_mask)
                rec_num +=1
                print(dst_path)         
    print('all done:',rec_num)
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str,
                        default='/home/casxm/zhangqin/Data/train.csv', help='ori_image_dir')
    parser.add_argument('--train_dir', type=str,
                    default='/home/casxm/zhangqin/Data/train', help='ori_image_dir')
    parser.add_argument('--check_dir', type=str,
                default='/home/casxm/zhangqin/Data/check_data', help='ori_image_dir')
    parser.add_argument('--tissue_check_dir', type=str,
                default='/home/casxm/zhangqin/Data/tissue_check_data', help='ori_image_dir')

    parser.add_argument('--crop_check_dir', type=str,
            default='/home/casxm/zhangqin/Data/crop_check_data_1024_256', help='ori_image_dir')

 
    parser.add_argument('--crop_dir', type=str,
            default='/home/casxm/zhangqin/Data/crop_data_1024resizeto512', help='ori_image_dir')

    # parser.add_argument('--ori_image_dir', type=str,
    #                     default='/home/casxm/zhangqin/Data/DDTI/1_or_data/image', help='ori_image_dir')
    # parser.add_argument('--ori_mask_dir', type=str,
    #                     default='/home/casxm/zhangqin/Data/DDTI/1_or_data/mask', help='ori_mask_dir')                
    # parser.add_argument('--dst_dir', type=str,
    #                     default='/home/casxm/zhangqin/TNSCUI2020-Seg-Rank1st/DDTI/data', help='dst_dir')




    parser.add_argument('--aim_size',type=int,default=1024,help='stage 1 aim size')
    parser.add_argument('--overlap_size', type=int, default=256, help='stage 2 aim size')    
    # parser.add_argument('--value_threshold',type=int,default=5,help='remove the black border by cropping')
    opt = parser.parse_args()


    
    # check_data(opt)
    # get_tissue_mask(opt)

    crop_img(opt)

    

    
    
            
    