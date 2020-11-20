import os
import numpy as np
import cv2
from scipy.ndimage import binary_fill_holes
from skimage import filters, img_as_ubyte
from skimage.morphology import remove_small_objects

def thresh_slide(gray, thresh_val, sigma=13):
    """ Threshold gray image to binary image
    Parameters
    ----------
    gray : np.array
        2D gray image.
    thresh_val: float
        Thresholding value.
    smooth_sigma: int
        Gaussian smoothing sigma.
    Returns
    -------
    bw_img: np.array
        Binary image
    """

    # Smooth
    smooth = filters.gaussian(gray, sigma=sigma)
    smooth /= np.amax(smooth)
    # Threshold
    bw_img = smooth < thresh_val

    return bw_img

def fill_tissue_holes(bw_img):
    """ Filling holes in tissue image
    Parameters
    ----------
    bw_img : np.array
        2D binary image.
    Returns
    -------
    bw_fill: np.array
        Binary image with no holes
    """

    # Fill holes
    bw_fill = binary_fill_holes(bw_img)

    return bw_fill

def remove_small_tissue(bw_img, min_size=10000):
    """ Remove small holes in tissue image
    Parameters
    ----------
    bw_img : np.array
        2D binary image.
    min_size: int
        Minimum tissue area.
    Returns
    -------
    bw_remove: np.array
        Binary image with small tissue regions removed
    """

    bw_remove = remove_small_objects(bw_img, min_size=min_size, connectivity=8)

    return bw_remove


def find_tissue_cnts(bw_img):
    """ Fint contours of tissues
    Parameters
    ----------
    bw_img : np.array
        2D binary image.
    Returns
    -------
    cnts: list
        List of all contours coordinates of tissues.
    """

    cnts, _ = cv2.findContours(img_as_ubyte(bw_img), mode=cv2.RETR_EXTERNAL,
                               method=cv2.CHAIN_APPROX_NONE)

    return cnts

def locate_tissue(img, smooth_sigma=13, thresh_val=0.88, min_tissue_size=20000):
    if img.ndim == 2:
        gray_img = img
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        print('locate tissue fail: img type is not surport')
        return None    
    bw_img = thresh_slide(gray_img, thresh_val, sigma=smooth_sigma)
    bw_fill = fill_tissue_holes(bw_img)
    bw_remove = remove_small_tissue(bw_fill, min_tissue_size)
    cnts = find_tissue_cnts(bw_remove)
    return cnts,img_as_ubyte(bw_img)



if __name__ == "__main__":
    img_dir ='/home/casxm/zhangqin/Biospy/oripng/image'
    mask_dir = '/home/casxm/zhangqin/Biospy/oripng/mask'
    dst_dir = '/home/casxm/zhangqin/Biospy/normalmask'
    dst_dir2 = '/home/casxm/zhangqin/Biospy/normalmask_check'
    smooth_sigma=13
    thresh_val = 0.88
    min_tissue_size=20000

    for i, img_name in enumerate(os.listdir(img_dir)):
        img_path = os.path.join(img_dir, img_name)
        dst_path = os.path.join(dst_dir, img_name)
        dst_path2 = os.path.join(dst_dir2, img_name)
        # mask_path = os.path.join(mask_dir,img_name[:-4]+'_mask.png')
        gray_img = cv2.imread(img_path, 0)
        ori_img = cv2.imread(img_path)

        bw_img = thresh_slide(gray_img, thresh_val, sigma=smooth_sigma)
        bw_fill = fill_tissue_holes(bw_img)
        bw_remove = remove_small_tissue(bw_fill, min_tissue_size)
        cnts = find_tissue_cnts(bw_remove)

        mask = np.zeros(gray_img.shape,dtype=np.uint8)
        mask = cv2.drawContours(mask, cnts, -1, (255), -1)

        img = cv2.drawContours(ori_img, cnts, -1, (255), 10)
        img = cv2.resize(img, (0, 0), fx=0.1, fy=0.1)
        
        cv2.imwrite(dst_path, mask)
        cv2.imwrite(dst_path2, img)
        print(i,img_name)