import torch
import torchvision
# import pycocotools

import numpy as np








def get_miou(result, reference):
    if np.count_nonzero(reference) == 0:
        result = 1 - result
        reference = 1- reference
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))
    intersection = np.count_nonzero(result & reference)    
    size_i1 = np.count_nonzero(result)
    size_i2 = np.count_nonzero(reference)
    if size_i1 == 0 and size_i2 == 0:
        miou = 1.0
    else:
        miou = intersection / float(size_i1 + size_i2 - intersection) 
    return miou


gt = np.zeros((2, 4))
print(gt)
arr = np.array([[0.9,0.2,0.8,0.7],[0.2,0.52,0.4,0.8]])
print(arr)

miou = get_miou(arr > 0.5, gt > 0.5)
print(miou)
miou = get_miou(arr < 0.5, gt > 0.5)
print(miou)



print(torch.__version__)
print(torchvision.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())

print(torch.backends.cudnn.is_available())

print(torch.cuda.get_device_name(0))