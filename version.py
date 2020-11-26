import torch
import torchvision
# import pycocotools

print(torch.__version__)
print(torchvision.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())

print(torch.backends.cudnn.is_available())

print(torch.cuda.get_device_name(0))