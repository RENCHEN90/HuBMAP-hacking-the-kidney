import argparse
import os
import numpy as np
import cv2

import torch
from torch.backends import cudnn
from torch.autograd import Variable
import torch.nn.functional as F
import random

from evaluation import Dice_th_pred, Model_pred
###### model 
from UneXt50.UneXt50 import UneXt50,split_layers


####  dataset
# from utils.dataloader import MY_HuBMAPDataset, get_aug
####  dataset
from utils.dataloader import get_loader



from utils.utils import select_device
from tensorboardX import SummaryWriter
from tqdm import tqdm

from datetime import datetime

from utils.process import img_add_mask

from loss_metric.Lovasz_loss import symmetric_lovasz
from loss_metric.metric import Dice_soft,Dice_th
    
import warnings
warnings.filterwarnings("ignore")




def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    #the following line gives ~10% speedup
    #but may lead to some stochasticity in the results 
    torch.backends.cudnn.benchmark = True




# def symmetric_lovasz(outputs, targets):
#     return 0.5*(lovasz_hinge(outputs, targets) + lovasz_hinge(-outputs, 1.0 - targets))


def get_miou(result, reference):
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

def get_dice(result, reference):
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))
    intersection = np.count_nonzero(result & reference)    
    size_i1 = np.count_nonzero(result)
    size_i2 = np.count_nonzero(reference)    
    if size_i1 == 0 and size_i2 == 0:
        dc = 1.0
    else:
        dc = 2. * intersection / float(size_i1 + size_i2)   
    return dc


def creat_dir(opt):
    # Create directories if not exist
    opt.log_path = os.path.join(opt.log_path,opt.modelmark)
    if not os.path.exists(opt.log_path):
        os.makedirs(opt.log_path,exist_ok=True)
    opt.model_save_path = os.path.join(opt.model_save_path,opt.modelmark)    
    if not os.path.exists(opt.model_save_path):
        os.makedirs(opt.model_save_path,exist_ok=True)
    opt.result_path = os.path.join(opt.result_path,opt.modelmark)
    if not os.path.exists(opt.result_path):
        os.makedirs(opt.result_path, exist_ok=True)
    
    
def print_network(model, name):
    """
    Print out the network information.
    """
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print(name)
    print("The number of parameters: {}".format(num_params))


def to_data(x):
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data()


def reset_grad(model):
    """Zero the gradient buffers."""
    return model.zero_grad()

def tensor2img(x):
    img = (x[:,0,:,:]>x[:,1,:,:]).float()
    img = img*255
    return img



def build_model(opt):
    """Build generator and discriminator."""
    # if opt.model_type =='U_Net':
    #     model = U_Net(img_ch=3,output_ch=1)
    # elif opt.model_type =='R2U_Net':
    #     model = R2U_Net(img_ch=3,output_ch=1,t=opt.t)
    # elif opt.model_type =='AttU_Net':
    #     model = AttU_Net(img_ch=3,output_ch=1)
    # elif opt.model_type == 'R2AttU_Net':
    #     model = R2AttU_Net(img_ch=3, output_ch=1, t=opt.t)
    # elif opt.model_type == 'PraNet':
    #     model = PraNet(pretrained=True)

    model = UneXt50()

    return model

def train(epoch,model,optimizer,train_loader,val_loader=None,device='cuda'):
    trainsize = opt.train_size
    model.train()
    if opt.multi_scale_training:
        size_rates = [0.75, 1, 1.25]
    else:
        size_rates = [1]
    epoch_loss = 0

    batch_tqdm = tqdm(enumerate(train_loader),total = len(train_loader),ncols=100)

    for i, pack in batch_tqdm:
        for rate in size_rates:
            optimizer.zero_grad()
            # ----  data  prepare ----
            images, gts = pack
            images = Variable(images).to(device)
            gts = Variable(gts).to(device)
            # ---- rescale ----
            trainsize = int(round(trainsize * rate / 32) * 32)
            if rate != 1:
                images = torch.nn.functional.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = torch.nn.functional.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

            #  forward
            output = model(images)
            out_probs = F.sigmoid(output)

            out_flat = out_probs.view(out_probs.size(0),-1)
            gts_flat = gts.view(gts.size(0),-1)

            #  loss function
            loss = symmetric_lovasz(out_flat,gts_flat)    # TODO: try different weights for loss
            # loss = torch.nn.BCELoss(out_flat,gts_flat)
            epoch_loss += loss.item()
            batch_tqdm.set_description('iter %i'%i)
            batch_tqdm.set_postfix(loss=loss.item(),epoch_loss=epoch_loss)
            # ---- backward ----
            model.zero_grad()
            loss.backward()               
            optimizer.step()

    print('Epoch %d,loss:%.6f'%(epoch,epoch_loss))

    writer.add_scalar('lr/model_lr', optimizer.param_groups[0]['lr'], epoch)
    writer.add_scalar('train/loss', epoch_loss, epoch)

    if epoch % 1 == 0 and val_loader:
        dice, miou = eval_net(model, val_loader, epoch,device)
        writer.add_scalar('val/dice', dice, epoch)
        writer.add_scalar('val/miou', miou, epoch)
        print('Epoch {},val data set,dice:{:0.6f},miou:{:0.6f}'.format(epoch, dice, miou))
    save_name = '%s-%d_%.6f_%.6f.pth' % (opt.model_type, epoch, dice, miou)
    save_path = os.path.join( opt.model_save_path,save_name)
    if epoch > opt.start_save_epoch-1:
        torch.save(model.state_dict(), save_path)
        print(epoch, '[Saving Snapshot:]', save_path)


def eval_net(model, test_loader,epoch,device):
    model = model.eval()
    dice = 0
    miou = 0
    test_loader.index = 0
    draw_num=0
    for i,(image,gt) in tqdm(enumerate(test_loader),total = len(test_loader),ncols=50):        
        ori_img = image.numpy().copy()
        gt = np.asarray(gt, np.float32)
        # gt /= (gt.max() + 1e-8)
        image = image.to(device)
        if opt.model_type == 'PraNet':
            res5, res4, res3, res2 = model(image)
            res = res2
        else:
            res = model(image)
        res = F.upsample(res, size=(opt.train_size,opt.train_size), mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        res = res > 0.5
        gt = gt.squeeze() > 0.5

        if draw_num < opt.save_train_img:        
            dst_img = np.transpose(ori_img.squeeze(), (1, 2, 0))
            # print(np.max(dst_img))
            dst_img = (dst_img * 255).astype(np.uint8)
            # print(np.max(dst_img))
            dst_img = cv2.cvtColor(dst_img, cv2.COLOR_RGB2BGR)
            dst_img = img_add_mask(dst_img.copy(), gt.astype(np.uint8).copy(), (0, 255, 0), 1)
            dst_img = img_add_mask(dst_img.copy(), res.astype(np.uint8).copy(), (255, 0, 0), 1)
            dst_path = os.path.join(opt.result_path, str(epoch) + '_' + str(i) + '_res.png')            
            cv2.imwrite(dst_path, dst_img)
            draw_num +=1

        dd = get_dice(res, gt)
        mm = get_miou(res, gt)
        dice = dice + dd
        miou = miou + mm    
    return dice / len(test_loader), miou / len(test_loader)


def main(opt):
        # cudnn.benchmark = True

    device = select_device(opt.device, batch_size=opt.batchsize)
    # device = torch.device('cuda')

    # print(opt)

    #########  数据加载   #########
    train_loader = get_loader(opt.train_data,opt.batchsize,shuffle=True, num_workers=4, pin_memory=False)
    
    if opt.val_data:
        val_loader = get_loader(opt.val_data,opt.val_bs,shuffle=False, num_workers=1, pin_memory=False)
    



    ##########  模型加载  ########
    model = build_model(opt)
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    ######## 预训练参数加载 ########
    if opt.preweight_path:
        model.load_state_dict(torch.load(opt.preweight_path))

    ####### 定义模型优化器 #############
    if opt.optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    elif opt.optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(),
                              lr=opt.lr,
                              momentum=0.9,
                              weight_decay=1e-8,
                              nesterov=True)
    ####### 学习率 衰减方式
    if opt.lr_scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=0, last_epoch=-1)
    elif opt.lr_scheduler == 'LambdaLR':
        lambda1 = lambda epoch: epoch // 30
        lambda2 = lambda epoch: 0.95 ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
    elif opt.lr_scheduler == 'MultiplicativeLR':
        lmbda = lambda epoch: 0.95
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
    elif opt.lr_scheduler == 'StepLR':
        lmbda = lambda epoch: 0.95
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif opt.lr_scheduler == 'MultiStepLR':        
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[30,80,120,200], gamma=0.1) 
    elif opt.lr_scheduler == 'ExponentialLR':        
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    elif opt.lr_scheduler == 'CosineAnnealingLR':        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0,)
    elif opt.lr_scheduler == 'ReduceLROnPlateau':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)      
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')            
    elif opt.lr_scheduler == 'ReduceLROnPlateau':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)      
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1)
    
    #### ---- loss function ----
    if opt.lossfunction == 'BCE':
        opt.criterion = torch.nn.BCELoss()
    elif opt.lossfunction == 'structure_loss':
        opt.criterion = structure_loss
    elif opt.lossfunction == 'symmetric_lovasz' :
        opt.criterion = symmetric_lovasz


    print('##########################################\n\n\ntrain start')
    for epoch in range(0, opt.epoch):
        if opt.model_type == 'PraNet':
            parnet_train(epoch,model,optimizer,train_loader,val_loader,device)
        else:
            train(epoch,model,optimizer,train_loader,val_loader,device)
        scheduler.step()
    print('all train done')

if __name__ == "__main__":
    
    seed_everything(2020)

    from opt import my_opt
    opt = my_opt()
    creat_dir(opt)
    #### ---- log
    writer = SummaryWriter(opt.log_path)

    print(opt)
    main(opt)

