import os
import argparse

def my_opt():
    parser = argparse.ArgumentParser()
    
    ###### General settings
    parser.add_argument('--device',default='1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    parser.add_argument('--num_workers', default=1)

    parser.add_argument('--log_path',default='logs', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    parser.add_argument('--model_save_path',default='runs')

    parser.add_argument('--result_path', default='train_temp')

    
    ###### TRAIN SETTING ######
    parser.add_argument('--modelmark', default='20201126_01')
       
    parser.add_argument('--train_data', type=str,
                            default='/home/casxm/zhangqin/HuBMAP-hacking-the-kidney/data/train_set.csv', help='ori_image_dir')

    parser.add_argument('--nfolds',default=4)

    parser.add_argument('--batchsize', default=1)
    
    parser.add_argument('--train_size',default=512)
    parser.add_argument('--epoch',default=500)
    parser.add_argument('--start_save_epoch', default=1)
    parser.add_argument('--lr',default=0.001)
    parser.add_argument('--optimizer_type', default='SGD')
    parser.add_argument('--lr_scheduler', default='CosineAnnealingWarmRestarts')
    parser.add_argument('--lossfunction',default='symmetric_lovasz')
    parser.add_argument('--multi_scale_training',default=False)

    parser.add_argument('--preweight_path',default='')

    parser.add_argument('--model_type',default='R2AttU_Net',help='R2AttU_Net/PraNet/UneXt50')
    parser.add_argument('--t',default=2)
    parser.add_argument('--clip',default=0.5)


    parser.add_argument('--save_train_img', default=8)

    ######## VAL SETTING
    parser.add_argument('--val_data', default='/home/casxm/zhangqin/HuBMAP-hacking-the-kidney/data/val_set.csv')
    parser.add_argument('--val_bs', default=1)
    

    ###### Test SETTING ######



    ######################################

    opt = parser.parse_args()
    opt.modelmark = opt.model_type+'_'+opt.modelmark 

    with open(opt.modelmark+'.txt', 'w') as f:
        for key, value in vars(opt).items():
            f.write('%s:%s\n'%(key, value))
        # print(key, value)
    return opt