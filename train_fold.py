# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline


from fastai.vision.all import *
from evaluation import Dice_th_pred, Model_pred


###### model 
from UneXt50.UneXt50 import UneXt50,split_layers


####  dataset
from utils.dataloader import MY_HuBMAPDataset,get_aug    

from utils.utils import select_device
from tensorboardX import SummaryWriter
from tqdm import tqdm

from datetime import datetime

from utils.process import img_add_mask

from loss_metric.Lovasz_loss import symmetric_lovasz
from loss_metric.metric import Dice_soft,Dice_th
    
import warnings
warnings.filterwarnings("ignore")

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
####   train   ####
def seed_everything(seed):
    random.seed(seed)
    # dls.rng.seed(x)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    #the following line gives ~10% speedup
    #but may lead to some stochasticity in the results 
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def main(opt):
    torch.backends.cudnn.benchmark = True
    device = select_device(opt.device, batch_size=opt.batchsize)
    dice = Dice_th_pred(np.arange(0.2, 0.7, 0.01))
    for fold in range(opt.nfolds):
        ds_t = MY_HuBMAPDataset(data_csv=opt.train_data, nfolds=opt.nfolds, fold=fold, train=True, tfms=get_aug())
        ds_v = MY_HuBMAPDataset(data_csv=opt.train_data, nfolds=opt.nfolds, fold=fold, train=False, tfms=get_aug())
        data = ImageDataLoaders.from_dsets(ds_t,ds_v,bc=opt.batchsize,num_workers=opt.num_workers,pin_memory=False).to(device)
        model = UneXt50().to(device)
        learn = Learner(data,model,loss_func = symmetric_lovasz,metrics = [Dice_soft(),Dice_th()],splitter = split_layers).to_fp16(clip=0.5)

        #start with training the head
        learn.freeze_to(-1) #doesn't work
        for param in learn.opt.param_groups[0]['params']:
            param.requires_grad = False
        learn.fit_one_cycle(6, lr_max=0.5e-2)
        #continue training full model
        learn.unfreeze()
        learn.fit_one_cycle(32, lr_max=slice(2e-4,2e-3),
            cbs=SaveModelCallback(monitor='dice_th', comp=np.greater))
        
        save_name = f'model_{fold}.pth'
        save_path = os.path.join( opt.model_save_path,save_name)

        torch.save(learn.model.state_dict(),save_path)
        
        #model evaluation on val and saving the masks
        mp = Model_pred(learn.model,learn.dls.loaders[1])
        with zipfile.ZipFile('val_masks_tta.zip', 'a') as out:
            for p in progress_bar(mp):
                dice.accumulate(p[0],p[1])
                save_img(p[0],p[2],out)
        gc.collect()

    dices = dice.value
    noise_ths = dice.ths
    best_dice = dices.max()
    best_thr = noise_ths[dices.argmax()]
    plt.figure(figsize=(8,4))
    plt.plot(noise_ths, dices, color='blue')
    plt.vlines(x=best_thr, ymin=dices.min(), ymax=dices.max(), colors='black')
    d = dices.max() - dices.min()
    plt.text(noise_ths[-1]-0.1, best_dice-0.1*d, f'DICE = {best_dice:.3f}', fontsize=12);
    plt.text(noise_ths[-1]-0.1, best_dice-0.2*d, f'TH = {best_thr:.3f}', fontsize=12);
    # plt.show()
    plt.savefig('train_dice.png')

if __name__ == "__main__":
    seed_everything(2020)
    
    from opt import my_opt
    opt = my_opt()
    creat_dir(opt)
    #### ---- log
    writer = SummaryWriter(opt.log_path)
    print(opt)
    main(opt)