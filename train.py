


# Import necessary packages.
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from load_data import Dataset_paried,DataLoaderTrain,DataLoaderVal
from torch.utils.data import DataLoader

# This is for the progress bar.
from tqdm.auto import tqdm
from image_utiles import batch_PSNR,myPSNR
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
from torch.utils.tensorboard import SummaryWriter
from model_utiles import load_optim,load_start_epoch,load_checkpoint,load_psnr
from timm.utils import NativeScaler

from torch.optim.lr_scheduler import CosineAnnealingLR
# model
import yaml


batch_size = 48
step = 300
epochs = 400
use_gpu = True
lr = 1e-3
resume = False #是否断点续训

dataset = 'Rain100L'



train_path = 'Datasets/'+dataset+'/Train'
val_path = 'Datasets/'+dataset+'/Test'
#val_path = '../papers/Deraining/Datasets/test/Test1200'
save_path = './loger/GADB/'+dataset
loger_path = './loger/GADB/'+dataset+'/record'
record_path = './loger/GADB/'+dataset+'/log.txt'


if not os.path.exists(loger_path):
    os.makedirs(loger_path)

if not os.path.exists(save_path):
    os.makedirs(save_path)
    
writer = SummaryWriter(log_dir=loger_path)



print('------------------------')
print('Loading datasets')
img_options_train = {'patch_size':128}
train_dataset = DataLoaderTrain(train_path, img_options_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,num_workers = 8,pin_memory=True, drop_last=False)

val_dataset = DataLoaderVal(val_path)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False,num_workers = 8)
print('Loading done')




loss_scaler = NativeScaler()

######### DataParallel ###########
model = GADN()
#model = PReNet()
model = torch.nn.DataParallel (model,device_ids=[0,1,2,3])
print('---------------------')

criterion1 = nn.L1Loss()

#criterion = PSNRLoss()
if use_gpu:            
    model = model.cuda()
    criterion1 = criterion1.cuda()
    
optimizer = optim.AdamW([{'params': model.parameters(), 'initial_lr': lr}], lr=lr, betas=(0.9, 0.999),eps=1e-8, weight_decay=0.02)

best_psnr = 0
if resume:
    path_chk_rest = os.path.join(save_path, "model_rain100.pth")
    load_checkpoint(model,path_chk_rest)
    start_epoch = load_start_epoch(path_chk_rest) + 1
    lr = load_optim(optimizer, path_chk_rest)
    best_psnr = load_psnr(path_chk_rest)
    print("目前的epoch是:{},目前的学习率是{},val_psnr是{}".format(start_epoch,lr,best_psnr))
else:
    start_epoch = 1

with torch.no_grad():
    model.eval()
    psnr_val = []
    for ii, data_val in enumerate((val_loader), 0):
        target = data_val[1]
        input_ = data_val[0]
        psnr_val.append(calculate_psnr(input_, target,crop_border=0,test_y_channel=True))
    psnr_val = np.mean(psnr_val)
    print('原始的的PSNR为%.2f'%(psnr_val))
    

        
print('测试成功！！！')


scheduler = CosineAnnealingLR(optimizer,T_max=step,eta_min=1e-7,last_epoch=start_epoch)
eval_now = len(train_loader)//2
print('每{}iteration测试一次'.format(eval_now))
for epoch in range(start_epoch,epochs):
    model.train()
    epoch_loss = 0
    train_psnr = []
    for i,batch in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        input,target = batch
        if use_gpu:
            input = input.cuda()
            target = target.cuda()
        
        result = model(input)
        result = torch.clamp(result, 0, 1)
            
        loss = criterion1(result,target)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        if (i + 1) % eval_now == 0 and i > 0 and epoch>200:
            with torch.no_grad():
                model.eval()
                psnr_val = []
                for ii, data_val in enumerate((val_loader), 0):
                    target = data_val[1].cuda()
                    input_ = data_val[0].cuda()
                    
                    restored = model(input_)
                    restored = torch.clamp(restored, 0, 1) #将输出的数值限制到0，1
                    psnr_val.append(calculate_psnr(target, restored,crop_border=0,test_y_channel=True))# 我是傻逼！！！
                psnr_val = np.mean(psnr_val)
                print('val的PSNR为%.2f'%(psnr_val))
                writer.add_scalar('PSNR/val', psnr_val, epoch)
                
                
                if psnr_val > best_psnr:
                    best_psnr = psnr_val
                    best_epoch = epoch
                    torch.save({'epoch': best_epoch,
                                'state_dict': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'best_psnr':best_psnr
                                }, os.path.join(save_path, "model_rain100.pth"))
                    print('epoch:{},best_psnr:{}保存成功'.format(best_epoch,best_psnr))
            
    scheduler.step()
    print('--------------')
    print(epoch)
    torch.save({'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'best_psnr':best_psnr
                            }, os.path.join(save_path, "model_latest.pth"))
    
    epoch_loss = epoch_loss/len(train_loader)
    print('loss:{},lr:{}'.format(epoch_loss,scheduler.get_last_lr()[0]))
    writer.add_scalar('loss', epoch_loss, epoch)
    with open(record_path,"a+") as f:
        f.write(str(epoch_loss)+' '+str(psnr_val)+'\n')
    



