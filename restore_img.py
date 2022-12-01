import torch
import numpy as np
from image_utiles import load_img
from model_utiles import load_checkpoint,load_start_epoch
import os
import time
import cv2
from GADN import GADN
import torch.nn.functional as F

dataset = 'Rain100H'
model_name = 'GADN'
input_path = 'Datasets/'+dataset+'/Test/input'

save_path = './loger/'+model_name+'/'+dataset+'/best.pth'

restore_path = './restore/'+model_name+'/'+dataset
if not os.path.exists(restore_path):
    os.makedirs(restore_path)
    
def caculate_fps(model,restorefiles,input_path):
    with torch.no_grad():
        model.eval()
        all_cost_time = 0
        for ii, filename in enumerate(restorefiles):
            img = torch.from_numpy(np.float32(load_img(os.path.join(input_path,filename))))
            img = img.permute(2, 0, 1)
            img = torch.unsqueeze(img, dim=0)
            #print(img.shape)
            input_ = img.cuda()
            strat_time = time.time()
            _, _, h, w = input_.size()
            padsize = 16
            mod_pad_h = (padsize - h % padsize) % padsize
            mod_pad_w = (padsize - w % padsize) % padsize
            input_ = F.pad(input_, (0, mod_pad_w, 0, mod_pad_h))
            restored = model(input_)
            restored = restored[:,:,:h,:w]
            end_time = time.time()
            
            restored = torch.clamp(restored, 0, 1)
            cost_time = end_time - strat_time
            all_cost_time += cost_time
            save_out = np.uint8(255 * restored.data.cpu().numpy().squeeze())
            save_out = save_out.transpose(1, 2, 0)
            b, g, r = cv2.split(save_out)
            save_out = cv2.merge([r, g, b])
            cv2.imwrite(os.path.join(restore_path, filename), save_out)
        return all_cost_time,len(restorefiles)


model = GADN().cuda()

load_checkpoint(model,save_path)
start_epoch = load_start_epoch(save_path)
print(start_epoch)
restorefiles = os.listdir(input_path)
cost_time,num = caculate_fps(model,restorefiles,input_path)
print(cost_time,num,num/cost_time)
print("FPS为{}".format(num/cost_time))

