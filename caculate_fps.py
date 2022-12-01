import numpy as np
from torchvision.models import resnet50,resnet18
import torch
from torch.backends import cudnn
import tqdm
import time
from model.GADN import GADN


torch.backends.cudnn.enabled = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True

device = 'cuda:0'
model = GADN().to(device)


repetitions = 100
    

dummy_input = torch.rand(1, 3, 480, 320).to(device)

# 预热, GPU 平时可能为了节能而处于休眠状态, 因此需要预热
print('warm up ...\n')
with torch.no_grad():
    for _ in range(1):
        _ = model(dummy_input,)

# synchronize 等待所有 GPU 任务处理完才返回 CPU 主线程
torch.cuda.synchronize()


# 设置用于测量时间的 cuda Event, 这是PyTorch 官方推荐的接口,理论上应该最靠谱
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
# 初始化一个时间容器
timings = np.zeros((repetitions, 1))

all_time = 0
print('testing ...\n')
with torch.no_grad():
    for rep in tqdm.tqdm(range(repetitions)):
        #starter.record()
        start = time.time()
        _ = model(dummy_input,)
        torch.cuda.synchronize()
        all_time += time.time() - start
        #ender.record()
        #curr_time = starter.elapsed_time(ender) # 从 starter 到 ender 之间用时,单位为毫秒
        #timings[rep] = curr_time

# avg = timings.sum()/repetitions #1张图片的时间
# fps = 1/avg
# print('\navg={}s\n'.format(avg))
# print('\nFPS={}\n'.format(fps))

avg = all_time/repetitions 
fps = repetitions/all_time
print('\nreal_avg={}s\n'.format(avg))
print('\nreal_FPS={}\n'.format(fps))
