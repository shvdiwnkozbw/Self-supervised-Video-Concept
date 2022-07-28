import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
import os
import sys
sys.path.append('../utils')
from augmentation import *
from model import *
from dataloader import Kinetics_Data, DataAllocate
import argparse
from tqdm import tqdm
from progress.bar import Bar
import time
import torch.nn.functional as F
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--net', default='resnet18', type=str)
parser.add_argument('--seq', default=16, type=int)
parser.add_argument('--sample', default=2, type=int)
parser.add_argument('--rate', default=3.0, type=float)
parser.add_argument('--train_batch', default=96, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--wd', default=1e-5, type=float)
parser.add_argument('--img_dim', default=128, type=int)
parser.add_argument('--workers', default=32, type=int)
parser.add_argument('--channel', default=128, type=int)
parser.add_argument('--pos', default=6, type=int)
parser.add_argument('--neg', default=201, type=int)
parser.add_argument('--concept', default=100, type=int)
parser.add_argument('--gpu', default='1,2,3', type=str)
parser.add_argument('--epoch', default=200, type=int)
parser.add_argument('--start', default=0, type=int)
parser.add_argument('--ckpt', default='r3d-kin', type=str)
parser.add_argument('--resume', default=0, type=int)
parser.add_argument('--path', default='', type=str)
parser.add_argument('--mode', default='grad', type=str)
parser.add_argument('--static', default=0, type=int)
parser.add_argument('--eval', default=0, type=int)

args = parser.parse_args()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
    
def contrast(q, k, celoss):
    # q,k nc
    q = F.normalize(q)
    k = F.normalize(k)
    matrix = torch.einsum('nc,mc->nm', q, k.detach())
    label = torch.arange(matrix.shape[0]).to(device)
    loss = celoss(matrix/0.07, label)
    return loss

def train(model, dataloader, optimizer, nllloss, celoss, epoch):
    model.train()
    total_loss = 0
    
    for idx, video in enumerate(dataloader):
        if idx > len(dataloader):
            break
        video = video.to(device)
        end = time.time()
        
        with torch.no_grad():
            w = model.module.prototype.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            model.module.prototype.weight.copy_(w)
            w = model.module.prototype_s.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            model.module.prototype_s.weight.copy_(w)
            w = model.module.prototype_d.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            model.module.prototype_d.weight.copy_(w)

        use_queue = 1 if epoch > 0 else 0

        proto, MI_est = model(video, use_queue)

        rgb_v1, rgb_v2, rgb_s1, rgb_s2, rgb_d1, rgb_d2, frame_v1, frame_v2, res_v1, res_v2 = MI_est
        prob, loss_static, loss_dynamic, entropy_static, entropy_dynamic, rec_v, rec_s, rec_d, local_v, local_s, local_d = proto
        
        I_vv = contrast(rgb_v1, rgb_v2, celoss) + contrast(rgb_v2, rgb_v1, celoss)
        I_vv /= 2
        I_vs = contrast(rgb_s1, frame_v2, celoss) + contrast(rgb_s2, frame_v1, celoss) + contrast(frame_v1, rgb_s2, celoss) + contrast(frame_v2, rgb_s1, celoss)
        I_vs /= 2
        I_vd = contrast(rgb_d1, res_v2, celoss) + contrast(rgb_d2, res_v1, celoss) + contrast(res_v1, rgb_d2, celoss) + contrast(res_v2, rgb_d1, celoss)
        I_vd /= 2
        
        if epoch < 10:
            loss = I_vv + I_vd + I_vs + loss_static.mean() + loss_dynamic.mean() + 0.1*entropy_static.mean() + 0.1*entropy_dynamic.mean() + rec_v.mean() + rec_s.mean() + rec_d.mean()
        else:
            loss = I_vv + I_vd + I_vs + loss_static.mean() + loss_dynamic.mean() + 0.1*entropy_static.mean() + 0.1*entropy_dynamic.mean() + rec_v.mean() + rec_s.mean() + rec_d.mean() + local_v.mean() + local_s.mean() + local_d.mean()
        # loss = loss_static.mean() + loss_dynamic.mean() + loss_joint.mean() + 0.5*entropy_static.mean() + 0.5*entropy_dynamic.mean()
        model.module.queue[:-prob.shape[0]] = model.module.queue[prob.shape[0]:].clone()
        model.module.queue[-prob.shape[0]:] = prob

        if idx % 10 == 0:
            print('iteration: %d/%d | loss: %.3f | I_vv: %.3f | I_vs: %.3f | I_vd: %.3f | L_s: %.3f | E_s: %.3f | L_d: %.3f | E_d: %.3f | R_v: %.3f | R_s: %.3f | R_d: %.3f | L_v: %.3f | L_s: %.3f | L_d: %.3f' % 
                  (idx, len(dataloader), loss.item(), I_vv.item(), I_vs.item(), I_vd.item(), loss_static.mean().item(), entropy_static.mean().item(), 
                  loss_dynamic.mean().item(), entropy_dynamic.mean().item(), rec_v.mean().item(), rec_s.mean().item(), rec_d.mean().item(), local_v.mean().item(), local_s.mean().item(), local_d.mean().item()))

        # if idx % 10 == 0:
        #     print('iteration: %d/%d | loss: %.3f | L_s: %.3f | E_s: %.3f | L_d: %.3f | E_d: %.3f | L_v: %.3f' % 
        #           (idx, len(dataloader), loss.item(), loss_static.mean().item(), entropy_static.mean().item(), 
        #           loss_dynamic.mean().item(), entropy_dynamic.mean().item(), loss_joint.mean().item()))

        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        if epoch == 0:
            for name, p in model.named_parameters():
                if 'prototype' in name:
                    p.grad = None
        optimizer.step()
    return total_loss/idx

def main():
    global device; device = torch.device('cuda')
    transform = transforms.Compose([
        RandomSizedCrop(size=args.img_dim, consistent=True, p=1.0),
        RandomHorizontalFlip(consistent=True),
        RandomGray(consistent=True, p=0.5),
        ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, consistent=True, p=1.0),
        ToTensor(),
        Normalize()
    ])
    dataset = Kinetics_Data(file='../process_data/src/ucf.csv', mode='train',
                            seq=args.seq, sample=args.sample, sr=[1.0, args.rate],
                            transform=transform)
    dataloader = data.DataLoader(dataset, shuffle=True, batch_size=args.train_batch,
                                 collate_fn=DataAllocate, num_workers=args.workers, drop_last=True)
    
    model = GreatModel(args.concept)
    model = nn.DataParallel(model)
    start = 0
    if args.resume:
        state = torch.load(args.path)
        model.load_state_dict(state, strict=True)
        start = 156
        print('load weight')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.999))
    nllloss = nn.NLLLoss(reduce=False, size_average=False)
    celoss = nn.CrossEntropyLoss().cuda()
    print('start training')
    print(len(dataloader))
    for e in range(start, args.epoch):
        if e == 120 or e == 160:
            args.lr = 0.1 * args.lr
            for param in optimizer.param_groups:
                param['lr'] = args.lr
        
        loss = train(model, dataloader, optimizer, nllloss, celoss, e)
        if (e+1) % 10 == 0:
            torch.save(model.state_dict(), '../ckpt/r2d-view-order-128/r3d_%d_%.3f.pth'%(e, loss))

if __name__ == '__main__':
    setup_seed(20)
    main()
