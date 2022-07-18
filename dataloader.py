import torch
from torch.utils import data
from torchvision import transforms
import csv
import numpy as np
import cv2
import sys
import os
sys.path.append('../utils')
import random
from augmentation import *

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            # img = img.resize((200, 150))
            return img.convert('RGB')

def DataAllocate(batch):
    seqs = []
    for sample in batch:
        seqs.append(sample)
    seqs = torch.stack(seqs, 0)
    return seqs

def resize_dim(w, h, target):
    '''resize (w, h), such that the smaller side is target, keep the aspect ratio'''
    if w >= h:
        return (int(target * w / h), int(target))
    else:
        return (int(target), int(target * h / w)) 


class Kinetics_Data(data.Dataset):
    
    def __init__(self, file, mode='train', seq=16, sample=1, sr=[1, 3], transform=None):
        self.mode = mode
        self.seq = seq
        self.sample = sample
        self.sr = sr
        self.transform = transform
        self.list = {}
        reader = csv.reader(open(file))
        for row in reader:
            path, frames = row
            frames = int(frames)
            if frames > self.sample * self.seq:
                self.list[path] = frames
            else:
                continue
        self.keys = list(self.list.keys())
        
    def __len__(self):
        return len(self.list)
    
    def sample_frame(self, index):
        path = self.keys[index]
        frames = self.list[path]
        back_idx = np.random.randint(low=0, high=self.seq)
        max_sr = min(self.sr[1], frames/(self.seq*self.sample))
        sr = np.random.uniform(low=self.sr[0], high=max_sr)
        start = np.random.randint(low=0, high=int(frames-sr*self.seq*self.sample)+1)
        samples = sr*np.arange(self.seq*self.sample) + start
        samples = np.round(samples)
        samples = np.minimum(samples, frames-1)
        return path, samples, back_idx
    
    def __getitem__(self, index):
        path, samples, back_idx = self.sample_frame(index)
        images = [pil_loader(os.path.join(path, 'image_%05d.jpg' % (i+1))) for i in samples]
        frame_1 = [images[back_idx] for _ in range(self.seq)]
        frame_2 = [images[back_idx+self.seq] for _ in range(self.seq)]
        rgb_1 = self.transform(images[:self.seq])
        rgb_2 = self.transform(images[self.seq:])
        frame_1 = self.transform(frame_1)
        frame_2 = self.transform(frame_2)
        c, h, w = rgb_1[0].size()
        rgb_1 = torch.stack(rgb_1, 0).view(1, self.seq, c, h, w).transpose(1, 2)
        rgb_2 = torch.stack(rgb_2, 0).view(1, self.seq, c, h, w).transpose(1, 2)
        frame_1 = torch.stack(frame_1, 0).view(1, self.seq, c, h, w).transpose(1, 2)
        frame_2 = torch.stack(frame_2, 0).view(1, self.seq, c, h, w).transpose(1, 2)
        # frame_1 = rgb_1[:, :, back_idx].unsqueeze(2).repeat([1, 1, self.seq, 1, 1])
        # frame_2 = rgb_2[:, :, back_idx].unsqueeze(2).repeat([1, 1, self.seq, 1, 1])
        res_1 = rgb_1 - torch.roll(rgb_1, 1, 2)
        res_2 = rgb_2 - torch.roll(rgb_2, 1, 2)
        return torch.cat([rgb_1, rgb_2, frame_1, frame_2, res_1, res_2], dim=0)
        
