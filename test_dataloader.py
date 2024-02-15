import os
import torch
import numpy as np
import os.path as osp

# import sys
# sys.path.append('../tools')
# from utils import load_image
from utils import load_image
from torch.utils import data


class Robocup(data.Dataset):
    def __init__(self, scene, data_path, train, val, transform=None, target_transform=None, real=False, seed=7):
        self.transform = transform
        self.target_transform = target_transform
        self.data_path = data_path
        self.seed = seed
        self.data_dir = osp.join(data_path, 'Robocup', scene)

        all_imgs = [f for f in os.listdir(self.data_dir) if f.endswith('.jpg')]
        np.random.seed(self.seed) 
        np.random.shuffle(all_imgs)
        
        self.imgs = all_imgs

        for img_name in self.imgs:
            img_name = img_name[6:11]

    def __getitem__(self, index):
        # 이미지를 로드합니다.
        img_path = osp.join(self.data_dir, self.imgs[index])
        img = load_image(img_path)  
        

        # 변환(transform)이 존재하는 경우 적용합니다.
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.imgs)

