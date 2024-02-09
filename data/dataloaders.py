import os
import torch
import numpy as np
import pickle
import os.path as osp

from tools.utils import process_poses, calc_vos_simple, load_image
from torch.utils import data
from functools import partial


class Robocup(data.Dataset):
    def __init__(self, data_path, train, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.data_path = data_path

        all_imgs = [f for f in os.listdir(data_path) if f.endswith('.jpg')]
        np.random.seed(7) 
        np.random.shuffle(all_imgs)

        # test/val 분할
        split_index = int(len(all_imgs) * 0.8)  # 80%가 train
        if train:
            self.imgs = all_imgs[:split_index] # test
        else:
            self.imgs = all_imgs[split_index:] # val

        self.poses = []
        # 파일 이름에서 pose 추출
        for img_name in self.imgs:
            # 파일 이름 : x_y_z.jpg
            pose = np.array(img_name[:-4].split('_'), dtype=np.float32)
            self.poses.append(pose)
        self.poses = np.array(self.poses)

    def __getitem__(self, index):
        # 이미지를 로드합니다.
        img_path = osp.join(self.data_path, self.imgs[index])
        img = load_image(img_path)  # 여기서 `load_image`는 이미지 로드를 위한 적절한 함수입니다.
        # 포즈를 가져옵니다.
        pose = self.poses[index]

        # 변환(transform)이 존재하는 경우 적용합니다.
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            pose = self.target_transform(pose)

        return img, pose

    def __len__(self):
        return len(self.imgs)


class MF(data.Dataset):
    def __init__(self, dataset, include_vos=False, no_duplicates=False, *args, **kwargs):

        self.steps = kwargs.pop('steps', 2)
        self.skip = kwargs.pop('skip', 1)
        self.variable_skip = kwargs.pop('variable_skip', False)
        self.real = kwargs.pop('real', False)
        self.include_vos = include_vos
        self.train = kwargs['train']
        self.vo_func = kwargs.pop('vo_func', calc_vos_simple)
        self.no_duplicates = no_duplicates

        if dataset == 'Robocup':
            self.dset = Robocup(*args, real=self.real, **kwargs)
            if self.include_vos and self.real:
                self.gt_dset = Robocup(*args, skip_images=True, real=False, **kwargs)
        else:
            raise NotImplementedError

        self.L = self.steps * self.skip

    def get_indices(self, index):
        if self.variable_skip:
            skips = np.random.randint(1, high=self.skip+1, size=self.steps-1)
        else:
            skips = self.skip * np.ones(self.steps-1)
        offsets = np.insert(skips, 0, 0).cumsum()
        offsets -= offsets[len(offsets) / 2]
        if self.no_duplicates:
            offsets += self.steps/2 * self.skip
        offsets = offsets.astype(np.int)
        idx = index + offsets
        idx = np.minimum(np.maximum(idx, 0), len(self.dset)-1)
        assert np.all(idx >= 0), '{:d}'.format(index)
        assert np.all(idx < len(self.dset))
        return idx

    def __getitem__(self, index):
        idx = self.get_indices(index)
        clip = [self.dset[i] for i in idx]

        imgs  = torch.stack([c[0] for c in clip], dim=0)
        poses = torch.stack([c[1] for c in clip], dim=0)
        if self.include_vos:
            vos = self.vo_func(poses.unsqueeze(0))[0]
            if self.real:  # absolute poses need to come from the GT dataset
                clip = [self.gt_dset[self.dset.gt_idx[i]] for i in idx]
                poses = torch.stack([c[1] for c in clip], dim=0)
            poses = torch.cat((poses, vos), dim=0)

        return imgs, poses

    def __len__(self):
        L = len(self.dset)
        if self.no_duplicates:
            L -= (self.steps-1)*self.skip
        return L
