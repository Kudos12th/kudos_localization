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

        split_index = int(len(all_imgs) * 0.7)
        split_index2 = int(len(all_imgs) * 0.85)
        
        if train:
            self.imgs = all_imgs[:split_index]  # train
        elif val:
            self.imgs = all_imgs[split_index:split_index2]  # val
        elif scene == 'test_images': #test_images
            self.imgs = all_imgs
        else:
            self.imgs = all_imgs[split_index2:]  # test

        self.poses = []
        self.yaws = []
        self.angles = []

        # 파일 이름 : n_x_y_yaw_angle.jpg
        for img_name in self.imgs:
            img_name = img_name[:-4]
            temp = np.array(img_name.split('_'), dtype=np.float32)

            self.poses.append(temp[1:3])
            self.yaws.append(temp[3])
            self.angles.append(temp[4])

        self.poses = np.array(self.poses)
        self.yaws = np.array(self.yaws)
        self.angles = np.array(self.angles)
        
        if train:
            pose_stats = np.empty((0,2))
            for p in self.poses:
                pose_stats = np.vstack((pose_stats, p))
            pose_stats_filename = f"{self.data_dir}/pose_stats.txt"
            mean_t = np.mean(self.poses, axis=0)
            std_t = np.std(self.poses, axis=0)
            np.savetxt(pose_stats_filename, np.vstack((mean_t, std_t)), fmt='%8.7f')
        

    def __getitem__(self, index):
        # 이미지를 로드합니다.
        img_path = osp.join(self.data_dir, self.imgs[index])
        img = load_image(img_path)  
        
        # 포즈를 가져옵니다.
        pose = self.poses[index]
        yaw = self.yaws[index]
        angle = self.angles[index]

        # 변환(transform)이 존재하는 경우 적용합니다.
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            pose = self.target_transform(pose)

        return img, pose, yaw, angle

    def __len__(self):
        return len(self.imgs)



class MF(data.Dataset):
    def __init__(self, dataset, no_duplicates=False, *args, **kwargs):

        self.steps = kwargs.pop('steps', 2)
        self.skip = kwargs.pop('skip', 1)
        self.variable_skip = kwargs.pop('variable_skip', False)
        self.real = kwargs.pop('real', False)
        self.train = kwargs['train']
        self.no_duplicates = no_duplicates

        if dataset == 'Robocup':
            self.dset = Robocup(*args, real=self.real, **kwargs)
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
        yaws = torch.stack([c[2] for c in clip], dim=0)
        angles = torch.stack([c[3] for c in clip], dim=0)
        
        return imgs, poses, yaws, angles

    def __len__(self):
        L = len(self.dset)

        if self.no_duplicates:
            L -= (self.steps-1)*self.skip
        return L
