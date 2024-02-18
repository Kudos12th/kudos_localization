import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import torch
import os.path as osp
import numpy as np
import matplotlib
import sys

DISPLAY = 'DISPLAY' in os.environ
if not DISPLAY:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tools.options import Options
from network.atloc import AtLoc
from torchvision import transforms, models
from utils import load_state_dict
from dataloaders import Robocup
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn.metrics import r2_score

# Config
opt = Options().parse()
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

# Model
feature_extractor = models.resnet18(weights=None)
atloc = AtLoc(feature_extractor, droprate=opt.test_dropout, pretrained=False, lstm=opt.lstm)
if opt.model == 'AtLoc':
    model = atloc
else:
    raise NotImplementedError
model.eval()

# loss functions
t_criterion = lambda t_pred, t_gt: np.linalg.norm(t_pred - t_gt)
y_criterion = lambda yaw_pred, yaw_target: abs(yaw_pred - yaw_target)

stats_file = osp.join(opt.data_dir, opt.dataset, opt.scene, 'stats.txt')
stats = np.loadtxt(stats_file)
# transformer
data_transform = transforms.Compose([
    transforms.Resize(opt.cropsize),
    transforms.CenterCrop(opt.cropsize),
    transforms.ToTensor(),
    transforms.Normalize(mean=stats[0], std=np.sqrt(stats[1]))])
target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())

# read mean and stdev for un-normalizing predictions
pose_stats_file = osp.join(opt.data_dir, opt.dataset, opt.scene, 'pose_stats.txt')
pose_m, pose_s = np.loadtxt(pose_stats_file)  # mean and stdev

# Load the dataset
kwargs = dict(scene=opt.scene, data_path=opt.data_dir, train=False, val=False, transform=data_transform, target_transform=target_transform, seed=opt.seed)
if opt.model == 'AtLoc' and opt.dataset == 'Robocup':
        data_set = Robocup(**kwargs)
else:
    raise NotImplementedError

L = len(data_set)
kwargs = {'num_workers': opt.nThreads, 'pin_memory': True} if cuda else {}
loader = DataLoader(data_set, batch_size=1, shuffle=False, **kwargs)

pred_poses = np.zeros((L, 2))  # store all predicted poses
targ_poses = np.zeros((L, 2))  # store all target poses

# load weights
model.to(device)
weights_filename = osp.expanduser(opt.weights)

if osp.isfile(weights_filename):
    checkpoint = torch.load(weights_filename, map_location=device)
    load_state_dict(model, checkpoint['model_state_dict'])
    print('Loaded weights from {:s}'.format(weights_filename))
else:
    print('Could not load weights from {:s}'.format(weights_filename))
    sys.exit(-1)

# inference loop
for idx, (data, pose, yaw, angle) in enumerate(loader):
    if idx % 200 == 0:
        print('Image {:d} / {:d}'.format(idx, len(loader)))

    # output : 1 x 6
    data_var = Variable(data, requires_grad=False)
    data_var = data_var.to(device)

    with torch.set_grad_enabled(False):
        output = model(data_var)
    s = output.size()
    output_pose = output[:, :2].cpu().data.numpy().reshape((-1, s[-1] - 1))
    output_yaw = output[:, 2:].cpu().data.numpy().reshape((-1, 1))
    pose = pose.numpy().reshape((-1, s[-1] - 1))
    yaw = yaw.numpy().reshape((-1, 1))


    # un-normalize the predicted and target translations
    output_pose = (output_pose * pose_s) + pose_m
    pose = (pose * pose_s) + pose_m

    # take the middle prediction
    pred_poses[idx, :] = output_pose[len(output_pose) // 2]
    targ_poses[idx, :] = pose[len(pose) // 2]

# calculate losses

r2_translation = r2_score(targ_poses, pred_poses)
r2_rotation = r2_score(yaw, output_yaw)

print('R2 Score - Translation: {:.4f}'.format(r2_translation))
print('R2 Score - Rotation: {:.4f}'.format(r2_rotation))

fig = plt.figure()
real_pose = (pred_poses - pose_m) / pose_s
gt_pose = (targ_poses - pose_m) / pose_s

plt.plot(gt_pose[:, 1], gt_pose[:, 0], color='black')
plt.plot(real_pose[:, 1], real_pose[:, 0], color='red')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.plot(gt_pose[0, 1], gt_pose[0, 0], 'y*', markersize=15)
plt.show(block=True)

image_filename = osp.join(osp.expanduser(opt.results_dir), '{:s}.png'.format(opt.exp_name))
fig.savefig(image_filename)