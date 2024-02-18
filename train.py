import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import torch
import sys
import time
import os.path as osp
import numpy as np

from tensorboardX import SummaryWriter
from tools.options import Options
from network.atloc import AtLoc
from torchvision import transforms, models
from utils import AtLocCriterion, AverageMeter, Logger, load_state_dict
from dataloaders import Robocup
from torch.utils.data import DataLoader
from torch.autograd import Variable

# Config
opt = Options().parse()
cuda = torch.cuda.is_available()
device = "cuda:" + ",".join(str(i) for i in opt.gpus) if cuda else "cpu"
logfile = osp.join(opt.runs_dir, 'log.txt')
stdout = Logger(logfile)
print('Logging to {:s}'.format(logfile))
sys.stdout = stdout

# Model
feature_extractor = models.resnet18(weights=None)
atloc = AtLoc(feature_extractor, droprate=opt.train_dropout, pretrained=True, lstm=opt.lstm)
if opt.model == 'AtLoc':
    model = atloc
    train_criterion = AtLocCriterion(saq=opt.beta, learn_beta=True)
    val_criterion = AtLocCriterion()
    param_list = [{'params': model.parameters()}]
else:
    raise NotImplementedError

# Optimizer
param_list = [{'params': model.parameters()}]
if hasattr(train_criterion, 'sax') and hasattr(train_criterion, 'saq'):
    print('learn_beta')
    param_list.append({'params': [train_criterion.sax, train_criterion.saq]})
optimizer = torch.optim.Adam(param_list, lr=opt.lr, weight_decay=opt.weight_decay)

stats_file = osp.join(opt.data_dir, opt.dataset, opt.scene, 'stats.txt')
stats = np.loadtxt(stats_file)

tforms = [transforms.Resize(opt.cropsize)]
tforms.append(transforms.RandomCrop(opt.cropsize))

if opt.color_jitter > 0:
    assert opt.color_jitter <= 1.0
    print('Using ColorJitter data augmentation')
    tforms.append(transforms.ColorJitter(brightness=opt.color_jitter, contrast=opt.color_jitter, saturation=opt.color_jitter, hue=0.5))
else:
    print('Not Using ColorJitter')

tforms.append(transforms.ToTensor())
tforms.append(transforms.Normalize(mean=stats[0], std=np.sqrt(stats[1])))

data_transform = transforms.Compose(tforms)
target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())

# Load the dataset
kwargs = dict(scene=opt.scene, data_path=opt.data_dir, transform=data_transform, target_transform=target_transform, seed=opt.seed)
robocup_kwargs = {k: kwargs[k] for k in ['data_path', 'transform', 'target_transform', 'scene'] if k in kwargs}

if opt.model == 'AtLoc' and opt.dataset == 'Robocup':
    train_set = Robocup(train=True, val=False,**robocup_kwargs)
    val_set = Robocup(train=False, val=True, **robocup_kwargs)
else:
    raise NotImplementedError

kwargs = {'num_workers': opt.nThreads, 'pin_memory': True} if cuda else {}
train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, **kwargs)
val_loader = DataLoader(val_set, batch_size=opt.batch_size, shuffle=False, **kwargs)

model.to(device)
train_criterion.to(device)
val_criterion.to(device)

total_steps = opt.steps
writer = SummaryWriter(log_dir=opt.runs_dir)
experiment_name = opt.exp_name

# load weights
if opt.weights:
    weights_filename = osp.expanduser(opt.weights)
    if osp.isfile(weights_filename):
        checkpoint = torch.load(weights_filename, map_location=device)
        load_state_dict(model, checkpoint['model_state_dict'])
        print('Loaded weights from {:s}'.format(weights_filename))
    else:
        print('Could not load weights from {:s}'.format(weights_filename))
        sys.exit(-1)

# 학습 중에 손실 기록
train_loss_list = []  # 학습 손실 기록
val_loss_list = []  # 검증 손실 기록

for epoch in range(opt.start_epochs, opt.epochs):
    if epoch % opt.val_freq == 0 or epoch == (opt.epochs - 1):
        val_batch_time = AverageMeter()
        val_loss = AverageMeter()
        model.eval()
        end = time.time()
        val_data_time = AverageMeter()

        for batch_idx, (val_data, val_pose, val_yaw, val_angle) in enumerate(val_loader):
            val_data_time.update(time.time() - end)

            val_data_var = Variable(val_data, requires_grad=False)
            val_pose_var = Variable(val_pose, requires_grad=False)
            val_yaw_var = Variable(val_yaw, requires_grad=False)

            val_data_var = val_data_var.to(device)
            val_pose_var = val_pose_var.to(device)
            val_yaw_var = val_yaw_var.to(device)

            with torch.set_grad_enabled(False):
                val_output = model(val_data_var)
                val_loss_tmp = val_criterion(val_output, val_pose_var, val_yaw_var)
                val_loss_tmp = val_loss_tmp.item()

            val_loss.update(val_loss_tmp)
            val_batch_time.update(time.time() - end)

            writer.add_scalar('val_err', val_loss_tmp, total_steps)
            if batch_idx % opt.print_freq == 0:
                print('Val {:s}: Epoch {:d}\tBatch {:d}/{:d}\tData time {:.4f} ({:.4f})\tBatch time {:.4f} ({:.4f})\tLoss {:f}' \
                      .format(experiment_name, epoch, batch_idx, len(val_loader) - 1, val_data_time.val, val_data_time.avg, val_batch_time.val, val_batch_time.avg, val_loss_tmp))
            end = time.time()

        print('Val {:s}: Epoch {:d}, val_loss {:f}'.format(experiment_name, epoch, val_loss.avg))

        # 검증 손실 기록
        val_loss_list.append(val_loss.avg)  # 검증 손실 기록

        if epoch % opt.save_freq == 0:
            filename = osp.join(opt.models_dir, 'epoch_{:03d}.pth.tar'.format(epoch))
            checkpoint_dict = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optim_state_dict': optimizer.state_dict(), 'criterion_state_dict': train_criterion.state_dict()}
            torch.save(checkpoint_dict, filename)
            print('Epoch {:d} checkpoint saved for {:s}'.format(epoch, experiment_name))

    model.train()
    train_data_time = AverageMeter()
    train_batch_time = AverageMeter()
    end = time.time()

    for batch_idx, (data, pose, yaw, angle) in enumerate(train_loader):
        train_data_time.update(time.time() - end)

        data_var = Variable(data, requires_grad=True)
        pose_var = Variable(pose, requires_grad=False)
        yaw_var = Variable(yaw, requires_grad=False)

        data_var = data_var.to(device)
        pose_var = pose_var.to(device)
        yaw_var = yaw_var.to(device)

        with torch.set_grad_enabled(True):
            output = model(data_var)
            loss_tmp = train_criterion(output, pose_var, yaw_var)

        loss_tmp.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_batch_time.update(time.time() - end)
        writer.add_scalar('train_err', loss_tmp.item(), total_steps)
        
        # 학습 중에 손실 기록
        train_loss_list.append(loss_tmp.item())  # 학습 손실 기록

        if batch_idx % opt.print_freq == 0:
            print('Train {:s}: Epoch {:d}\tBatch {:d}/{:d}\tData time {:.4f} ({:.4f})\tBatch time {:.4f} ({:.4f})\tLoss {:f}' \
                  .format(experiment_name, epoch, batch_idx, len(train_loader) - 1, train_data_time.val, train_data_time.avg, train_batch_time.val, train_batch_time.avg, loss_tmp.item()))
        end = time.time()
        # 학습이 끝난 후에 손실 그래프 시각화
import matplotlib.pyplot as plt

# 에폭 별 평균 손실 계산
epoch_losses = []
for epoch in range(opt.start_epochs, opt.epochs):
    epoch_loss = sum(train_loss_list[epoch * len(train_loader): (epoch + 1) * len(train_loader)]) / len(train_loader)
    epoch_losses.append(epoch_loss)

# 손실 그래프 생성 및 저장
plt.plot(epoch_losses, label='Train Loss')
plt.plot(val_loss_list, label='Validation Loss')  # 검증 손실 그래프 추가
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('train_val_loss_graph.png')
plt.show()
writer.close()