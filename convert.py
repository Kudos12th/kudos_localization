import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import torch
import os.path as osp
import sys

import torch.onnx
from tools.options import Options
from torchvision import models
from network.atloc import AtLoc

from utils import load_state_dict

opt = Options().parse()
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

batch_size = 1    # 임의의 수

feature_extractor = models.resnet18(weights=None)
atloc = AtLoc(feature_extractor, droprate=opt.train_dropout, pretrained=True, lstm=opt.lstm)

model = atloc

# 모델을 미리 학습된 가중치로 초기화합니다
map_location = lambda storage, loc: storage
if torch.cuda.is_available():
    map_location = None

weights_filename = osp.expanduser(opt.weights)

if osp.isfile(weights_filename):
    checkpoint = torch.load(weights_filename, map_location=device)
    load_state_dict(model, checkpoint['model_state_dict'])
    print('Loaded weights from {:s}'.format(weights_filename))
else:
    print('Could not load weights from {:s}'.format(weights_filename))
    sys.exit(-1)
    
model.eval()  # 모델 전체를 평가 모드로 설정


# 드롭아웃 레이어를 평가 모드로 설정
for module in model.modules():
    if isinstance(module, torch.nn.modules.Dropout):
        module.train(False)

# 모델에 대한 입력값
x = torch.randn(batch_size, 3, 224, 224, requires_grad=True)
torch_out = model(x)

# 모델 변환
torch.onnx.export(model,
                  x,
                  f"{weights_filename}.onnx",
                  export_params=True,
                  opset_version=10,
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'},
                                'output': {0: 'batch_size'}})
