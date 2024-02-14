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
    
# 모델을 추론 모드로 전환합니다
model.eval()


# 모델에 대한 입력값
x = torch.randn(batch_size, 3, 224, 224, requires_grad=True)
torch_out = model(x)

# 모델 변환
torch.onnx.export(model,               # 실행될 모델
                  x,                         # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
                  "super_resolution.onnx",   # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
                  export_params=True,        # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
                  opset_version=10,          # 모델을 변환할 때 사용할 ONNX 버전
                  do_constant_folding=True,  # 최적화시 상수폴딩을 사용할지의 여부
                  input_names = ['input'],   # 모델의 입력값을 가리키는 이름
                  output_names = ['output'], # 모델의 출력값을 가리키는 이름
                  dynamic_axes={'input' : {0 : 'batch_size'},    # 가변적인 길이를 가진 차원
                                'output' : {0 : 'batch_size'}})