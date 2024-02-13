import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import torch
import os.path as osp
import numpy as np
import matplotlib
import sys
from PIL import Image


DISPLAY = 'DISPLAY' in os.environ
if not DISPLAY:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import logging
import time
import rospy
import cv2
import numpy as np

from nav_msgs.msg import Odometry 
from geometry_msgs.msg import Point
from std_msgs.msg import Float64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from tools.options import Options
from network.atloc import AtLoc
from torchvision import transforms, models
from utils import load_state_dict
from dataloaders import Robocup
from torch.utils.data import DataLoader
from torch.autograd import Variable

from edgetpu_model import EdgeTPUModel
from edgetpu_utils import resize_and_pad, get_image_tensor, save_one_json, coco80_to_coco91_class, StreamingDataProcessor

# Config
opt = Options().parse()
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

class priROS:
    def __init__(self):
        rospy.init_node('kudos_vision', anonymous = False)
        self.pose_pub = rospy.Publisher("/Odometry", Odometry, queue_size = 10)
        self.angle_pub = rospy.Publisher('/dxl_ang', Float64, queue_size = 10)

    def pose_talker(self, pose, fps):
        print("Mean FPS: {:1.2f}".format(fps))
        msg = Odometry()
        msg.header.stamp = rospy.Time.now()
        print(pose)
        msg.pose.pose.position = Point(pose[0][0], pose[0][1], 0.0)
        self.pose_pub.publish(msg)

    def angle_talker(self, angle):
        msg = Float64()
        msg.data = angle
        self.angle_pub.publish(msg)


if __name__ == "__main__":
 
    priROS = priROS()
    
    # Model
    feature_extractor = models.resnet18(weights=None)
    atloc = AtLoc(feature_extractor, droprate=opt.test_dropout, pretrained=False, lstm=opt.lstm)
    if opt.model == 'AtLoc':
        model = atloc
    else:
        raise NotImplementedError
    model.eval()


    stats_file = osp.join(opt.data_dir, opt.dataset, opt.scene, 'stats.txt')
    stats = np.loadtxt(stats_file)
    # transformer
    data_transform = transforms.Compose([
        transforms.Resize(opt.cropsize),
        transforms.CenterCrop(opt.cropsize),
        transforms.ToTensor(),
        transforms.Normalize(mean=stats[0], std=np.sqrt(stats[1]))])
    target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())

    # TODO: pose_stats??
    # read mean and stdev for un-normalizing predictions
    pose_stats_file = osp.join(opt.data_dir, opt.dataset, opt.scene, 'pose_stats.txt')
    pose_m, pose_s = np.loadtxt(pose_stats_file)  # mean and stdev

    # Load the dataset
    kwargs = dict(scene=opt.scene, data_path=opt.data_dir, train=False, transform=data_transform, target_transform=target_transform, seed=opt.seed)
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
        
    logger.info("Opening stream on device: {}".format(opt.cam))
    constant = 40800
    
    cam = cv2.VideoCapture(opt.cam)
    
    start_time = time.time()  # Record the start time


    while True:  # Run the loop for 20 seconds
        try:
            res, image = cam.read()
            height, width = image.shape[:2]
            # new_img_size = (width, width)
            if res is False:
                logger.error("Empty image received")
                break
            else:
                
                # OpenCV 이미지를 PIL 이미지로 변환
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                # PIL 이미지를 PyTorch 텐서로 변환
                transform = transforms.ToTensor()
                image_tensor = transform(pil_image)

                # 모델에 입력으로 사용하기 위해 차원 추가
                image_tensor = image_tensor.unsqueeze(0)

                total_times = []

                data_var = Variable(image_tensor, requires_grad=False)
                data_var = data_var.to(device)

                with torch.set_grad_enabled(False):
                    output = model(data_var)    
                
                tinference = model.get_last_inference_time()
                total_times=np.append(total_times,tinference)
                total_times = np.array(total_times)
                fps = 1.0/total_times.mean()

                s = output.size()
                output_pose = output[:, :2].cpu().data.numpy().reshape((-1, s[-1] - 1))
                output_yaw = output[:, 2:].cpu().data.numpy().reshape((-1, 1))

                priROS.pose_talker(output_pose, fps)
                priROS.angle_talker(output_yaw)
                tinference = model.get_last_inference_time()
                logger.info("Frame done in {}".format(tinference))
                
        except KeyboardInterrupt:
            cam.release()   
            break

    cam.release()