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

from test_dataloader import Robocup
from tools.options import Options
from network.atloc import AtLoc
from torchvision import transforms, models
from utils import load_state_dict
from torch.utils.data import DataLoader
from torch.autograd import Variable


# Config
opt = Options().parse()
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

class ROSImageProcessor:
    def __init__(self):
        rospy.init_node('ros_image_processor', anonymous=True)
        self.pose_pub = rospy.Publisher("/Odometry", Odometry, queue_size=10)
        self.angle_pub = rospy.Publisher('/dxl_ang', Float64, queue_size=10)

    def pose_talker(self, pose):
        msg = Odometry()
        msg.header.stamp = rospy.Time.now()
        print("Pose :", pose)
        msg.pose.pose.position = Point(pose[0][0], pose[0][1], 0.0)
        self.pose_pub.publish(msg)

    def angle_talker(self, angle):
        msg = Float64()
        print("Angle :", angle)
        print()
        msg.data = angle
        self.angle_pub.publish(msg)  
    
def extract_frames(video_path, frame_rate=30):
    cap = cv2.VideoCapture(video_path)
    # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 지정된 프레임 레이트로 프레임 추출
    interval = fps // frame_rate
    frame_number = 0
    output_folder = os.path.join(os.path.dirname(video_path), "test_images")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_number % interval == 0:
            # 이미지 저장
            image_path = os.path.join(output_folder, f"frame_{frame_number // interval:04d}.jpg")
            cv2.imwrite(image_path, frame)

        frame_number += 1

    # 종료
    cap.release()


if __name__ == "__main__":
    ros_image_processor = ROSImageProcessor()
    
    video_path = 'kudos_localization/data/Robocup/test_video/TalkMedia_talkv_high.mp4.mp4'
    extract_frames(video_path, frame_rate=30)

    print("프레임 추출이 완료되었습니다.")

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
        
    logger.info("Opening stream on device: {}".format(opt.cam))
    constant = 40800
    

    start_time = time.time()  # Reco

    # inference loop
    for idx, data in enumerate(loader):
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

        output_pose = output[:, :2].cpu().data.numpy().reshape((-1, s[-1] - 1))
        output_yaw = output[:, 2:].cpu().data.numpy().reshape((-1, 1))

        ros_image_processor.pose_talker(output_pose)
        ros_image_processor.angle_talker(output_yaw)
        tinference = model.get_last_inference_time()
        logger.info("Frame done in {}".format(tinference))
                    