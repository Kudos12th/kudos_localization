import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import torch
import os.path as osp
import numpy as np
import sys
from PIL import Image

import logging
import time
import rospy
import cv2
import numpy as np

from nav_msgs.msg import Odometry 
from geometry_msgs.msg import Point
from std_msgs.msg import Float64

from sensor_msgs.msg import CompressedImage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from tools.options import Options
from network.atloc import AtLoc
from torchvision import transforms, models
from utils import load_state_dict
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
        rospy.Subscriber("/output/image_raw2/compressed", CompressedImage, self.image_callback)
        self.image_received = False
        self.image = None

    def image_callback(self, compressed_msg):
        np_arr = np.fromstring(compressed_msg.data, np.uint8)
        self.image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.image_received = True

    def get_image(self):
        return self.imageos_image_processor

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
    ros_image_processor = ROSImageProcessor()
    
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

    # read mean and stdev for un-normalizing predictions
    pose_stats_file = osp.join(opt.data_dir, opt.dataset, opt.scene, 'pose_stats.txt')
    pose_m, pose_s = np.loadtxt(pose_stats_file)  # mean and stdev

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
    

    start_time = time.time()  # Record the start time


    while True:  # Run the loop for 20 seconds
        try:
            while not rospy.is_shutdown():
                current_image = ros_image_processor.get_image()
                height, width = current_image.shape[:2]
                # new_img_size = (width, width)
                if current_image is None:
                    logger.error("Empty image received")
                    break
                else:
                    # OpenCV 이미지를 PIL 이미지로 변환
                    pil_image = Image.fromarray(cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB))

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

                    ros_image_processor.pose_talker(output_pose, fps)
                    ros_image_processor.angle_talker(output_yaw)
                    tinference = model.get_last_inference_time()
                    logger.info("Frame done in {}".format(tinference))
                    
        except KeyboardInterrupt:
              
            break

  