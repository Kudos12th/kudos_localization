import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import torch
import os.path as osp
import numpy as np
import matplotlib
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

from tools.options import Options

from edgetpu.edgetpu_model import EdgeTPUModel
from edgetpu.edgetpu_utils import get_image_tensor

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
    
    model = EdgeTPUModel(opt.weights)
    input_size = model.get_image_size()

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
                total_times = []
                
                input_size = model.get_image_size()
                full_image, net_image, pad = get_image_tensor(image, input_size[0])
                output = model.forward(net_image) 
                
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