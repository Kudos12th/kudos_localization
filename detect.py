import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

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
# Import the Coral Edge TPU dependencies
import pycoral.utils.edgetpu as edgetpu
from pycoral.adapters import common

# Config
opt = Options().parse()

class priROS:
    def __init__(self):
        rospy.init_node('kudos_vision', anonymous=False)
        self.pose_pub = rospy.Publisher("/Odometry", Odometry, queue_size=10)
        self.angle_pub = rospy.Publisher('/dxl_ang', Float64, queue_size=10)

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
    
    # Load the Edge TPU model
    model_path = "path/to/your/model.tflite"
    interpreter = edgetpu.make_interpreter(model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    logger.info("Opening stream on device: {}".format(opt.cam))
    constant = 40800
    
    cam = cv2.VideoCapture(opt.cam)
    
    start_time = time.time()  # Record the start time

    while True:  # Run the loop for 20 seconds
        try:
            res, image = cam.read()
            height, width = image.shape[:2]
            if res is False:
                logger.error("Empty image received")
                break
            else:
                # Resize image to match model input shape if needed
                resized_image = cv2.resize(image, (input_details[0]['shape'][2], input_details[0]['shape'][1]))

                # Preprocess image for Edge TPU model
                input_data = np.expand_dims(resized_image, axis=0)
                input_data = common.input_image(input_data)
                interpreter.set_tensor(input_details[0]['index'], input_data)

                # Run inference
                interpreter.invoke()

                # Get the output
                output_pose = interpreter.get_tensor(output_details[0]['index'])
                output_yaw = interpreter.get_tensor(output_details[1]['index'])

                # Post-process and handle the output as needed
                # For example, publish to ROS topics
                priROS.pose_talker(output_pose, 0)  # Replace 0 with actual FPS
                priROS.angle_talker(output_yaw)

                logger.info("Frame done")

        except KeyboardInterrupt:
            cam.release()   
            break

    cam.release()
