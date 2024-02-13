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




def resize_and_pad(image, desired_size):
    old_size = image.shape[:2] 
    ratio = float(desired_size/max(old_size))
    new_size = tuple([int(x*ratio) for x in old_size])
    
    # new_size should be in (width, height) format
    
    image = cv2.resize(image, (new_size[1], new_size[0]))
    
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    
    pad = (delta_w, delta_h)
    
    color = [100, 100, 100]
    new_im = cv2.copyMakeBorder(image, 0, delta_h, 0, delta_w, cv2.BORDER_CONSTANT,
        value=color)
        
    return new_im, pad

def get_image_tensor(img, max_size, debug=False):
    """
    Reshapes an input image into a square with sides max_size
    """
    if type(img) is str:
        img = cv2.imread(img)
    
    resized, pad = resize_and_pad(img, max_size)
    resized = resized.astype(np.float32)
    
    if debug:
        cv2.imwrite("intermediate.png", resized)

    # Normalise!
    resized /= 255.0
    
    return img, resized, pad





if __name__ == "__main__":
 
    priROS = priROS()
    
    # Load the Edge TPU model
    interpreter = edgetpu.make_interpreter(opt.weights)
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
                
                if interpreter is not None:
                    input_size = common.input_size(interpreter)
                    logger.debug("Expecting input shape: {}".format(input_size))
                    input_size = input_size
                else:
                    logger.warn("Interpreter is not yet loaded")
                    continue
                
                full_image, net_image, pad = get_image_tensor(image, input_size[0])

                # Preprocess image for Edge TPU model
                resized_image = cv2.resize(image, (224, 224))
                input_data = np.expand_dims(resized_image, axis=0).astype(np.float32)
                input_data = input_data.transpose((0, 3, 1, 2))

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
