import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry 
from std_msgs.msg import String
import math
import cv2
import os
def euler_from_quaternion(x, y, z, w):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return yaw_z # in radians

class ImageSubscriberNode:
    def __init__(self):
        rospy.init_node('image_subscriber_node', anonymous=True)

        # AMCL 토픽에서 로봇의 위치 메시지를 구독합니다.
        rospy.Subscriber('/Odometry', Odometry, self.odom_callback)
        rospy.Subscriber('/dxl_ang',String , self.angle_callback)

        # OpenCV 초기화
        self.bridge = CvBridge()

        # 웹캠 열기
        self.cap = cv2.VideoCapture(0)

        # 작업 디렉토리에 'received_images' 폴더 생성
        self.output_folder = 'received_images'
        os.makedirs(self.output_folder, exist_ok=True)

        # 콜백에서 받은 값 저장을 위한 변수
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_z = 0.0
        self.ori_x = 0.0
        self.ori_y = 0.0
        self.ori_z = 0.0
        self.ori_w = 0.0
        self.yaw = 0.0
        self.angle = 0.0
        self.n = 0

    def odom_callback(self, odom_msg):
        # Odometry 메시지에서 x, y, z, w 좌표를 추출하여 변수에 저장
        self.odom_x = odom_msg.pose.pose.position.x
        self.odom_y = odom_msg.pose.pose.position.y
        self.odom_z = odom_msg.pose.pose.position.z
        
        self.ori_x = odom_msg.pose.pose.orientation.x
        self.ori_y = odom_msg.pose.pose.orientation.y
        self.ori_z = odom_msg.pose.pose.orientation.z
        self.ori_w = odom_msg.pose.pose.orientation.w
        self.yaw = euler_from_quaternion(self.ori_x,self.ori_y,self.ori_z,self.ori_w)

    def angle_callback(self, angle_msg):
        self.angle = angle_msg.data
        image_subscriber_node.save_image()
        print(self.angle)
 
    
    def save_image(self):
        try:
            # 프레임 읽기
            self.n+=1
            ret, frame = self.cap.read()

            # 받은 좌표를 파일 이름으로 사용하여 이미지 저장
            image_filename = f"{self.output_folder}/{self.n}_"
            image_filename += f"{self.odom_x}_{self.odom_y}_{self.yaw}_{self.angle}.jpg"

            cv2.imwrite(image_filename, frame)

            rospy.loginfo(f"Saved image: {image_filename}")

        except CvBridgeError as e:
            rospy.logerr(e)

if __name__ == "__main__":
    try:
        image_subscriber_node = ImageSubscriberNode()
        
        # rospy.Rate를 사용하여 이미지 저장 주기 설정 (예: 1초에 한 번)
        rate = rospy.Rate(50)

        while not rospy.is_shutdown():
            #image_subscriber_node.save_image()
            rate.sleep()

    except rospy.ROSInterruptException:
        pass
