import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry  # 수정된 부분
import cv2
import os

class ImageSubscriberNode:
    def __init__(self):
        rospy.init_node('image_subscriber_node', anonymous=True)

        # AMCL 토픽에서 로봇의 위치 메시지를 구독합니다.
        rospy.Subscriber('/move_base_simple/goal', Pose, self.move_base_goal_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)

        # OpenCV 초기화
        self.bridge = CvBridge()

        # 웹캠 열기
        self.cap = cv2.VideoCapture(0)

        # 작업 디렉토리에 'received_images' 폴더 생성
        self.output_folder = 'received_images'
        os.makedirs(self.output_folder, exist_ok=True)

        # 콜백에서 받은 값 저장을 위한 변수
        self.move_base_x = 0.0
        self.move_base_y = 0.0
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_z = 0.0
        self.odom_w = 0.0

    def move_base_goal_callback(self, move_base_goal_msg):
        # MoveBase의 목표 위치 메시지에서 x, y 좌표를 추출하여 변수에 저장
        self.move_base_x = move_base_goal_msg.position.x
        self.move_base_y = move_base_goal_msg.position.y

    def odom_callback(self, odom_msg):
        # Odometry 메시지에서 x, y, z, w 좌표를 추출하여 변수에 저장
        self.odom_x = odom_msg.pose.pose.position.x
        self.odom_y = odom_msg.pose.pose.position.y
        self.odom_z = odom_msg.pose.pose.orientation.z
        self.odom_w = odom_msg.pose.pose.orientation.w

    def save_image(self):
        try:
            # 프레임 읽기
            ret, frame = self.cap.read()

            # 받은 좌표를 파일 이름으로 사용하여 이미지 저장
            image_filename = f"{self.output_folder}/{self.move_base_x}_{self.move_base_y}_"
            image_filename += f"{self.odom_x}_{self.odom_y}_{self.odom_z}_{self.odom_w}.jpg"

            cv2.imwrite(image_filename, frame)

            rospy.loginfo(f"Received MoveBase goal: ({self.move_base_x}, {self.move_base_y}), "
                          f"Odometry data: ({self.odom_x}, {self.odom_y}, {self.odom_z}, {self.odom_w}), "
                          f"Saved image: {image_filename}")

        except CvBridgeError as e:
            rospy.logerr(e)

if __name__ == "__main__":
    try:
        image_subscriber_node = ImageSubscriberNode()
        
        # rospy.Rate를 사용하여 이미지 저장 주기 설정 (예: 1초에 한 번)
        rate = rospy.Rate(1)

        while not rospy.is_shutdown():
            image_subscriber_node.save_image()
            rate.sleep()

    except rospy.ROSInterruptException:
        pass
