import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseWithCovarianceStamped
import cv2
import os

class ImageSubscriberNode:
    def __init__(self):
        rospy.init_node('image_subscriber_node', anonymous=True)

        # AMCL 토픽에서 로봇의 위치 메시지를 구독합니다.
        rospy.Subscriber('amcl_pose', PoseWithCovarianceStamped, self.amcl_pose_callback)

        # OpenCV 초기화
        self.bridge = CvBridge()

        # 웹캠 열기
        self.cap = cv2.VideoCapture(0)

        # 작업 디렉토리에 'received_images' 폴더 생성
        self.output_folder = 'received_images'
        os.makedirs(self.output_folder, exist_ok=True)

    def amcl_pose_callback(self, amcl_pose_msg):
        try:
            # 프레임 읽기
            ret, frame = self.cap.read()

            # 로봇의 위치 메시지에서 x, y 좌표 및 orientation의 x, y, z, w 값을 추출
            x = amcl_pose_msg.pose.pose.position.x
            y = amcl_pose_msg.pose.pose.position.y
            orientation_x = amcl_pose_msg.pose.pose.orientation.x
            orientation_y = amcl_pose_msg.pose.pose.orientation.y
            orientation_z = amcl_pose_msg.pose.pose.orientation.z
            orientation_w = amcl_pose_msg.pose.pose.orientation.w

            # 받은 좌표와 orientation을 파일 이름으로 사용하여 이미지 저장
            image_filename = f"{self.output_folder}/received_image_x_{int(x)}_y_{int(y)}_"
            image_filename += f"orient_x_{int(orientation_x)}_orient_y_{int(orientation_y)}_orient_z_{int(orientation_z)}_orient_w_{int(orientation_w)}.jpg"
            
            cv2.imwrite(image_filename, frame)

            rospy.loginfo(f"Received AMCL pose: ({x}, {y}), "
                          f"Orientation: ({orientation_x}, {orientation_y}, {orientation_z}, {orientation_w}), "
                          f"Saved image: {image_filename}")

        except CvBridgeError as e:
            rospy.logerr(e)

if __name__ == "__main__":
    try:
        image_subscriber_node = ImageSubscriberNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
