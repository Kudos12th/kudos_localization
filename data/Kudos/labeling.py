import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point, Quaternion
from tf.transformations import euler_from_quaternion
import cv2
import os

class ImageSubscriberNode:
    def __init__(self):
        rospy.init_node('image_subscriber_node', anonymous=True)

        # 이미지 구독자 초기화
        rospy.Subscriber('image_coordinates_quaternion', PointQuaternion, self.image_callback)

        # OpenCV 초기화
        self.bridge = CvBridge()

        # 웹캠 열기
        self.cap = cv2.VideoCapture(0)

        # 작업 디렉토리에 'received_images' 폴더 생성
        self.output_folder = 'received_images'
        os.makedirs(self.output_folder, exist_ok=True)

    def image_callback(self, image_info):
        try:
            # 프레임 읽기
            ret, frame = self.cap.read()

            # 쿼터니언을 오일러 앵글로 변환
            quaternion = image_info.quaternion
            euler_angles = euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w])

            # 받은 좌표와 오일러 앵글로 이미지 저장
            image_filename = f"{self.output_folder}/received_image_{int(image_info.coordinates.x)}_{int(image_info.coordinates.y)}_"
            image_filename += f"Roll_{int(euler_angles[0])}_Pitch_{int(euler_angles[1])}_Yaw_{int(euler_angles[2])}.jpg"
            
            cv2.imwrite(image_filename, frame)

            rospy.loginfo(f"Received coordinates: ({image_info.coordinates.x}, {image_info.coordinates.y}), "
                          f"Quaterion: {quaternion}, "
                          f"Saved image: {image_filename}")

        except CvBridgeError as e:
            rospy.logerr(e)

if __name__ == "__main__":
    try:
        image_subscriber_node = ImageSubscriberNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
