import cv2
import numpy as np

# 이미지 파일을 읽어옵니다. 여기서는 'brownpurple.jpg' 파일을 사용합니다.
img = cv2.imread('brownpurple.jpg', cv2.IMREAD_COLOR)

# 갈색과 보라색 부분을 회색으로 바꾸기 위해 갈색과 보라색의 마스크를 만듭니다.
brown_lower = np.array([5, 55, 150], dtype=np.uint8)  # 갈색의 범위
brown_upper = np.array([40, 255, 255], dtype=np.uint8)
purple_lower = np.array([120, 50, 50], dtype=np.uint8)  # 보라색의 범위
purple_upper = np.array([160, 255, 255], dtype=np.uint8)

# 이미지를 HSV 색 공간으로 변환합니다.
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 갈색, 보라색에 해당하는 영역을 마스킹합니다.
brown_mask = cv2.inRange(hsv_img, brown_lower, brown_upper)
purple_mask = cv2.inRange(hsv_img, purple_lower, purple_upper)

# 갈색, 보라색 영역을 회색으로 바꿉니다.
gray_img = img.copy()
gray_img[np.logical_or(brown_mask != 0, purple_mask != 0)] = [128, 128, 128]

# 결과 이미지를 화면에 표시합니다.
cv2.imshow('Gray Image', gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('Gray Image.jpg', gray_img)
