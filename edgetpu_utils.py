import numpy as np
import cv2

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

