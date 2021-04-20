import cv2

def spring2autumn(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img[:,:,1] = 127
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    return img
