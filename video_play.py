import numpy as np
import cv2

cap = capture =cv2.VideoCapture('video1_out.mp4')
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
    	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    	cv2.imshow('frame',gray)
    	cv2.waitKey(25)
    else: break

cap.release()
cv2.destroyAllWindows()
