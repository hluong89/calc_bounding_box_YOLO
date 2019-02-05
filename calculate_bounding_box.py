import cv2 as cv2
import os
import numpy as np

img = cv2.imread('6_apple.jpg')
#img = cv2.imread('apple.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img,127,255,0)

contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#contours, hierarchy = cv2.findContours(thresh,1,2)
cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
#cnt = contours[1]
#cv2.drawContours(img, [cnt], 0, (0, 255, 0), 3)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# for item in range(len(contours)):
#     cnt = contours[item]
#     if len(cnt)>20:
#         print(len(cnt))
#         M = cv2.moments(cnt)
#         cx = int(M['m10']/M['m00'])
#         cy = int(M['m01']/M['m00'])
#         x,y,w,h = cv2.boundingRect(cnt)
#         cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
#         cv2.imshow('image',img)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()