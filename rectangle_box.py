import cv2 as cv2

#img = cv2.imread('cat.png')
#img = cv2.imread('image_test.png')
img = cv2.imread('apple1.jpg')
#img = cv2.imread('potato1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img,127,255,0)

contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#contours, hierarchy = cv2.findContours(thresh,1,2)
# cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
# cnt = contours[1]
# cv2.drawContours(img, [cnt], 0, (0, 255, 0), 3)

#sort contours
#cnts = contours[0] if imutils.is_cv2() else contours[1]
sorted_contours = sorted(contours, key=cv2.contourArea,  reverse=True)

#sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0] + cv2.boundingRect(ctr)[1] * img.shape[1] )
# x + y * w

#cnt = contours[0]
# cnt = sorted_contours[1]
# x,y,w,h = cv2.boundingRect(cnt)
# cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
#
# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
for item in range(len(contours)):
    cnt = sorted_contours[item]
    if len(cnt)>10:
        print(len(cnt))
        # M = cv2.moments(cnt)
        # cx = int(M['m10']/M['m00'])
        # cy = int(M['m01']/M['m00'])
        x,y,w,h = cv2.boundingRect(cnt)
        if w > 10 and h > 10:
            print(x,y,w,h)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.imshow('image',img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()



# print(len(contours))
# cv2.drawContours(img, contours, -1, (255, 255, 0), 1)
#
# cv2.imshow("contours", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()