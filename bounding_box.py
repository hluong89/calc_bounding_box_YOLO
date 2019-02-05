
import cv2 as cv
import numpy as np


img_name = ['kiwi1.jpg', 'kiwi2.jpg','apple1.jpg', 'apple2.jpg', 'banana1.jpg','banana2.jpg', 'lemon1.jpg', 'potato1.jpg',
            'potato2.jpg','persimmon1.jpg','persimmon2.jpg', 'image_test.png']

def watershed(img_name):
    img = cv.imread(img_name)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

    # sure background area
    sure_bg = cv.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]

    cv.imshow('image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def rectangle_box(img_name):
    img = cv.imread(img_name)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #ret, thresh = cv.threshold(img, 127, 255, 0)
    #blur = cv.GaussianBlur(gray, (5, 5), 0)

#    blur = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C)
    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    #ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    #contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv.findContours(thresh,1,2)
    # cv.drawContours(img, contours, -1, (0, 255, 0), 3)
    # cnt = contours[1]
    # cv.drawContours(img, [cnt], 0, (0, 255, 0), 3)

    # sort contours
    # cnts = contours[0] if imutils.is_cv() else contours[1]
    sorted_contours = sorted(contours, key=cv.contourArea, reverse=True)

    # sorted_contours = sorted(contours, key=lambda ctr: cv.boundingRect(ctr)[0] + cv.boundingRect(ctr)[1] * img.shape[1] )
    # x + y * w

    # cnt = contours[0]
    # cnt = sorted_contours[1]
    # x,y,w,h = cv.boundingRect(cnt)
    # cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    #
    # cv.imshow('image',img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    #
    for item in range(len(contours)):
        cnt = sorted_contours[item]
        if len(cnt) > 10:
            print(len(cnt))

            # M = cv.moments(cnt)
            # cx = int(M['m10']/M['m00'])
            # cy = int(M['m01']/M['m00'])
            x, y, w, h = cv.boundingRect(cnt)
            if w > 10 and h > 10:
                print(x, y, w, h)
                cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv.imshow('image', img)
                cv.waitKey(0)
                cv.destroyAllWindows()

    # print(len(contours))
    # cv.drawContours(img, contours, -1, (255, 255, 0), 1)
    #
    # cv.imshow("contours", img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

for i in range(len(img_name)):
    rectangle_box(img_name[i])
    #watershed(img_name[i])