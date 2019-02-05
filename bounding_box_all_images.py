
import cv2 as cv
import fnmatch
import numpy as np
import os

#img_name = ['0_persimmon.jpg','kiwi1.jpg', 'kiwi2.jpg','apple1.jpg', 'apple2.jpg', 'banana1.jpg','banana2.jpg', 'lemon1.jpg', 'potato1.jpg',
#            'potato2.jpg','persimmon1.jpg','persimmon2.jpg', 'image_test.png']

# the class number of fruits for YOLO
fruit_dict = dict(apple=0, banana=1, garlic=2, kiwi=3, lemon=4, pear=5, persimmon=6, potato=7, starfruit=8)

#folder_name = 'test_sample'
folder_name = 'apple'
precision = 6

# list all image files
#img_list = os.listdir(folder_name)
img_list = fnmatch.filter(os.listdir(folder_name), '*.jpg')

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
    path_name = folder_name + '/' + img_name
    img = cv.imread(path_name)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # get dimensions of image
    dimensions = img.shape
    #print(len(dimensions), dimensions[0], dimensions[1], dimensions[2])
    dimensions = gray.shape
    #print(len(dimensions), dimensions[0], dimensions[1])
    h_img = dimensions[0]
    w_img = dimensions[1]

    #ret, thresh = cv.threshold(img, 127, 255, 0)
    # improve tyhe bounding box with these parameters
    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    #ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    #contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contours, hierarchy = cv.findContours(thresh,1,2)
    # cv.drawContours(img, contours, -1, (0, 255, 0), 3)
    # cnt = contours[1]
    # cv.drawContours(img, [cnt], 0, (0, 255, 0), 3)

    # sort contours by area
    sorted_contours = sorted(contours, key=cv.contourArea, reverse=True)

    # Take a specific contour
    # cnt = contours[0]
    # cnt = sorted_contours[1]
    # x,y,w,h = cv.boundingRect(cnt)
    # cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    for item in range(len(contours)):
        cnt = sorted_contours[item]
        if len(cnt) > 30:
            #print(len(cnt))
            x, y, w, h = cv.boundingRect(cnt)

            # extract fruit class
            fruit_name = img_name.split('.')[0].split('_')[1]
            fruit_number = fruit_dict[fruit_name]
            print(img_name)
            #print(fruit_name, fruit_number)

            # write bounding box files
            text_name = folder_name + '/' + img_name.split('.')[0] + '.txt'
            #print(text_name)
            file = open(text_name, 'a')

            if fruit_name != 'lemon':
                if x*y != 0 and w > 20 and w < 255 and h > 20 and h < 255:
                    print(x, y, w, h)
                    x, y, w, h = convert_shape_based_on_fruit(x, y, w, h, fruit_name)
                    # convert bounding box files to the requirements of YOLO bounding box files
                    x_c, y_c, w_c, h_c = convert_bounding_box_to_YOLO(x, y, w, h, w_img, h_img)

                    # bounding_box = str(fruit_number) + ' ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h)
                    bounding_box = str(fruit_number) + ' ' + str(x_c) + ' ' + str(y_c) + ' ' + str(w_c) + ' ' + str(h_c)
                    file.write(bounding_box + '\n')
                    print(x_c, y_c, w_c, h_c)
                    file.close()

                    # display images with bounding boxes
                    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv.imshow('image', img)
                    cv.waitKey(0)
                    cv.destroyAllWindows()

            else:
                if w < 255 and h < 255:
                    print(x, y, w, h)
                    x, y, w, h = convert_shape_based_on_fruit(x, y, w, h, fruit_name)
                    # convert bounding box files to the requirements of YOLO bounding box files
                    x_c, y_c, w_c, h_c = convert_bounding_box_to_YOLO(x, y, w, h, w_img, h_img)

                    # bounding_box = str(fruit_number) + ' ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h)
                    bounding_box = str(fruit_number) + ' ' + str(x_c) + ' ' + str(y_c) + ' ' + str(w_c) + ' ' + str(h_c)
                    file.write(bounding_box + '\n')
                    print(x_c, y_c, w_c, h_c)
                    file.close()

                    # display images with bounding boxes
                    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv.imshow('image', img)
                    cv.waitKey(0)
                    cv.destroyAllWindows()



# convert bounding box files to the requirements of YOLO bounding box files
# x, y, w, h are floating numbers [0.0, 1.0]
# https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects
def convert_bounding_box_to_YOLO(x, y, w, h, w_img, h_img):
    x_c = x + w/2
    x_c = round(x_c/w_img, precision)
    y_c = y + h/2
    y_c = round(y_c/h_img, precision)
    w_c = round(w/w_img, precision)
    h_c = round(h/h_img, precision)
    return x_c, y_c, w_c, h_c

# convert x, y, w, h based on fruit types
def convert_shape_based_on_fruit (x, y, w, h, fruit_name):
    x_f = x
    y_f = y
    w_f = w
    h_f = h

    # apple, pear
    if fruit_name == 'apple' or fruit_name == 'pear':
        if x > 24:
            x_f = x - 24
        w_f = w + 20
    # kiwi, persimmon, potato and starfruit
    if fruit_name == 'kiwi' or fruit_name == 'persimmon' or fruit_name == 'potato' or fruit_name == 'starfruit':
        if x > 14:
            x_f = x - 14
        w_f = w + 10
    # lemon and banana
    if fruit_name == 'lemon' or fruit_name == 'banana':
        w_f = w + 22
        h_f = h + 10
        if x > 20:
            x_f = x - 20
    # garlic
    if fruit_name == 'garlic':
        w_f = w + 10
    print (x_f, y_f, w_f, h_f, fruit_name)

    return x_f, y_f, w_f, h_f

for i in range(len(img_list)):
    rectangle_box(img_list[i])
    # Watershed algorithm does not work well
    #watershed(img_name[i])
