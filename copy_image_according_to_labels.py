import os
import cv2
import shutil

source_path = 'test_copy_image'
des_path = 'train_image_label'

def get_all_label_file_to_image_file():
    list_file = os.listdir(source_path)
    list_label = [file for file in list_file if file.endswith('.txt')]
    return list_label

def copy_image_according_to_label():
    label_name = get_all_label_file_to_image_file()
    print('There are {} label files'.format(len(label_name)))
    print(label_name)

    # copy files
    for name in label_name:
        # copy text file
        orig_label = os.path.join(source_path, name)
        des_label = os.path.join(des_path, name)
        print(des_label)
        shutil.copy(orig_label, des_label)

        # copy image file
        img_name = name.split('.')[0] + '.jpg'
        orig_img = os.path.join(source_path, img_name)
        #print(origin)
        des_img = os.path.join(des_path, img_name)
        #print(des)
        img = cv2.imread(orig_img)
        cv2.imwrite(des_img, img)



copy_image_according_to_label()