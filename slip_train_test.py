# convert jpg to png file

import cv2
import os

source_path = "/home/hien/Documents/yolov3_training_testing_256/train/"
dest_path = "/home/hien/Documents/yolov3_training_testing_256/train/"


def is_image_file(a_file):
    image = cv2.imread(a_file)
    if image is not None:
        return True
    else:
        return False


"""
    Get all image path from a folder, the image path includes only the folder name and the file name
    @param a_dir: the absoluate path to the folder
    @return: a list of image files 
"""


def get_all_image_files_from_folder(a_dir):
    image_files = [os.path.join(a_dir, file_name) for file_name in os.listdir(a_dir) if
                   (os.path.isfile(os.path.join(a_dir, file_name)))]
    return image_files


def print_separator():
    print("=" * 20)


def convert_images(convert_to_color=True):
    print("Source Path: " + source_path)
    print_separator()

    print("Destination Path: " + dest_path)
    print_separator()

    print("Retrieve all image files from source folder")
    image_files = get_all_image_files_from_folder(source_path)
    print_separator()

    batch_size = 100
    no_of_step = int(len(image_files) / batch_size) + (1 if len(image_files) % batch_size != 0 else 0)

    for i in range(no_of_step):
        lower_bound = i * batch_size + 1
        upper_bound = (i + 1) * batch_size if (i + 1) * batch_size < len(image_files) else len(image_files)
        print("Reading images from {} to {}".format(lower_bound, upper_bound))
        image_batch = image_files[lower_bound - 1: upper_bound]

        #print("Read all image files")
        #images = [cv2.cvtColor(cv2.imread(file, cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2RGB) for file in image_batch]
        images = [cv2.imread(file) for file in image_batch]
        #print_separator()

        for i, image in enumerate(images):
            #print("Copying image: " + image_files[i].split("/")[-1])
            #cv2.imshow("Raw", image)
            #cv2.waitKey(0)
            cv2.imwrite(os.path.join(dest_path, image_batch[i].split("\\")[-1].split(".")[0] + ".jpg"), image)
            print_separator()


convert_images()