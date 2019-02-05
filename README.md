# calc_bounding_box_YOLO
This Python program is to generate files containing bounding box information for fruit objects. 
These bouding box information is used to train the YOLO algorithm. 
Specificially, for each image, this program will detect objects using opencv, generate files containing bounding box information. 
Each bounding box will contain 5 information: the_class_number x y w h, where x and y are the central point of the bounding box.
