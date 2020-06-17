import glob
import cv2

data_dir = "dataset_simple/"
class_dir = ["fork/","knife/","/spoon/"]
for classs in class_dir:
    pic_list = glob.glob("./dataset/"+ data_dir +classs + "*.jpg")
    for pic in pic_list:
        print(pic)
        inputs = cv2.imread(pic,0)
        inputs = cv2.resize(inputs,(200,200))
        cv2.imwrite(pic, inputs)