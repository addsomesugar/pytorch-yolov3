import glob
import os
import numpy as np

image_dir = '/home/work_dir/pytorch-yolov3/data/VOCdevkit/VOC2012/JPEGImages'
label_dir = '/home/work_dir/pytorch-yolov3/data/VOCdevkit/VOC2012/Annotations'
data_list = '/home/work_dir/pytorch-yolov3/data/voc2012.txt'


if __name__ == '__main__':
    image_list = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    label_list = sorted(glob.glob(os.path.join(label_dir, "*.xml")))
    print("image num = {}, label num = {}".format(len(image_list), len(label_list)))
    image_label_list = []
    for image_file, label_file in zip(image_list, label_list):
        print(os.path.splitext(os.path.basename(image_file)))
        assert os.path.splitext(os.path.basename(image_file))[0] == os.path.splitext(os.path.basename(label_file))[0]
        image_label_list.append([image_file, label_file])
    with open(data_list, 'w') as f:
        for img, lbl in image_label_list:
            f.write(img+' '+lbl+'\n')


