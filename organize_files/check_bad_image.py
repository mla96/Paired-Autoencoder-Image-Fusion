#!/usr/bin/env python3
"""
This script identifies corrupted images in a directory.
"""


import os
from os import listdir
import PIL.Image

dir_path = "../../../OakData/Transfer_learning_datasets/train"


for filename in listdir(dir_path):
    try:
        img = PIL.Image.open(os.path.join(dir_path, filename))  # open the image file
        img.verify()  # verify that it is, in fact an image
    except (IOError) as e:
        print('Bad file:', filename)
        #os.remove(base_dir+"\\"+filename) (Maybe)