#! /usr/bin/env python

###################
# Load dependencies

import os
import sys
import argparse
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
from PIL import Image

from keras import backend as K
from keras.models import load_model

sys.path.insert(0, 'YOLOw-Keras/')
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes

from YAD2K.yad2k.models.keras_yolo import yolo_head, yolo_eval

###################
# Declare arguments

argparser = argparse.ArgumentParser(
    description='Identify people in videos using the YOLO_v2 model')

argparser.add_argument(
    '-i',
    '--input',
    help='path to the video file')

def main():
    args = argparser.parse_args()
    video_path = args.input
    print(video_path)

main()
