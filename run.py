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

##################
# Define functions

def main():
    args = argparser.parse_args()
    video_path = args.input
    predict(video_path)

def predict(video_path):
    width, height = get_image_dimensions(video_path)
    class_names, anchors, yolo_model = load_yolo()
    # Convert final layer features to bounding box parameters
    yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
    # More about yolo_eval on keras_yolo.py in yad2k/models
    boxes, scores, classes = yolo_eval(yolo_outputs, (height, width))
    # Preprocess the input image before feeding into the convolutional network
    image, image_data = preprocess_image(video_path, model_image_size = (608, 608))
    # Feed image into network to get prediction
    out_scores, out_boxes, out_classes = feed(scores, boxes, classes, yolo_model, image_data)
    # Show results
    print_results(image, out_scores, out_boxes, out_classes, class_names)

def print_results(image, out_scores, out_boxes, out_classes, class_names):
    # Define which results to display
    display_classes = ['person']
    # Produce the colors for the bounding boxes
    colors = generate_colors(class_names)
    # Draw the bounding boxes
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors, display_classes)
    # Apply the predicted bounding boxes to the image and save it
    out_path = 'out.jpg'
    print("Saving image to", out_path)
    image.save(out_path, quality=90)

def feed(scores, boxes, classes, yolo_model, image_data):
    # Initiate Keras session
    sess = K.get_session()
    # Run the session
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input:image_data,K.learning_phase(): 0})
    return out_scores, out_boxes, out_classes

def get_image_dimensions(path):
    input_image = Image.open(path)
    width, height = input_image.size
    width = np.array(width, dtype=float)
    height = np.array(height, dtype=float)
    return width, height

def load_yolo():
    class_names = read_classes("YOLOw-Keras/model_data/coco_classes.txt")
    anchors = read_anchors("YOLOw-Keras/model_data/yolo_anchors.txt")
    yolo_model = load_model("yolo.h5")
    return class_names, anchors, yolo_model

main()
