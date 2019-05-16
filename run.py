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
import cv2
from tqdm import tqdm

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
    if video_path[-4:] != '.mp4':
        print("Input path must point to an mp4 file.")
    else:
        process_video(video_path)

def process_video(video_path):
    video_out = video_path[:-4] + '_detected' + video_path[-4:]
    video_reader = cv2.VideoCapture(video_path)

    nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

    video_writer = cv2.VideoWriter(video_out,
                               cv2.VideoWriter_fourcc(*'MPEG'),
                               50.0,
                               (frame_w, frame_h))

    for i in tqdm(range(nb_frames)):
        _, image = video_reader.read()

        # Convert image into pillow image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)

        result = predict(frame_w, frame_h, pil_image)

        # Reconvert into opencv format
        result = np.asarray(result)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

        video_writer.write(np.uint8(result))

    video_reader.release()
    video_writer.release()

def preprocess_image(image):
    resized_image = image.resize((608, 608), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    return image, image_data

def predict(width, height, pre_image):
    class_names, anchors, yolo_model = load_yolo()
    # Convert final layer features to bounding box parameters
    yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
    # More about yolo_eval on keras_yolo.py in yad2k/models
    boxes, scores, classes = yolo_eval(yolo_outputs, (float(height), float(width)))
    # Preprocess the input image before feeding into the convolutional network
    image, image_data = preprocess_image(pre_image)
    # Feed image into network to get prediction
    out_scores, out_boxes, out_classes = feed(scores, boxes, classes, yolo_model, image_data)
    # Apply results to image
    image = draw_result(image, out_scores, out_boxes, out_classes, class_names)
    return image

def draw_result(image, out_scores, out_boxes, out_classes, class_names):
    # Define which results to display
    display_classes = ['person']
    # Produce the colors for the bounding boxes
    colors = generate_colors(class_names)
    # Draw the bounding boxes
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors, display_classes)
    return image

def feed(scores, boxes, classes, yolo_model, image_data):
    # Initiate Keras session
    sess = K.get_session()
    # Run the session
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input:image_data,K.learning_phase(): 0})
    return out_scores, out_boxes, out_classes

def load_yolo():
    class_names = read_classes("YOLOw-Keras/model_data/coco_classes.txt")
    anchors = read_anchors("YOLOw-Keras/model_data/yolo_anchors.txt")
    yolo_model = load_model("yolo.h5")
    return class_names, anchors, yolo_model

main()
