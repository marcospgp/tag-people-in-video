# tag-people-in-video

Tags people in a video file using the YOLO v2 model.

Based on YAD2K and YOLOw-Keras.

## How to run

1. Install dependencies (more details below)
2. Add the YOLO v2 `yolo.h5` model file to the root folder
3. Run `run.py` with argument `-i` or `--image` being the path to the video file
to run the model on

## Installing dependencies

```
pip install numpy scipy matplotlib pillow
pip install tensorflow-gpu  # CPU-only: conda install -c conda-forge tensorflow
pip install keras # Possibly older release: conda install keras
```
