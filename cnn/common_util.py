import sys
import numpy as np
import os
import re
import cv2
import time
import math
from datetime import datetime
from drawille import Canvas

INPUT_DIM = (492, 46)
FILTER_MIN = 0.01
FILTER_MAX = 0.45
THRESHOLD = 60
BANNER_V_START = 0.19
BANNER_V_END = 0.28
BANNER_H_START = 0.23
BANNER_H_END = 0.77

TF_LOGGING = False # Set to true to enable TensorFlow logging

def gen_seed():
    """Generate a random seed between 1000 and 9999."""
    return int(time.time() * 1000) % 9000 + 1000

class Args:
    model = None
    video = None
    #raw = []
    rect = []
    data = []
    #data2 = []
    processes = 1
    config = []

def parse_args(arg_list=None):
    if not arg_list:
        arg_list = sys.argv[1:]
    if len(arg_list) % 2 != 0:
        print("Expected even number of arguments (--key value) pairs")
        exit(1)
    args = Args()
    for i in range(0, len(arg_list), 2):
        arg_key = arg_list[i]
        arg_value = arg_list[i+1]

        if arg_key in ["--model", "-m"]:
            # Specify a tflite model
            args.model = arg_value
        elif arg_key in ["--video"]:
            # Specify a video file
            args.video = arg_value
        # elif arg_key in ["--raw"]:
        #     # Specify a raw image directory
        #     args.raw.append(arg_value)
        elif arg_key in ["--rect"]:
            # Specify rectangles
            try:
                rect = [int(x) for x in arg_value.split(",")]
            except ValueError:
                raise ValueError("Expected integers for rectangle bounds")
            if len(rect) != 4:
                raise ValueError(f"Not a valid rectangle: {arg_value} (need x,y,w,h)")
            args.rect.append(rect)
        elif arg_key in ["--data", "-d"]:
            # Specify a data file
            args.data.append(arg_value)
        # elif arg_key in ["--data2", "-d2"]:
        #     # Specify a data file
        #     args.data2.append(arg_value)
        elif arg_key in ["--processes", "-j"]:
            # Specify the number of parallel processes to use
            try:
                args.processes = int(arg_value)
            except ValueError:
                raise ValueError(f"Expected integer for argument {arg_key}, got {arg_value}")
        elif arg_key in ["--config", "-c"]:
            # Specify a json file
            args.config.append(arg_value)
        else:
            raise ValueError(f"Unknown argument: {arg_key}")

    return args

def preinit_tensorflow():
    if not TF_LOGGING:
        # Set TensorFlow Verbosity to Error
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["AUTOGRAPH_VERBOSITY"] = "0"
    print("Initializing TensorFlow...")
    import tensorflow as tf
    import absl.logging
    tf.get_logger().setLevel('ERROR')
    absl.logging.set_verbosity(absl.logging.ERROR)
    # Our workflow is not suitable for GPU
    tf.config.set_visible_devices([], 'GPU')

def clean_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return text.strip()


def import_labels(captialize=False):
    arr = [ "None" ]
    with open("quests.txt", "r", encoding="utf-8") as f:
        arr += f.read().splitlines()
    if not captialize:
        arr = [clean_text(x) for x in arr]
    return arr



def get_image_score(image):
    """convert numpy image to percentage of black (non-white) pixels"""
    if image.shape != (INPUT_DIM[1], INPUT_DIM[0], 1):
        raise ValueError(f"Expected image of shape {INPUT_DIM[1], INPUT_DIM[0], 1}, got {image.shape}")
    white_count = 0
    total_count = 0
    for row in image:
        for pixel in row:
            if pixel > 128:
                white_count += 1
            total_count += 1
    return 1 - white_count / total_count


def is_score_valid(score):
    return FILTER_MIN <= score <= FILTER_MAX


def image_from_whole_frame(img):
    # get the dimensions of the image
    height, width, _ = img.shape

    # calculate the start and end coordinates for cropping
    v_start_coord = int(BANNER_V_START * height)
    v_end_coord = int(BANNER_V_END * height)
    h_start_coord = int(BANNER_H_START * width)
    h_end_coord = int(BANNER_H_END * width)

    img = img[v_start_coord:v_end_coord, h_start_coord:h_end_coord]
    return image_from_cropped_frame(img)


def image_from_cropped_frame(cropped_frame):
    image = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, INPUT_DIM, interpolation=cv2.INTER_AREA)
    _, image = cv2.threshold(image, THRESHOLD, 255, cv2.THRESH_BINARY)
    return image_from_resized_frame(image)


def image_from_resized_frame(resized_frame, dtype=np.uint8, value=255):
    shape = resized_frame.shape

    if len(shape) == 3:
        height, width, _ = shape
        accessor = lambda i, j: resized_frame[i][j][0]
    elif len(shape) == 2:
        height, width = shape
        accessor = lambda i, j: resized_frame[i][j]
    else:
        raise ValueError(f"Invalid shape: {shape}")
    
    if height != INPUT_DIM[1] or width != INPUT_DIM[0]:
        raise ValueError(f"Invalid shape: {shape}")

    cv2_image = np.empty((INPUT_DIM[1], INPUT_DIM[0], 1), dtype=dtype)
    for i in range(INPUT_DIM[1]):
        for j in range(INPUT_DIM[0]):
            pixel = accessor(i, j)
            cv2_image[i][j][0] = value if pixel > 128 else 0
    return cv2_image

def measure_sec(start=None):
    if not start:
        return time.perf_counter()
    return time.perf_counter() - start

def measure_str(start=None):
    if not start:
        return datetime.now()
    return str(datetime.now() - start)


def display_image(image):
    """Display an image"""
    term_size = (os.get_terminal_size().columns-2)*2
    SCALE = INPUT_DIM[0]/term_size
    c = Canvas()
    if image.shape != (INPUT_DIM[1], INPUT_DIM[0], 1):
        raise ValueError(f"Expected image of shape {INPUT_DIM[1], INPUT_DIM[0], 1}, got {image.shape}")
    w = math.floor(INPUT_DIM[0]/SCALE)
    h = math.floor(INPUT_DIM[1]/SCALE)
    for x in range(w):
        c.set(x, 0)
        c.set(x, h + 1)
    for y in range(h):
        c.set(0, y)
        c.set(w + 1, y)
    c.set(w + 1, h + 1)
    for x in range(w):
        for y in range(h):
            if image[math.floor(y*SCALE)][math.floor(x*SCALE)][0] == 0:
                c.set(x + 1, y + 1)
    return c.frame()