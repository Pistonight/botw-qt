import sys
import os
import re
import time
import math
from datetime import datetime
from drawille import Canvas

INPUT_DIM = (492, 46)


TF_LOGGING = False # Set to true to enable TensorFlow logging

def gen_seed():
    """Generate a random seed between 1000 and 9999."""
    return int(time.time() * 1000) % 9000 + 1000

class Args:
    model = None
    video = []
    rect = []
    data = []
    processes = 1
    config = []
    flags = set()

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
            args.video.append(arg_value)
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
            # Specify a data directory
            args.data.append(arg_value)
        elif arg_key in ["--processes", "-j"]:
            # Specify the degree of parallelism
            try:
                args.processes = int(arg_value)
                if args.processes < 1:
                    print("Must specify processes >= 1")
                    exit(1)
            except ValueError:
                raise ValueError(f"Expected integer for argument {arg_key}, got {arg_value}")
        elif arg_key in ["--config", "-c"]:
            # Specify a config file
            args.config.append(arg_value)
        elif arg_key in ["--enable", "-e"]:
            # Enable a flag
            args.flags.add(arg_value)
        else:
            raise ValueError(f"Unknown argument: {arg_key}")

    
    
    return args

def preinit_tensorflow(use_gpu=False):
    if not TF_LOGGING:
        # Set TensorFlow Verbosity to Error
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["AUTOGRAPH_VERBOSITY"] = "0"
    print("Initializing TensorFlow...")
    
    import tensorflow as tf
    import absl.logging
    tf.get_logger().setLevel('ERROR')
    absl.logging.set_verbosity(absl.logging.ERROR)
    if not use_gpu:
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