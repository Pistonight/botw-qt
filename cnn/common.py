from bitarray import bitarray
import sys
import base64
import gzip
import numpy as np
import os
import re
import cv2
import time
import math
import multiprocessing
from tqdm import tqdm
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
    raw = []
    rect = []
    data = []
    data2 = []
    processes = 1
    json = []

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
        elif arg_key in ["--raw"]:
            # Specify a raw image directory
            args.raw.append(arg_value)
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
        elif arg_key in ["--data2", "-d2"]:
            # Specify a data file
            args.data2.append(arg_value)
        elif arg_key in ["--processes", "-j"]:
            # Specify the number of parallel processes to use
            try:
                args.processes = int(arg_value)
            except ValueError:
                raise ValueError(f"Expected integer for argument {arg_key}, got {arg_value}")
        elif arg_key in ["--json"]:
            # Specify a json file
            args.json.append(arg_value)
        else:
            raise ValueError(f"Unknown argument: {arg_key}")

    return args

def preinit_tensorflow(processes=None):
    if processes:
        os.environ["OMP_NUM_THREADS"] = str(processes)
        os.environ["TF_NUM_INTRAOP_THREADS"] = str(processes)
        os.environ["TF_NUM_INTEROP_THREADS"] = str(processes)


    if not TF_LOGGING:
        # Set TensorFlow Verbosity to Error
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["AUTOGRAPH_VERBOSITY"] = "0"
    print("Initializing TensorFlow...")
    import tensorflow as tf
    import absl.logging
    tf.get_logger().setLevel('ERROR')
    absl.logging.set_verbosity(absl.logging.ERROR)
    if processes:
        tf.config.threading.set_inter_op_parallelism_threads(processes)
        tf.config.threading.set_intra_op_parallelism_threads(processes)

def clean_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return text.strip()


def encode_image(cv2_image):
    """Encode a numpy INPUT_DIM binary image to an ASCII base64 bytes."""
    if cv2_image.shape != (INPUT_DIM[1], INPUT_DIM[0], 1):
        raise ValueError(f"Expected image of shape {INPUT_DIM[1], INPUT_DIM[0], 1}, got {cv2_image.shape}")
    ba = bitarray(endian="big")
    for row in cv2_image:
        for pixel in row:
            ba.append(pixel[0] > 128) # 0 for black, 1 for white
    raw = ba.tobytes()
    compressed = gzip.compress(raw)
    return base64.b64encode(compressed)


def decode_image(encoded_image):
    """Decode an ASCII base64 bytes to a numpy INPUT_DIM binary image."""
    return decode_decompressed_image(decompress_encoded_image(encoded_image))

def decompress_encoded_image(encoded_image):
    compressed = base64.b64decode(encoded_image)
    raw = gzip.decompress(compressed)
    return raw

def decode_decompressed_image(raw):
    ba = bitarray(endian="big")
    ba.frombytes(raw)
    cv2_image = np.zeros((INPUT_DIM[1], INPUT_DIM[0], 1), dtype=np.uint8)
    for i in range(INPUT_DIM[1]):
        for j in range(INPUT_DIM[0]):
            cv2_image[i][j][0] = 255 if ba[i*INPUT_DIM[0]+j] else 0
    return cv2_image

def decode_decompressed_image_to_input(raw):
    ba = bitarray(endian="big")
    ba.frombytes(raw)
    cv2_image = np.zeros((INPUT_DIM[1], INPUT_DIM[0], 1), dtype=np.float32)
    for i in range(INPUT_DIM[1]):
        for j in range(INPUT_DIM[0]):
            cv2_image[i][j][0] = 1 if ba[i*INPUT_DIM[0]+j] else 0
    return cv2_image


def decode_dataset(image_data, label_data, description):
    assert len(image_data) == len(label_data)
    images = []
    image_bytes = []
    labels = []

    with multiprocessing.Pool() as pool:
        for result_image, input_bytes, input_label in tqdm(pool.imap_unordered(_decode_with_label_task, zip(image_data, label_data)), desc=f"Decoding {description} images", total=len(image_data), leave=False):
            images.append(result_image / 255.0)
            image_bytes.append(input_bytes)
            labels.append(input_label)
    return images, labels, image_bytes


def _decode_with_label_task(args):
    image, label = args
    return decode_image(image), image, label


def import_labels(captialize=False):
    arr = [ "None" ]
    with open("quests.txt", "r", encoding="utf-8") as f:
        arr += f.read().splitlines()
    if not captialize:
        arr = [clean_text(x) for x in arr]
    return arr


def import_data(data_file_path=None):
    """Import the data from data txt file, returns a list of sets of encoded images organized by quest ids"""
    labels = import_labels()
    data = []
    if not data_file_path:
        for _ in labels:
            data.append([])
    else:
        with open(data_file_path, "r", encoding="utf-8") as f:
            header = f.readline()
            if not header.startswith("SIZE"):
                raise ValueError("Invalid data file")
            size = int(header[4:].strip())
            for line in tqdm(f, desc=data_file_path, total=size, leave=False):
                if line.startswith("SIZE"):
                    continue
                line = line.strip()
                if "#" in line:
                    line = line[:line.index("#")].strip()
                if line == "LABEL":
                    data.append([])
                else:
                    data[-1].append(line)
    
    return data

def export_data(data, data_file_path):
    """Export the data as txt file"""
    labels = import_labels()
    size = 0
    for s in data:
        size += len(s)
    # get parent directory
    os.makedirs(os.path.dirname(data_file_path), exist_ok=True)
    with open(data_file_path, "w", encoding="utf-8") as f:
        f.write(f"SIZE{size}\n")
        for i in tqdm(range(len(labels)), desc=data_file_path, leave=False):
            f.write(f"LABEL#{labels[i]}\n")
            for line in data[i]:
                f.write(f"{line}\n")


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


def input_from_image(image):
    """Convert numpy image to model input"""
    if image.shape != (INPUT_DIM[1], INPUT_DIM[0], 1):
        raise ValueError(f"Expected image of shape {INPUT_DIM[1], INPUT_DIM[0], 1}, got {image.shape}")
    return image / 255.0


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


def image_from_resized_frame(resized_frame):
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

    cv2_image = np.zeros((INPUT_DIM[1], INPUT_DIM[0], 1), dtype=np.uint8)
    for i in range(INPUT_DIM[1]):
        for j in range(INPUT_DIM[0]):
            pixel = accessor(i, j)
            cv2_image[i][j][0] = 255 if pixel > 128 else 0
    return cv2_image

def measure_sec(start=None):
    if not start:
        return time.perf_counter()
    return time.perf_counter() - start

def measure_str(start=None):
    if not start:
        return datetime.now()
    return str(datetime.now() - start)


def augment_image_shift(image, shift_x, shift_y):
    """Augment an image by shifting it by x and y"""
    return np.roll(image, (shift_y, shift_x), axis=(0, 1))

def augment_image_flip(image, flip_rate):
    # randomly flip some pixels
    new_image = image.copy()
    for row in new_image:
        for pixel in row:
            if np.random.random() < flip_rate:
                pixel[0] = 1 - pixel[0]
    return new_image

def augment_image_wipe_edges(image, distance, threshold):
    # wipe some pixels
    new_image = image.copy()
    for i in range(INPUT_DIM[1]):
        for j in range(INPUT_DIM[0]):
            if image[i][j][0] != 1:
                black_pixels = 0
                total = 1
                for i2 in range(max(0, i - distance), min(INPUT_DIM[1], i + distance + 1)):
                    for j2 in range(max(0, j - distance), min(INPUT_DIM[0], j + distance + 1)):
                        if image[i2][j2][0] != 1:
                            black_pixels += 1
                        total += 1
                if black_pixels / total < threshold:
                    new_image[i][j][0] = 1

    return new_image

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