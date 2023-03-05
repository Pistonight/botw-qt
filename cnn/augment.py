 
from common import parse_args

if __name__ == "__main__":
    args = parse_args()
    if len(args.data) != 2:
        print("Input and output data must be specified with --data")
        exit(1)
    if len(args.json) != 1:
        print("Augment parameters must be specified with --json")
        exit(1)

import multiprocessing
import json
from tqdm import tqdm

from common import encode_image, decode_image, import_data, export_data, augment_image_shift, augment_image_flip, augment_image_wipe_edges

def run(in_data_path, out_data_path, parameter_path):
    print("Loading data...")
    with open(parameter_path, "r") as f:
        parameters = json.load(f)
    data = import_data(in_data_path)
    print("\rAugmenting...")
    data = augment(data, parameters)
    print("\rSaving data...")
    export_data(data, out_data_path)

def augment(data, augment_parameters):
    new_data = []
    for _ in data:
        new_data.append(set())
    size = 0
    for s in data:
        size += len(s)

    x_shift_count = augment_parameters["x_shift_count"]
    y_shift_count = augment_parameters["y_shift_count"]
    flip_count = augment_parameters["flip_count"]
    flip_rate = augment_parameters["flip_rate"]
    wipe_count = augment_parameters["wipe_count"]
    wipe_distance = augment_parameters["wipe_distance"]
    wipe_threshold = augment_parameters["wipe_threshold"]
    wipe_as_none = augment_parameters["wipe_as_none"]

    def generator():
        for label, encoded_images in enumerate(data):
            for encoded in encoded_images:
                yield encoded, label, x_shift_count, y_shift_count, flip_count, flip_rate, wipe_count, wipe_distance, wipe_threshold, wipe_as_none
        
    with multiprocessing.Pool() as pool:
        for encoded_images, labels in tqdm(pool.imap_unordered(get_augmented_images, generator()), total=size, leave=False):
            for encoded, label in zip(encoded_images, labels):
                new_data[label].add(encoded)
    
    return new_data



def get_augmented_images(args):
    encoded, label, x_shift_count, y_shift_count, flip_count, flip_rate, wipe_count, wipe_distance, wipe_threshold, wipe_as_none = args
    image = decode_image(encoded) / 255.0
    new_images = []
    labels = []
    # Shifting the image by a bit
    for shift_x in range(x_shift_count, x_shift_count + 1):
        for shift_y in range(-y_shift_count, y_shift_count + 1):
            new_image = augment_image_shift(image, shift_x, shift_y)
            new_images.append(new_image)
            labels.append(label)

    # Randomly flip some pixels in the image
    for _ in range(flip_count):
        # randomly flip some pixels
        new_image = augment_image_flip(image, flip_rate)
        new_images.append(new_image)
        labels.append(label)

    # Randomly wipe some pixels in the image, and label it none
    for _ in range(wipe_count):
        new_image = augment_image_wipe_edges(image, wipe_distance, wipe_threshold)
        new_images.append(new_image)
        if wipe_as_none:
            labels.append(0)
        else:
            labels.append(label)

    return [ encode_image(x * 255.0).decode("utf-8") for x in new_images ], labels

if __name__ == "__main__":
    run(args.data[0], args.data[1], args.json[0])