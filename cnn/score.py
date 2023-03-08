import cv2
import os
import multiprocessing
from tqdm import tqdm
from common_util import get_image_score, image_from_resized_frame, parse_args


def get_interval(root_dirs, min_gap):
    print("Processing images...")
    points = []
    with multiprocessing.Pool() as pool:
        for root_directory in tqdm(root_dirs, leave=False):
            add_root_directory(root_directory, points, pool)
    points.sort()
    print("Score interval:")
    return points_to_intervals(points, min_gap)

def add_root_directory(root_directory, points, pool):
    for directory in tqdm(os.listdir(root_directory), desc=root_directory, leave=False):
        if directory != "none" and directory != "none_filtered_out":
            add_directory(os.path.join(root_directory, directory), points, pool)

def add_directory(directory, points, pool):
    def generator():
        for image_path in os.listdir(directory):
            image_path = os.path.join(directory, image_path)
            if not os.path.isfile(image_path):
                continue
            if not image_path.endswith(".png"):
                continue
            yield image_path
    
    for point in pool.imap_unordered(test_image, generator()):
        points.append(point)

def test_image(image_path):
    image = cv2.imread(image_path)
    image = image_from_resized_frame(image)
    return get_image_score(image)

def points_to_intervals(points, min_gap):
    intervals = []
    current_interval = []
    for i in range(len(points)):
        if i == 0:
            current_interval.append(points[i])
        else:
            if points[i] - points[i - 1] > min_gap:
                intervals.append(current_interval)
                current_interval = []
            current_interval.append(points[i])
    intervals.append(current_interval)
    return [
        (x[0], x[0]) if len(x) == 1 else (x[0], x[-1]) for x in intervals
    ]

if __name__ == "__main__":
    args = parse_args()
    if not args.raw:
        print("Must specify at least 1 raw image dataset with --raw")
        exit(1)
    print(get_interval(args.raw, 0.1))