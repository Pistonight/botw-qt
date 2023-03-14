from common_util import parse_args, preinit_tensorflow

if __name__ == "__main__":
    args = parse_args()
    if not args.video:
        print("Please specify at lease one video with --video")
        exit(1)
    if len(args.data) != 1:
        print("Please specify a single data output directory with --data/-d")
        exit(1)
    if len(args.rect) > 1:
        print("Cannot specify more than 1 rectangle for cropping")
        exit(1)
    if not args.model:
        print("Please specify a model with --model/-m")
        exit(1)
    preinit_tensorflow()

import os
import cv2
import multiprocessing
import math
import hashlib
import numpy as np
from tqdm import tqdm
from common_util import import_labels, INPUT_DIM, measure_str
from common_dataset import create_dataset_from_paths_and_labels
from common_runner import init_runner_singleton, singleton_run_batch_with_paths

# show the frame for debugging purposes
SHOW = False
CONFIDENCE = 97
IS_BGR = False
# set cache as 16gb
CACHE_SIZE = 16 * 1024 * 1024 * 1024
SKIP_FACTOR = 3 # 1=get every frame, 3=get every 3rd frame, etc.
DEFAULT_FPS = 30

FILTER_MIN = 0.01
FILTER_MAX = 0.45
THRESHOLD = 60
BANNER_V_START = 0.19
BANNER_V_END = 0.28
BANNER_H_START = 0.23
BANNER_H_END = 0.77

class Cache:
    data: set
    size: int
    max_size: int
    def __init__(self, max_size):
        self.data = set()
        self.size = 0
        self.max_size = max_size
    
    def add(self, encoded):
        if encoded in self.data:
            return
        self.data.add(encoded)
        self.size += len(encoded)
        if self.size > self.max_size:
            self.data = set()
            self.size = 0
        
    def __contains__(self, encoded):
        return encoded in self.data

def scrape_vod(vod_path, output_path, crop, threads):
    print(f"\rLoading {vod_path}...")
    camera = cv2.VideoCapture(vod_path)

    try:
        total_frames = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames: {total_frames}")
    except:
        total_frames = None
        print("Unable to get total frames, falling back to single thread processing...")
        threads = 1

    try:
        fps = int(camera.get(cv2.CAP_PROP_FPS))
        print(f"FPS: {fps}")
        if fps < DEFAULT_FPS:
            print("WARNING: FPS is less than 30, using 30fps instead")
            fps = DEFAULT_FPS
    except:
        fps = DEFAULT_FPS
        print("Unable to get fps, defaulting to 30fps...")
    
    print("Scraping frames...")
    task_args = []
    frames_per_thread = None if total_frames is None else int(math.ceil(total_frames / threads))
    for i in range(threads):
        task_args.append((
            i,
            vod_path,
            fps,
            output_path,
            crop,
            None if frames_per_thread is None else i * frames_per_thread,
            None if frames_per_thread is None else min((i + 1) * frames_per_thread, total_frames),
        ))

    tqdm.set_lock(multiprocessing.RLock())
    with multiprocessing.Pool(threads, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as pool:
        pool.map(run_scraper, task_args)
    

def run_scraper(args):
    runner_idx, vod_path, fps, output_path, crop, start_frame, end_frame = args
    tmp_output_path = f"{output_path}.tmp"
    os.makedirs(tmp_output_path, exist_ok=True)
    camera = cv2.VideoCapture(vod_path)

    file_name = 1
    full_file_name = f"{tmp_output_path}/{runner_idx}_{file_name}.png"
    while os.path.exists(full_file_name):
        file_name += 1
        full_file_name = f"{tmp_output_path}/{runner_idx}_{file_name}.png"
    ignore_count = int(max(0, SKIP_FACTOR * (fps/DEFAULT_FPS) - 1))
    total_frames = math.floor((end_frame - start_frame)/(ignore_count+1)) if start_frame is not None else None

    for success, frame in tqdm(frame_generator(camera, ignore_count, start_frame, end_frame), desc=f"Thread {runner_idx}", total=total_frames, leave=False, position=runner_idx, unit="frames"):
        if not success:
            break
        if crop:
            x, y, w, h = crop
            frame = frame[y:y+h, x:x+w]
        if IS_BGR:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get the black and white banner image
        banner_image = image_from_whole_frame(frame)

        if SHOW:
            cv2.imshow(f"Scraping {vod_path} (Thread {runner_idx})", overlay_frame(frame, banner_image))
            cv2.waitKey(1)
        
        score = get_image_score(banner_image)

        if not is_score_valid(score):
            continue

        full_file_name = f"{tmp_output_path}/{runner_idx}_{file_name}.png"
        cv2.imwrite(full_file_name, banner_image)
        file_name += 1
    

def frame_generator(camera, ignore_count, start_frame, end_frame):
    if start_frame:
        camera.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    success = False
    if end_frame:
        i = start_frame
        while i < end_frame:
            for _ in range(ignore_count):
                success, _ = camera.read()
                i += 1
                if not success:
                    break
            if not success:
                break
            yield camera.read()
            i += 1
    else:
        while True:
            for _ in range(ignore_count):
                success, _ = camera.read()
                if not success:
                    break
            if not success:
                    break
            yield camera.read()

def overlay_frame(base, overlay):
    ov_height, ov_width, _= overlay.shape
    base_height, base_width, _ = base.shape
    
    for v in range(min(ov_height, base_height)):
        for h in range(min(ov_width, base_width)):
            base[v][h] = overlay[v][h][0]
    return base


def condense_directory(directory):
    hashes = Cache(CACHE_SIZE)
    kept_count = 0
    total_count = 0

    sub_files = [ os.path.join(directory, d) for d in os.listdir(directory) ]

    with multiprocessing.Pool() as pool:
        for hash, path in tqdm(pool.imap_unordered(get_hash, sub_files), total=len(sub_files), leave=False):
            if not hash:
                continue
            if not hash in hashes:
                hashes.add(hash)
                kept_count += 1
            else:
                os.remove(path)
            total_count += 1

    return kept_count, total_count

def create_raw(directory):
    labels = import_labels()
    for label in labels:
        os.makedirs(os.path.join(directory, label), exist_ok=True)



def get_hash(path):
    if not path.endswith(".png"):
        return None, path
    with open(path, "rb") as file:
        data = file.read()
        return hashlib.md5(data).hexdigest(), path
    
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

def apply_labels(output_directory, model_path, processes):
    quest_labels = import_labels()
    batch_size = processes

    input_directory = output_directory+".tmp"

    image_paths = [ os.path.join(input_directory, d) for d in os.listdir(input_directory) if d.endswith(".png")]
    image_labels = [0] * len(image_paths) # stub labels
   
    workers = processes // 2

    dataset = create_dataset_from_paths_and_labels(image_paths, image_labels, batch_size, workers=workers, keep_path=True)

    lock = multiprocessing.RLock()
    tqdm.set_lock(lock)

    print("\rLabeling images...")

    kept_count = 0
    with multiprocessing.Pool(processes=workers, initializer=init_runner_singleton, initargs=(model_path, batch_size, lock)) as pool:
        for image_paths_batch, _, predicted_labels_batch, predicted_confidences_batch in tqdm(
            pool.imap(singleton_run_batch_with_paths, dataset.as_numpy_iterator()),
            total=math.ceil(total / batch_size), unit="batch", leave=False
        ):
            for i in range(len(image_paths_batch)):
                old_path = image_paths_batch[i]
                image_base_name = os.path.basename(old_path)
                predicted_label = predicted_labels_batch[i]
                if predicted_label == 0:
                    # move to output/none
                    target_path = os.path.join(output_directory, "none", image_base_name)
                    os.rename(old_path, target_path)
                else:
                    if predicted_confidences_batch[i] < CONFIDENCE:
                        # keep in output.tmp
                        kept_count += 1
                    else:
                        # move to output/quest
                        target_path = os.path.join(output_directory, quest_labels[predicted_label], image_base_name)
                        os.rename(old_path, target_path)

    print(f"\r{kept_count} images were left in {input_directory} because they were not confident enough.")

if __name__ == "__main__":
    start = measure_str()
    model_path = args.model
    output_path = args.data[0]
    crop = args.rect[0] if args.rect else None

    # Step 1: Scrape VODs
    for v in args.video:
        scrape_vod(v, output_path, crop, args.processes)

    # Step 2: Remove identical images
    print("\rCondensing...")
    kept, total = condense_directory(output_path+".tmp")
    print(f"\rKept {kept} of {total} images ({kept / total * 100:.2f}%)")
    create_raw(output_path+".par")
    if "dlc" in args.flags:
        os.makedirs(os.path.join(output_path+".dlc", "none"), exist_ok=True)
    create_raw(output_path)

    # Step 3: Auto labeling
    apply_labels(output_path, model_path, args.processes)
    print()
    print(f"Done in {measure_str(start)}")

    print()
    print("Next steps:")
    print(f"1. Check images in {output_path} with a labelled quest, ane move partials to {output_path}.par")
    print(f"2. Check images in {output_path}.tmp, move them appropriately, and delete {output_path}.tmp")

    if "dlc" in args.flags:
        print(f"For DLC quests, put them in {output_path}.dlc/none")
    