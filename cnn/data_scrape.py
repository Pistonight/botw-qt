from common import parse_args, preinit_tensorflow

if __name__ == "__main__":
    args = parse_args()
    if not args.model:
        print("Please specify a model with --model/-m")
        exit(1)
    if not args.video:
        print("Please specify a video with --video")
        exit(1)
    if len(args.raw) != 1:
        print("Please specify a single raw output directory with --raw")
        exit(1)
    if len(args.rect) > 1:
        print("Cannot specify more than 1 rectangle for cropping")
        exit(1)
    if args.processes < 1:
        print("Must specify processes >= 1")
        exit(1)
    preinit_tensorflow()

import os
import cv2
import multiprocessing
import math
import hashlib
import tensorflow as tf
from tqdm import tqdm
from runner import ModelRunner
from common import import_labels, encode_image, image_from_whole_frame, get_image_score, is_score_valid, measure_str, input_from_image

# show the frame for debugging purposes
SHOW = False
IS_BGR = False
NON_BANNER_CACHE_SIZE_TOTAL = 16 * (2 ** 30) # 16GB
SKIP_FACTOR = 3 # 1=get every frame, 3=get every 3rd frame, etc.

class NoneBannerCache:
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

def scrape_vod(vod_path, model_path, output_path, crop, threads):
    print("Loading VOD...")
    camera = cv2.VideoCapture(vod_path)
    try:
        total_frames = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames: {total_frames}")
        print(f"Using {threads} threads to scrape {vod_path}")
    except:
        total_frames = None
        print("Unable to get total frames, falling back to single thread processing")
        threads = 1
    
    task_args = []
    frames_per_thread = None if total_frames is None else int(math.ceil(total_frames / threads))
    for i in range(threads):
        task_args.append((
            i,
            vod_path,
            model_path,
            output_path,
            crop,
            None if frames_per_thread is None else i * frames_per_thread,
            None if frames_per_thread is None else min((i + 1) * frames_per_thread, total_frames),
            NON_BANNER_CACHE_SIZE_TOTAL // threads
        ))

    start = measure_str()
    tqdm.set_lock(multiprocessing.RLock())
    with multiprocessing.Pool(threads, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as pool:
        for _ in pool.imap_unordered(run_scraper, task_args):
            pass
    print("\rCondensing...")
    condense_root_directory(f"{output_path}_unchecked")
    create_raw(f"{output_path}_partial")
    print(f"Done in {measure_str(start)}")
    os.makedirs(f"{output_path}/none", exist_ok=True)

def run_scraper(args):
    runner_idx, vod_path, model_path, output_path, crop, start_frame, end_frame, cache_size = args

    camera = cv2.VideoCapture(vod_path)
    runner = ModelRunner(model_path)
    quest_labels = import_labels()
    file_name = 1
    none_banner_cache = NoneBannerCache(cache_size)
    total_frames = math.floor((end_frame - start_frame)/SKIP_FACTOR) if start_frame is not None else None

    for success, frame in tqdm(frame_generator(camera, start_frame, end_frame), desc=f"Thread {runner_idx}", total=total_frames, leave=False, position=runner_idx):
        if not success:
            break
        if crop:
            x, y, w, h = crop
            frame = frame[y:y+h, x:x+w]
        if IS_BGR:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        banner_image = image_from_whole_frame(frame)

        if SHOW:
            cv2.imshow(f"Scraping {vod_path} (Thread {runner_idx})", overlay_frame(frame, banner_image))
            cv2.waitKey(1)
        
        score = get_image_score(banner_image)

        if not is_score_valid(score):
            continue

        predicted_idx, confidence = runner.run_one(input_from_image(banner_image))
        if confidence < 100:
            predicted_idx = 0

        if predicted_idx == 0:
            encoded = encode_image(banner_image).decode("utf-8")
            # don't save the same image twice
            if encoded in none_banner_cache:
                continue
            none_banner_cache.add(encoded)
        
        save_image(runner_idx, output_path, quest_labels[predicted_idx], banner_image, file_name)
        file_name += 1
    

def frame_generator(camera, start_frame, end_frame):
    ignore_count = max(0, SKIP_FACTOR - 1)
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

def save_image(runner_idx, out_path, quest_name, frame, file_name):
    dir_name = f"{out_path}_unchecked/{quest_name}"
    os.makedirs(dir_name, exist_ok=True)

    cv2.imwrite(f"{dir_name}/{runner_idx}_{file_name}.png", frame)

def overlay_frame(base, overlay):
    ov_height, ov_width, _= overlay.shape
    base_height, base_width, _ = base.shape
    
    for v in range(min(ov_height, base_height)):
        for h in range(min(ov_width, base_width)):
            base[v][h] = overlay[v][h][0]
    return base

def condense_root_directory(root_directory):
    kept_count = 0
    total_count = 0
    sub_dirs = [os.path.join(root_directory, d) for d in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, d))]

    for d in tqdm(sub_dirs, leave=False):
        c, t = condense_directory(d)
        kept_count += c
        total_count += t

    return kept_count, total_count

def condense_directory(directory):
    hashes = set()
    kept_count = 0
    total_count = 0

    sub_files = [ os.path.join(directory, d) for d in os.listdir(directory) ]

    with multiprocessing.Pool() as pool:
        for hash, path in tqdm(pool.imap_unordered(get_hash, sub_files), total=len(sub_files), desc=directory, leave=False):
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
    

if __name__ == "__main__":
    if args.processes > 1:
        # Only use CPU, since we don't have enough GPUs for multi processing
        tf.config.set_visible_devices([], 'GPU')
    vod_path = args.video
    model_path = args.model
    output_path = args.raw[0]
    crop = args.rect[0] if args.rect else None
    scrape_vod(vod_path, model_path, output_path, crop, args.processes)
    