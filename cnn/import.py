import cv2
from common import encode_image, import_data, export_data, import_labels, clean_text, image_from_resized_frame, parse_args
import os
from tqdm import tqdm
import multiprocessing

def resize_and_encode(image_path):
    image = cv2.imread(image_path)
    image = image_from_resized_frame(image)
    return encode_image(image).decode("utf-8"), image_path

def get_name(image_path):
    return os.path.splitext(os.path.basename(image_path))[0]

def import_directory(quest_idx, dir_path, data):
    count = 0
    files = [ os.path.join(dir_path, file) for file in os.listdir(dir_path) if file.endswith(".png") ]
    if len(files) >= 10:
        with multiprocessing.Pool() as pool:
            for encoded, image_path in tqdm(pool.imap_unordered(resize_and_encode, files), total=len(files), desc=dir_path, leave=False):
                data[quest_idx].append(f"{encoded} #{get_name(image_path)}")
                count += 1
    else:
        for file in tqdm(files, desc=dir_path, leave=False):
            encoded, image_path = resize_and_encode(file)
            data[quest_idx].append(f"{encoded} #{get_name(image_path)}")
            count += 1
    return count

def import_root_directory(root_path, data_path):
    data = import_data()
    labels = import_labels()
    count = 0

    for dir in tqdm(os.listdir(root_path), leave=False):
        dir_path = os.path.join(root_path, dir)
        if os.path.isdir(dir_path):
            quest_name = clean_text(dir)
            if quest_name not in labels:
                continue
            
            quest_idx = labels.index(quest_name)
            count += import_directory(quest_idx, dir_path, data)

    export_data(data, data_path)
    return count
    

if __name__ == "__main__":
    args = parse_args()
    if len(args.raw) != 1:
        print("Must specify exactly 1 raw image dataset with --raw")
        exit(1)
    if len(args.data) != 1:
        print("Must specify exactly 1 data file with --data")
        exit(1)
    print(f"Importing images from {args.raw[0]} to {args.data[0]}...")
    count = import_root_directory(args.raw[0], args.data[0])
    print(f"\rImported {count} images.")