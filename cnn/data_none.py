
import os
import cv2
import numpy as np
from tqdm import tqdm
import multiprocessing
from common_util import INPUT_DIM, image_from_resized_frame, parse_args
def process_directory(directory):
    none_directory = os.path.join(directory, "none")
    mislabel_directory = os.path.join(directory, "__mislabel__")
    os.makedirs(mislabel_directory, exist_ok=True)

    mislabeled = filter_mislabeled(none_directory, os.listdir(none_directory))
    cv2.destroyAllWindows()

    print("Renaming...")
    for image_path in tqdm(mislabeled, leave=False):
        os.rename(os.path.join(none_directory, image_path), os.path.join(mislabel_directory, image_path))

def filter_mislabeled(none_directory, input_mislabeled) -> list:
    mislabeled = []
    current = []

    total = len(input_mislabeled)
    with multiprocessing.Pool() as pool:
        for i, image_path in enumerate(input_mislabeled):
            current.append(image_path)
            
            if i % 100 == 99 or i == total-1:
                canvas = put_canvas(current, pool, none_directory)
                print(f"{i}/{total}")
                print("[y] - has mislabeled images")
                print("other - no mislabeled images")
                cv2.imshow("canvas", canvas)
                key = cv2.waitKey(0)
                if key == ord('y'):
                    mislabeled.extend(current)
                current = []

    return mislabeled

def put_canvas(current_batch, pool, none_directory):
    canvas = np.zeros((INPUT_DIM[1]*20, INPUT_DIM[0]*5, 1), dtype=np.uint8)
    def generator():
        for image_path in current_batch:
            yield image_path, none_directory
    current_images = pool.map(load_image, generator())
    for idx, image in enumerate(current_images):
        canvas[idx//5*INPUT_DIM[1]:(idx//5+1)*INPUT_DIM[1], idx%5*INPUT_DIM[0]:(idx%5+1)*INPUT_DIM[0]] = image
    return canvas

def load_image(arg):
    image_path, none_directory = arg
    image = cv2.imread(os.path.join(none_directory, image_path))
    image = image_from_resized_frame(image)
    return image

if __name__ == "__main__":
    args = parse_args()
    if len(args.raw) != 1:
        print("Must specify exactly 1 raw image dataset with --raw")
        exit(1)
    process_directory(args.raw[0])
   