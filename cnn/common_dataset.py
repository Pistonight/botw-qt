import tensorflow as tf
import math
import random
import os

@tf.function
def read_image_from(image_path: tf.Tensor):
    image_content = tf.io.read_file(image_path)
    image = tf.image.decode_png(image_content, channels=1) / 255
    return image

def fix_shape(x: tf.Tensor, y: tf.Tensor):
    x.set_shape([None, None, None, 1])
    y.set_shape([None])
    return x, y

def fix_shape_with_path(x: tf.Tensor, y: tf.Tensor, path: tf.Tensor):
    x, y = fix_shape(x, y)
    path.set_shape([None])
    return x, y, path

def create_dataset(
    dataset_paths,
    batch_size,
    labels: list[str],
    workers=1,
    keep_path=False,
    normalize=False,
    augment_empty_factor=0,
    partial_as_none=False
) -> tuple[tf.data.Dataset, int]:
    if workers == 1:
        workers = tf.data.AUTOTUNE
    
    image_paths, image_labels = load_dataset_paths_and_labels(dataset_paths, labels, normalize, augment_empty_factor, partial_as_none)
    total = len(image_paths)
    dataset = create_dataset_from_paths_and_labels(image_paths, image_labels, batch_size, workers, keep_path)
    return dataset, total

def load_dataset_paths_and_labels(
    dataset_paths,
    labels: list[str],
    normalize=False,
    augment_empty_factor=0,
    partial_as_none=False
):
    images = {} # quest -> list of images
    for quest in labels:
        images[quest] = []
    
    # Process each data set
    for dataset_path in dataset_paths:
        is_partial = dataset_path.endswith(".par")
        total = 0
        # Process each quest in the data set
        for subpath in os.listdir(dataset_path):
            if subpath not in images:
                print(f"Warning: Ignoring unknown quest \"{subpath}\"")
                continue
            # Process each image in the quest
            for image_subpath in os.listdir(os.path.join(dataset_path, subpath)):
                if image_subpath.endswith(".png"):
                    image_full_path = os.path.join(dataset_path, subpath, image_subpath)
                    if is_partial and partial_as_none:
                        images["none"].append(image_full_path)
                    else:
                        images[subpath].append(image_full_path)
                    total += 1
        print(f"Detected {total} images from {dataset_path}")
    if normalize:
        extra_count = 0
        # Normalize the number of images per quest
        max_images = max([len(images[quest]) for quest in labels[1:]])
        for quest in labels[1:]:
            # If not enough images in the quest, randomly duplicate some
            while len(images[quest]) < max_images:
                images[quest].append(random.choice(images[quest]))
                extra_count += 1
        print(f"Duplicated {extra_count} images to normalize the dataset")

    image_paths = []
    image_labels = []
    for i, quest in enumerate(labels):
        for image_path in images[quest]:
            image_paths.append(image_path)
            image_labels.append(i)
    
    if augment_empty_factor > 0:
        for i in range(math.ceil(len(image_paths) * augment_empty_factor)):
            # Add an empty image for each image in the dataset
            image_paths.append("empty.png")
            image_labels.append(0)
    
    return image_paths, image_labels


def create_dataset_from_paths_and_labels(image_paths, image_labels, batch_size, workers=tf.data.AUTOTUNE, keep_path=False):
    total = len(image_paths)
    # Create a dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, image_labels))
    # Shuffle all images
    dataset = dataset.shuffle(total, reshuffle_each_iteration=True)

    # Load image as binary array
    @tf.function
    def read_image_with_label(image_path, label):
        image = read_image_from(image_path)
        if not keep_path:
            return image, label
        return image, label, image_path
    
    dataset = dataset.map(read_image_with_label, num_parallel_calls=workers)
    # Batch images
    dataset = dataset.batch(batch_size)
    # Fix shape after batching, since tf cannot infer the shape
    if not keep_path:
        dataset = dataset.map(fix_shape)
    else:
        dataset = dataset.map(fix_shape_with_path)
    # Prefetch data
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset