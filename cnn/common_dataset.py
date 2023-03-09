import tensorflow as tf
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

def create_dataset(dataset_paths, batch_size, labels: list[str], workers=1, keep_path=False, normalize=False) -> tuple[tf.data.Dataset, int]:
    if workers == 1:
        workers = tf.data.AUTOTUNE
    images = {} # quest -> list of images
    for quest in labels:
        images[quest] = []
    
    # Process each data set
    for dataset_path in dataset_paths:
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
            
    total = len(image_paths)
    
    # Create a dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, image_labels))
    # Shuffle all images
    dataset = dataset.shuffle(total, reshuffle_each_iteration=True)

    # Load image as binary array
    @tf.function
    def read_image(image_path, label):
        image = read_image_from(image_path)
        if not keep_path:
            return image, label
        return image, label, image_path
    dataset = dataset.map(read_image, num_parallel_calls=workers)
    # Batch images
    dataset = dataset.batch(batch_size)
    # Fix shape after batching, since tf cannot infer the shape
    if not keep_path:
        dataset = dataset.map(fix_shape)
    else:
        dataset = dataset.map(fix_shape_with_path)
    # Prefetch data
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset, total
