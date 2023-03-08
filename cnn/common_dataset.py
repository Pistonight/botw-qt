import tensorflow as tf
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

def create_dataset(dataset_paths, batch_size, labels: list[str], workers=1, keep_path=False) -> tuple[tf.data.Dataset, int]:
    if workers == 1:
        workers = tf.data.AUTOTUNE
    image_paths = []
    image_labels = []
    total = 0
    for dataset_path in dataset_paths:
        # Process each data set
        for subpath in os.listdir(dataset_path):
            # subpath is quest name, lowercase, cleaned
            if subpath in labels:
                label = labels.index(subpath)
                for image_subpath in os.listdir(os.path.join(dataset_path, subpath)):
                    if image_subpath.endswith(".png"):
                        image_full_path = os.path.join(dataset_path, subpath, image_subpath)
                        image_paths.append(image_full_path)
                        image_labels.append(label)
        print(f"Loaded {len(image_paths)} images from {dataset_path}")
        total += len(image_paths)
    
    # Create a dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, image_labels))
    # Shuffle all images
    dataset = dataset.shuffle(len(image_paths), reshuffle_each_iteration=True)

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
