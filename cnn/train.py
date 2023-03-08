import os

# if __name__ == "__main__":
#     if os.name == "nt":
#         print("Running on Windows is not supported. Please use WSL 2")
#         exit(1)
    
from common_util import preinit_tensorflow, parse_args

if __name__ == "__main__":
    args = parse_args()
    if len(args.config) != 1:
        print("Need exactly 1 training yaml config file with --config/-c")
        exit(1)
    preinit_tensorflow(args.processes if args.processes > 1 else None)


import tensorflow as tf
from keras import models, losses, regularizers, layers
import numpy as np
from tqdm.keras import TqdmCallback
import math
import json
import yaml
import hashlib
from datetime import datetime
import multiprocessing
from tqdm import tqdm


from common_util import import_labels, gen_seed, INPUT_DIM
from common_dataset import create_dataset


def create_model(config, num_labels) -> models.Sequential:
    model = models.Sequential()
    # Randomly shift the image
    model.add(
        layers.RandomTranslation(
            input_shape=(INPUT_DIM[1], INPUT_DIM[0], 1),
            height_factor=config["random_translation"]["height"],
            width_factor=config["random_translation"]["width"],
            fill_mode='nearest',
            interpolation='bilinear',
            seed=None,
        )
    )
    # Randomly add noise
    model.add(
        layers.GaussianNoise(
            0.1
        )
    )
    # Convolutional layers
    for _ in range(config["convolution"]["rounds"]):
        model.add(
            layers.Conv2D(
                2,
                (3, 3),
                activation='relu',
                activity_regularizer=regularizers.L1L2(
                    l1=config["convolution"]["l1"],
                    l2=config["convolution"]["l2"]
                )
            )
        )
        model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    # Dense layer
    model.add(
        layers.Dense(
            config["dense"]["size"],
            activation='relu',
            activity_regularizer=regularizers.L1L2(
                l1=config["dense"]["l1"],
                l2=config["dense"]["l2"]
            )
        )
    )
    # Output layer
    model.add(layers.Dense(num_labels))

    model.compile(
        optimizer='adam',
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.summary()

    return model

# def create_data(config, name, labels: list[str], workers=1) -> tf.data.Dataset:
#     if workers == 1:
#         workers = tf.data.AUTOTUNE
#    # os.makedirs("cache", exist_ok=True)
#     #sha256 = hashlib.sha256()

#     image_paths = []
#     image_labels = []
#     for dataset_path in config["data"][name]:
#         # Process each data set
#         for subpath in os.listdir(dataset_path):
#             # subpath is quest name, lowercase, cleaned
#             if subpath in labels:
#                 label = labels.index(subpath)
#                 for image_subpath in os.listdir(os.path.join(dataset_path, subpath)):
#                     if image_subpath.endswith(".png"):
#                         image_full_path = os.path.join(dataset_path, subpath, image_subpath)
#                         #sha256.update(image_full_path.encode("utf-8"))
#                         image_paths.append(image_full_path)
#                         image_labels.append(label)
#     # signature = sha256.hexdigest()
#     print(f"Loaded {len(image_paths)} images for {name} data set")
    
#     # Create a dataset
#     dataset = tf.data.Dataset.from_tensor_slices((image_paths, image_labels))
#     # Shuffle all images
#     dataset = dataset.shuffle(len(image_paths), reshuffle_each_iteration=True)

#     # Load image as binary array
#     @tf.function
#     def read_image(image_path, label):
#         image_content = tf.io.read_file(image_path)
#         image = tf.image.decode_png(image_content, channels=1) / 255
#         return image, label
#     dataset = dataset.map(read_image, num_parallel_calls=workers)
#     #dataset = dataset.cache()
#     # Batch images
#     dataset = dataset.batch(config["batch"])
#     # Fix shape after batching, since tf cannot infer the shape
#     def fix_shape(x,y):
#         x.set_shape([None, None, None, 1])
#         y.set_shape([None])
#         return x, y
#     dataset = dataset.map(fix_shape)
#     # Prefetch data
#     dataset = dataset.prefetch(tf.data.AUTOTUNE)
#     return dataset

# def create_dataset(image_paths, batch_size, seed=None, workers=tf.data.AUTOTUNE):
#     # get the parent directory of the first image
#     parent_dir = os.path.basename(os.path.dirname(image_paths[0]))

    # def get_data(i):
    #     #i = i.numpy()
    #     return dataset[i]
    # def fix_shape(x,y):
    #     x.set_shape([None, None, None, 1])
    #     y.set_shape([None])
    #     return x,y

    # z = list(range(len(dataset)))
    # ds = tf.data.Dataset.from_generator(lambda: z, tf.uint32)
    # if seed:
    #     ds = ds.shuffle(len(dataset), seed=seed, reshuffle_each_iteration=True)
    # ds = ds.map(get_data, num_parallel_calls=workers)
    # ds = ds.batch(batch_size)
    # ds = ds.map(fix_shape)
    # ds = ds.prefetch(tf.data.AUTOTUNE)
    # return ds

# class DataSetAdapter(utils.Sequence):
#     def __init__(self, dataset, batch_size, shuffle=False):
#         self.dataset = dataset
#         self.data_order = np.arange(len(self.dataset), dtype=np.int32)
#         self.shuffle = shuffle
#         self.batch_size = batch_size
#         self.on_epoch_end()

#     def __len__(self):
#         return math.ceil(len(self.dataset) / self.batch_size)

#     def __getitem__(self, index):
#         batch_indices = self.data_order[index * self.batch_size:(index + 1) * self.batch_size]
#         return self.__data_generation(batch_indices)
        
#     def __data_generation(self, batch_indices):
#         batch_images = np.empty((len(batch_indices), INPUT_DIM[1], INPUT_DIM[0], 1), dtype=np.float32)
#         batch_labels = np.empty((len(batch_indices)), dtype=int)

#         for batch_item_idx, data_i in enumerate(batch_indices):
#             entry = self.dataset[data_i]
#             batch_images[batch_item_idx,] = entry.get_input()
#             batch_labels[batch_item_idx] = entry.label
#         return batch_images, batch_labels

#     def on_epoch_end(self):
#         if self.shuffle:
#             np.random.shuffle(self.data_order)

def run(config_path, workers):
    start_time = datetime.now()
    #print(f"Using {workers} workers")
    
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    seed = config["seed"]
    if seed is None:
        seed = gen_seed()
        for _ in range(100):
            # attempt to regenerate seed up to 100 times
            if os.path.exists(model_path(seed)):
                seed = gen_seed()
                continue
            break
        config["seed"] = seed

    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"Seed: {seed}")

    labels = import_labels()

    print(f"\rLoading data...")
    training_dataset, total= create_dataset(config["data"]["training"], config["batch"], labels, workers)
    validation_dataset, _ = create_dataset(config["data"]["validation"], config["batch"], labels, workers)

    #validation_dataset = DataSets([ DataSet(labels, validation_data) for validation_data in validation_data_config ])
    
    # training_dataset, validation_dataset = create_dataset(training_data_paths, validation_data_paths, BATCH_SIZE)

    # image shape: (size, height, width, channels=1)
    # label shape: (size)

    print(f"\rLoading model...")
    model = create_model(config, len(labels))
    
    print(f"\rTraining...")
    epochs = config["epoch"]
    model.fit(
        training_dataset,
        epochs=epochs, 
        validation_data=validation_dataset,
        verbose=0, # turning off default keras output and using TqdmCallback instead
        callbacks=[TqdmCallback(epochs=epochs, batch_size=config["batch"], data_size=total, leave=False)],
        # use_multiprocessing=workers>1,
        # workers=workers
    )

    print(f"\rSaving model with seed {seed}...")
    save_model(model, config)
    delta = datetime.now() - start_time
    print(f"Done in {str(delta)}")


# def create_dataset(training_data_paths, validation_data_paths, batch_size):
#     non_validation_image_data = []
#     non_validation_labels = []
#     validation_image_data = []
#     validation_labels = []

#     for data_path in training_data_paths:
#         import_dataset_from(data_path, non_validation_image_data, non_validation_labels)
#     for data_path in validation_data_paths:
#         import_dataset_from(data_path, validation_image_data, validation_labels)

#     total = len(non_validation_image_data) + len(validation_image_data)

#     # Make training set a multiple of batch size
#     extras = len(non_validation_image_data) % batch_size
#     for _ in range(extras):
#         i = np.random.randint(0, len(non_validation_image_data) - 1)
#         validation_image_data.append(non_validation_image_data[i])
#         validation_labels.append(non_validation_labels[i])
#         del non_validation_image_data[i]
#         del non_validation_labels[i]

#     assert len(non_validation_image_data) % batch_size == 0
#     assert len(non_validation_image_data) + len(validation_image_data) == total

#     return DataSet(non_validation_image_data, non_validation_labels, batch_size, shuffle=True), DataSet(validation_image_data, validation_labels, batch_size, shuffle=False)
    
# def import_dataset_from(data_path, out_image_data, out_labels):
#     data = import_data(data_path)

#     for quest_idx, encoded_images in enumerate(data):
#         for encoded in encoded_images:
#             image_bytes = bytes(encoded, "utf-8")
#             label = quest_idx
#             out_image_data.append(image_bytes)
#             out_labels.append(label)

def save_model(keras_model: models.Sequential, config):
    seed = config["seed"]
    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()

    # Save the model.
    os.makedirs("models", exist_ok=True)
    with open(model_path(seed), 'wb') as f:
        f.write(tflite_model)

    with open(f"models/{seed}.yaml", "w") as f:
        yaml.dump(config, f)

def model_path(seed):
    return f"models/{seed}.tflite"

if __name__ == "__main__":
    tf.config.set_visible_devices([], 'GPU')
    run(args.config[0], args.processes)