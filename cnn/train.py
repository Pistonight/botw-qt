import os

if __name__ == "__main__":
    if os.name == "nt":
        print("Running on Windows is not supported. Please use WSL 2")
        exit(1)
    
from common import preinit_tensorflow, parse_args

if __name__ == "__main__":
    args = parse_args()
    if not args.data:
        print("Need at least 1 training data set with --data")
        exit(1)
    if not args.data2:
        print("Need at least 1 validation data set with --data2")
        exit(1)
    if len(args.json) != 1:
        print("Need exactly 1 json model file with --json")
        exit(1)
    preinit_tensorflow(args.processes if args.processes > 1 else None)


import tensorflow as tf
from keras import models, losses, utils
import numpy as np
from tqdm.keras import TqdmCallback
import math
import json
from datetime import datetime
import multiprocessing
from tqdm import tqdm

from common import decompress_encoded_image, decode_decompressed_image_to_input, import_data, input_from_image, gen_seed, INPUT_DIM

SEED = None # leave as None to use a random seed
BATCH_SIZE = 16
EPOCHS = 5

class DataSet(utils.Sequence):
    def __init__(self, image_bytes, labels, batch_size, shuffle=False):
        with multiprocessing.Pool() as pool:
            self.image_decompressed_bytes = pool.map(decompress_encoded_image, tqdm(image_bytes, desc="Decompressing images", leave=False))
        self.labels = labels
        self.total = len(self.image_decompressed_bytes)
        assert len(self.image_decompressed_bytes) == len(self.labels)
        self.data_order = np.arange(self.total, dtype=np.int32)
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(self.total / self.batch_size)

    def __getitem__(self, index):
        batch_indices = self.data_order[index * self.batch_size:(index + 1) * self.batch_size]
        return self.__data_generation(batch_indices)
        
    def __data_generation(self, batch_indices):
        batch_images = np.empty((self.batch_size, INPUT_DIM[1], INPUT_DIM[0], 1), dtype=np.float32)
        batch_labels = np.empty((self.batch_size), dtype=int)

        for batch_item_idx, data_i in enumerate(batch_indices):
            batch_images[batch_item_idx,] = decode_decompressed_image_to_input(self.image_decompressed_bytes[data_i])
            batch_labels[batch_item_idx] = self.labels[data_i]
        return batch_images, batch_labels

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.data_order)

def run(model_config_path, training_data_paths, validation_data_paths, workers):
    start_time = datetime.now()
    if SEED is None:
        seed = gen_seed()
        for _ in range(100):
            # attempt to regenerate seed up to 100 times
            if os.path.exists(model_path(seed)):
                seed = gen_seed()
                continue
            break
    else:
        seed = SEED
    np.random.seed(seed)
    parameters = {
        "seed": seed,
        "epochs": EPOCHS,
        "training_data": training_data_paths,
        "validation_data": validation_data_paths,
        "model": model_config_path,
    }

    print(f"\rLoading data...")
    training_dataset, validation_dataset = create_dataset(training_data_paths, validation_data_paths, BATCH_SIZE)

    # image shape: (size, height, width, channels=1)
    # label shape: (size)

    print(f"\rLoading model config...")
    with open(model_config_path, "r") as f:
        model = models.model_from_json(f.read())

    model.compile(
        optimizer='adam',
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    print(f"\rTraining using {training_dataset.total} training images and {validation_dataset.total} validation images...")
    
    model.fit(
        training_dataset,
        epochs=EPOCHS, 
        shuffle=False,
        validation_data=validation_dataset,
        verbose=0, # turning off default keras output and using TqdmCallback instead
        callbacks=[TqdmCallback(epochs=EPOCHS, batch_size=BATCH_SIZE, data_size=training_dataset.total, leave=False)],
        use_multiprocessing=workers>1,
        workers=workers
    )

    print("\rValidating...")
    test_loss, test_acc = model.evaluate(
        validation_dataset,
        verbose=2,
        use_multiprocessing=workers>1,
        workers=workers
    )
    print(f"\rAccuracy: {test_acc}")
    parameters["accuracy"] = test_acc
    parameters["loss"] = test_loss

    print(f"Saving model with seed {seed}...")
    save_model(model, seed, parameters)
    delta = datetime.now() - start_time
    print(f"Done in {str(delta)}")


def create_dataset(training_data_paths, validation_data_paths, batch_size):
    non_validation_image_data = []
    non_validation_labels = []
    validation_image_data = []
    validation_labels = []

    for data_path in training_data_paths:
        import_dataset_from(data_path, non_validation_image_data, non_validation_labels)
    for data_path in validation_data_paths:
        import_dataset_from(data_path, validation_image_data, validation_labels)

    total = len(non_validation_image_data) + len(validation_image_data)

    # Make training set a multiple of batch size
    extras = len(non_validation_image_data) % batch_size
    for _ in range(extras):
        i = np.random.randint(0, len(non_validation_image_data) - 1)
        validation_image_data.append(non_validation_image_data[i])
        validation_labels.append(non_validation_labels[i])
        del non_validation_image_data[i]
        del non_validation_labels[i]

    assert len(non_validation_image_data) % batch_size == 0
    assert len(non_validation_image_data) + len(validation_image_data) == total

    return DataSet(non_validation_image_data, non_validation_labels, batch_size, shuffle=True), DataSet(validation_image_data, validation_labels, batch_size, shuffle=False)
    
def import_dataset_from(data_path, out_image_data, out_labels):
    data = import_data(data_path)

    for quest_idx, encoded_images in enumerate(data):
        for encoded in encoded_images:
            image_bytes = bytes(encoded, "utf-8")
            label = quest_idx
            out_image_data.append(image_bytes)
            out_labels.append(label)

def save_model(keras_model: models.Sequential, seed, parameters):
    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()

    # Save the model.
    os.makedirs("models", exist_ok=True)
    with open(model_path(seed), 'wb') as f:
        f.write(tflite_model)

    with open(f"models/{seed}.param.json", "w") as f:
        json.dump(parameters, f, indent=2)

def model_path(seed):
    return f"models/{seed}.tflite"

if __name__ == "__main__":
    run(args.json[0], args.data, args.data2, args.processes)