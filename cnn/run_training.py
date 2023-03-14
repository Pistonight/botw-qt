import os
from common_util import preinit_tensorflow, parse_args

if __name__ == "__main__":
    args = parse_args()
    help_msg = "Usage: python run_training.py --config <data_config> --config <training_config>"
    if len(args.config) != 2:
        print(help_msg)
        exit(1)
    preinit_tensorflow(use_gpu="gpu" in args.flags)


import tensorflow as tf
from keras import models, losses, regularizers, layers
import numpy as np
from tqdm.keras import TqdmCallback
import random
import yaml
from datetime import datetime

from common_util import import_labels, gen_seed, INPUT_DIM
from common_dataset import create_dataset


def run(data_path, config_path, workers):
    start_time = datetime.now()
    if workers > 1:
        print(f"Warning: Parallelism is set to {workers}. The script should automatically let TensorFlow use the optimal number of parallel threads. Ignore this message if it's not working properly without -j.")
    
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    with open(data_path, "r") as f:
        data_config = yaml.load(f, Loader=yaml.FullLoader)
        config["data"] = data_config["data"]

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

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"Seed: {seed}")

    labels = import_labels()

    print(f"\rLoading data...")

    partial_as_none = "partial_as_none" in config and bool(config["partial_as_none"])
    training_dataset, total= create_dataset(
        config["data"]["training"],
        config["batch"],
        labels,
        workers,
        normalize=True,
        augment_empty_factor=config["augment_empty"],
        partial_as_none=partial_as_none)
    validation_dataset, _ = create_dataset(config["data"]["validation"], config["batch"], labels, workers)

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
    )

    print(f"\rValidating...")
    validation_loss, validation_accuracy = model.evaluate(validation_dataset, verbose=2)
    config["validation"] = {
        "loss": validation_loss,
        "accuracy": validation_accuracy,
    }
    print(f"Loss: {validation_loss}, Accuracy: {validation_accuracy}")

    print(f"\rSaving model with seed {seed}...")
    save_model(model, config)
    delta = datetime.now() - start_time
    print(f"Done in {str(delta)}")


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
                config["convolution"]["features"],
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
    run(args.config[0], args.config[1], args.processes)