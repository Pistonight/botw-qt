from common import parse_args, import_labels, preinit_tensorflow, INPUT_DIM
if __name__ == "__main__":
    args = parse_args()
    if len(args.json) != 1:
        print("Need exactly 1 json output file with --json")
        exit(1)
    preinit_tensorflow()

from keras import layers, models, regularizers
def declare_model():
    quest_labels = import_labels()
    model = models.Sequential()
    model.add(
        layers.Conv2D(
            1,
            (3, 3),
            activation='relu',
            input_shape=(INPUT_DIM[1], INPUT_DIM[0], 1),
            activity_regularizer=regularizers.L2(0.0015)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(1, (3, 3), activation='relu', activity_regularizer=regularizers.L2(0.0015)))
    model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(1, (3, 3), activation='relu', activity_regularizer=regularizers.L2(0.002)))
    # model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu', activity_regularizer=regularizers.L2(0.0015)))
    model.add(layers.Dense(len(quest_labels)))
    model.summary()
    return model

def save_model(path, model: models.Sequential):
    with open(path, "w") as f:
        f.write(model.to_json(indent=2))

if __name__ == "__main__":
    model = declare_model()
    save_model(args.json[0], model)
    print("Done")