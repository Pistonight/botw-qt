import os

if __name__ == "__main__":
    if os.name == "nt":
        print("Running on Windows is not supported. Please use WSL 2")
        exit(1)
from common import preinit_tensorflow, parse_args
if __name__ == "__main__":
    args = parse_args()
    if len(args.data) != 1:
        print("Must specify exactly 1 data file with --data")
        exit(1)
    if not args.model:
        print("Must specify a model with --model")
        exit(1)
    if len(args.json) != 1:
        print("Please specify exactly one json file wioth --json")
        exit(1)

    preinit_tensorflow()
import json
import tensorflow as tf
from tqdm import tqdm
import multiprocessing
import math

from common import import_labels, import_data, decode_image, input_from_image, measure_str
from runner import ModelRunner

def evaluate_model(model_path, data_path, output_path, processes):
    start_time = measure_str()
    print("Loading Data...")
    data = import_data(data_path)
    quest_labels = import_labels()

    image_bytes = []
    labels = []
    
    for quest_idx, encoded_images in enumerate(data):
        for encoded in encoded_images:
            image_bytes.append(encoded)
            label = [quest_idx]
            labels.append(label)

    #images, labels, image_bytes = decode_dataset(image_bytes, labels, "validation")
    #images = []
    
    print(f"\rRunning Model with {processes} threads...")

    predicted_labels = []
    predicted_confidences = []
    temp_image_bytes = []
    temp_labels = []
    def task():
        size_per_process = math.ceil(len(image_bytes) / processes)
        for i in range(processes):
            start = size_per_process * i
            end = size_per_process * (i + 1)
            yield i, model_path, image_bytes[start:end], labels[start:end]

    with multiprocessing.Pool(processes=processes) as pool:
        for b, l, predicted_idx, confidence in pool.imap_unordered(run_task, task()):
            temp_image_bytes.extend(b)
            temp_labels.extend(l)
            predicted_labels.extend(predicted_idx)
            predicted_confidences.extend(confidence)
    image_bytes = temp_image_bytes
    labels = temp_labels


    print("\rEvaluating...")
    
    wrong_idx_set = set()
    # run at 0% confidence to find all harmful wrongs
    eval_at_confidence(predicted_labels, predicted_confidences, labels, 0, wrong_idx_set)
    # run at 100% confidence first to see if it's possible to make model all correct
    correct, harmless_wrong, wrong, useful_correct, labeled_total = eval_at_confidence(predicted_labels, predicted_confidences, labels, 100, wrong_idx_set)
    if wrong > 0:
        print("BAD: Model cannot have 100% accuracy for any confidence level")
    else:
        # binary search for minimal confidence that will only make harmless wrongs
        confidence_lo = 0
        confidence_hi = 100
        while confidence_hi - confidence_lo > 0.01:
            confidence_mid = (confidence_lo + confidence_hi) / 2
            correct, harmless_wrong, wrong, useful_correct, labeled_total = eval_at_confidence(predicted_labels, predicted_confidences, labels, confidence_mid, wrong_idx_set)
            if wrong > 0:
                # search for a higher confidence
                confidence_lo = confidence_mid + 0.01
            else:
                # search for a lower confidence
                confidence_hi = confidence_mid - 0.01
        # fine tune confidence with linear search
        confidence = confidence_lo
        correct, harmless_wrong, wrong, useful_correct, labeled_total = eval_at_confidence(predicted_labels, predicted_confidences, labels, confidence, wrong_idx_set)
        while wrong > 0:
            confidence += 0.01
            correct, harmless_wrong, wrong, useful_correct, labeled_total = eval_at_confidence(predicted_labels, predicted_confidences, labels, confidence, wrong_idx_set)
        print()
        print(f"Minimum safe confidence threshold: {confidence}")

    total = len(labels)

    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Harmless Wrong: {harmless_wrong}")
    print(f"Wrong: {wrong}")
    print(f"Accuracy: {correct / total}")
    print(f"Useful: {useful_correct}/{labeled_total}")
    print()
    print(f"{len(wrong_idx_set)} wrong predictions were detected")

    with open(output_path, "w") as f:
        wrongs = []

        for i in wrong_idx_set:
            actual_idx = labels[i][0]
            predicted_idx = predicted_labels[i]
            predicted_confidence = predicted_confidences[i]
            harmful = bool(predicted_idx != 0 and predicted_idx != actual_idx)
            wrongs.append({
                "harmful": harmful,
                "image": image_bytes[i],
                "predicted": quest_labels[predicted_idx],
                "confidence": predicted_confidence,
                "actual": quest_labels[actual_idx]
            })

        json.dump(wrongs, f, indent=2)

    print(f"Wrongs saved to {output_path}")
    print(f"Done in {measure_str(start_time)}")

def run_task(args):
    runner_idx, model_path, image_bytes, labels = args
    runner = ModelRunner(model_path)
    predicted_labels = []
    predicted_confidences = []
    for encoded in tqdm(image_bytes, leave=False, desc=f"Thread {runner_idx}", position=runner_idx):
        predicted_idx, confidence = runner.run_one(input_from_image(decode_image(encoded)))
        predicted_labels.append(predicted_idx)
        predicted_confidences.append(confidence)
    return image_bytes, labels, predicted_labels, predicted_confidences

def eval_at_confidence(predicted_labels, predicted_confidences, actual_labels, confidence, wrong_idx_set):
    print(f"Testing confidence = {confidence}")
    correct = 0
    useful_correct = 0
    labeled_total = 0
    harmless_wrong = 0
    wrong = 0
    for i in range(len(actual_labels)):
        actual_idx = actual_labels[i][0]
        if actual_idx != 0:
            labeled_total += 1
        predicted_idx = predicted_labels[i]
        predicted_confidence = predicted_confidences[i]
        # If not confident enough, predict not quest banner
        if predicted_confidence < confidence:
            predicted_idx = 0
        # correct case
        if actual_idx == predicted_idx:
            correct += 1
            if actual_idx != 0:
                useful_correct += 1
        elif predicted_idx == 0:
            harmless_wrong += 1
            # we need to check if the model actually predicted wrong, or not confident enough
            if predicted_labels[i] != actual_idx:
                # if actually incorrect, add it to the harmless wrong set
                wrong_idx_set.add(i)
        else:
            # incorrect, and predicted a wrong quest banner
            wrong += 1
            wrong_idx_set.add(i)
    return correct, harmless_wrong, wrong, useful_correct, labeled_total


if __name__ == "__main__":
    if args.processes > 1:
        # Only use CPU, since we don't have enough GPUs for multi processing
        tf.config.set_visible_devices([], 'GPU')
    
    evaluate_model(args.model, args.data[0], args.json[0], args.processes)