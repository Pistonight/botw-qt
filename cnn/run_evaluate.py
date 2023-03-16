from common_util import preinit_tensorflow, parse_args
if __name__ == "__main__":
    args = parse_args()
    if not args.model:
        print("Must specify a model with --model/-m")
        exit(1)
    if not args.config:
        print("Please specify output json file with --config/-c")
        exit(1)

    preinit_tensorflow()
import json
import yaml
from tqdm import tqdm
import math
import multiprocessing

from common_util import import_labels, measure_str
from common_runner import init_runner_singleton, singleton_run_batch_with_paths
from common_dataset import create_dataset

def evaluate_model(model_path, data_paths, output_path, processes):
    start_time = measure_str()
    quest_labels = import_labels()
    batch_size = processes

    # Use half of the workers for dataset loading and half for running the model
    workers = processes//2

    dataset, total = create_dataset(data_paths, batch_size, quest_labels, workers=workers, keep_path=True)

    image_paths = []
    actual_labels = []
    predicted_labels = []
    predicted_confidences = []

    lock = multiprocessing.RLock()
    tqdm.set_lock(lock)

    print("Running model...")
    print()
    with multiprocessing.Pool(processes=workers, initializer=init_runner_singleton, initargs=(model_path, batch_size, lock)) as pool:
        for image_paths_batch, actual_labels_batch, predicted_labels_batch, predicted_confidences_batch in tqdm(
            pool.imap(singleton_run_batch_with_paths, dataset.as_numpy_iterator()),
            total=math.ceil(total / batch_size), unit="batch", leave=False
        ):
            image_paths.extend(image_paths_batch)
            actual_labels.extend(actual_labels_batch)
            predicted_labels.extend(predicted_labels_batch)
            predicted_confidences.extend(predicted_confidences_batch)

    print("\rEvaluating...")
    
    wrong_idx_set = set()
    # run at 0% confidence to find all harmful wrongs
    eval_at_confidence(len(quest_labels), predicted_labels, predicted_confidences, actual_labels, 0, wrong_idx_set)
    # run at 100% confidence first to see if it's possible to make model all correct
    correct, harmless_wrong, wrong, useful_corrects, labeled_totals = eval_at_confidence(len(quest_labels), predicted_labels, predicted_confidences, actual_labels, 100, wrong_idx_set)
    if wrong > 0:
        print("BAD: Model cannot have 100% accuracy for any confidence level")
    else:
        # binary search for minimal confidence that will only make harmless wrongs
        confidence_lo = 0
        confidence_hi = 100
        while confidence_hi - confidence_lo > 0.01:
            confidence_mid = (confidence_lo + confidence_hi) / 2
            correct, harmless_wrong, wrong, useful_corrects, labeled_totals = eval_at_confidence(len(quest_labels), predicted_labels, predicted_confidences, actual_labels, confidence_mid, wrong_idx_set)
            if wrong > 0:
                # search for a higher confidence
                confidence_lo = confidence_mid + 0.01
            else:
                # search for a lower confidence
                confidence_hi = confidence_mid - 0.01
        # fine tune confidence with linear search
        confidence = confidence_lo
        correct, harmless_wrong, wrong, useful_corrects, labeled_totals = eval_at_confidence(len(quest_labels), predicted_labels, predicted_confidences, actual_labels, confidence, wrong_idx_set)
        while wrong > 0:
            confidence += 0.01
            correct, harmless_wrong, wrong, useful_corrects, labeled_totals = eval_at_confidence(len(quest_labels), predicted_labels, predicted_confidences, actual_labels, confidence, wrong_idx_set)
        print()
        confidence = min(confidence, 100)
        print(f"Minimum safe confidence threshold: {confidence}")

    total = len(actual_labels)

    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Harmless Wrong: {harmless_wrong}")
    print(f"Wrong: {wrong}")
    print()
    useful_percentages = [ (i, 0) if labeled_totals[i] == 0 else (i, useful_corrects[i] / labeled_totals[i]) for i in range(1, len(quest_labels)) ]
    print("Lowest 5 Quests:")
    useful_percentages.sort(key=lambda x: x[1])
    for i in range(5):
        idx, percentage = useful_percentages[i]
        print(f"{quest_labels[idx]}: {percentage}")
    print()
    print("Highest 5 Quests:")
    useful_percentages.sort(key=lambda x: x[1], reverse=True)
    for i in range(5):
        idx, percentage = useful_percentages[i]
        print(f"{quest_labels[idx]}: {percentage}")
    print()
    print("Median 5 Quests:")
    for i in range(len(useful_percentages)//2-2, len(useful_percentages)//2+3):
        idx, percentage = useful_percentages[i]
        print(f"{quest_labels[idx]}: {percentage}")
    print()
    print(f"Accuracy: {correct / total}")
    print(f"Usefulness: {sum(useful_corrects) / sum(labeled_totals)}")
    print(f"Overall score: {sum(useful_corrects) / sum(labeled_totals) * correct / total}")
    print()
    print(f"{len(wrong_idx_set)} wrong predictions were detected")

    with open(output_path, "w") as f:
        wrongs = []

        for i in wrong_idx_set:
            actual_idx = actual_labels[i]
            predicted_idx = predicted_labels[i]
            predicted_confidence = predicted_confidences[i]
            harmful = bool(predicted_idx != 0 and predicted_idx != actual_idx)
            wrongs.append({
                "harmful": harmful,
                "image": image_paths[i],
                "predicted": quest_labels[predicted_idx],
                "confidence": predicted_confidence,
            })

        wrongs.sort(key=lambda x: x["confidence"], reverse=True)

        json.dump(wrongs, f, indent=2)

    print(f"Wrongs saved to {output_path}")
    print(f"Done in {measure_str(start_time)}")


def eval_at_confidence(num_labels, predicted_labels, predicted_confidences, actual_labels, confidence, wrong_idx_set):
    #print(f"Testing confidence = {confidence}")
    correct = 0
    useful_corrects = [0] * num_labels
    labeled_totals = [0] * num_labels
    harmless_wrong = 0
    wrong = 0
    for i in range(len(actual_labels)):
        actual_idx = actual_labels[i]
        if actual_idx != 0:
            labeled_totals[actual_idx] += 1
        predicted_idx = predicted_labels[i]
        predicted_confidence = predicted_confidences[i]
        # If not confident enough, predict not quest banner
        if predicted_confidence < confidence:
            predicted_idx = 0
        # correct case
        if actual_idx == predicted_idx:
            correct += 1
            if actual_idx != 0:
                useful_corrects[actual_idx] += 1
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
    return correct, harmless_wrong, wrong, useful_corrects, labeled_totals


if __name__ == "__main__":
    output_path = args.config[0]
    data_paths = set()
    for data_path in args.data:
        data_paths.add(data_path)
    use_training_data = "use-training-data" in args.flags
    for config_path in args.config[1:]:
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        for data_path in config["data"]["validation"]:
            data_paths.add(data_path)
        if use_training_data:
            for data_path in config["data"]["training"]:
                data_paths.add(data_path)

    
    evaluate_model(args.model, list(data_paths), output_path, args.processes)