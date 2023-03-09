
import json
import os
import inquirer
import cv2
from common_util import parse_args, display_image, preinit_tensorflow, measure_str
if __name__ == "__main__":
    args = parse_args()
    if len(args.config) != 1:
        print("Need to specify data with --config/-c")
        exit(1)
    preinit_tensorflow()
from common_dataset import read_image_from

def process_wrong(wrong_data, wrong_path):
    start_time = measure_str()
    out = []
    wrong_data = [ d for d in wrong_data if os.path.exists(d["image"]) ]
    
    for i, data in enumerate(wrong_data):
        image_path = data["image"]
        image_name = os.path.basename(image_path)
        data_dir = os.path.dirname(image_path)
        current_label = os.path.basename(data_dir)
        data_dir = os.path.dirname(data_dir)
        predicted_label = data["predicted"]

        partial_dir = data_dir[:-4] if data_dir.endswith(".par") else data_dir+".par"
        non_partial_dir = data_dir[:-4] if data_dir.endswith(".par") else data_dir

        predicted_location = os.path.join(data_dir, predicted_label)
        predicted_partial_location = os.path.join(partial_dir, predicted_label)
        actual_partial_location = os.path.join(partial_dir, current_label)
        none_location = os.path.join(non_partial_dir, "none")

        image = read_image_from(image_path).numpy()
        print(display_image(image))
        cv2.destroyAllWindows()
        cv2.imshow(image_path, cv2.imread(image_path))
        cv2.waitKey(1)
        print()
        print(f"Image: {image_path}")
        print(f"Prediction: {predicted_label} @ {data['confidence']}")
        print(f"Actual: {current_label}")
        print()
        choices = [
            f"current label is correct (remove from {wrong_path})",
            f"move image to {predicted_location}",
            f"move image to {predicted_partial_location}",
            f"move image to {actual_partial_location}",
            f"move image to {none_location}",
            f"skip (keep in {wrong_path})",
            "enter a location manually"
        ]
        questions = [
            inquirer.List(
                'choice',
                message=f"[{i+1}/{len(wrong_data)}]",
                choices=choices,
            )
        ]
        answers = inquirer.prompt(questions)
        if not answers:
            exit(1)
        choice = choices.index(answers["choice"])
        if choice == 0:
            continue
        elif choice == 1:
            os.makedirs(predicted_location, exist_ok=True)
            os.rename(image_path, os.path.join(predicted_location, image_name))
        elif choice == 2:
            os.makedirs(predicted_partial_location, exist_ok=True)
            os.rename(image_path, os.path.join(predicted_partial_location, image_name))
        elif choice == 3:
            os.makedirs(actual_partial_location, exist_ok=True)
            os.rename(image_path, os.path.join(actual_partial_location, image_name))
        elif choice == 4:
            os.makedirs(none_location, exist_ok=True)
            os.rename(image_path, os.path.join(none_location, image_name))
        elif choice == 5:
            out.append(data)
        elif choice == 6:
            
            new_path = None
            while not new_path or not os.path.exists(new_path):
                new_path = input("Enter new folder (must exist): ")
            os.rename(image_path, os.path.join(new_path, image_name))
        
    
    cv2.destroyAllWindows()
    print(f"Done in {measure_str(start_time)}")
    return out

if __name__ == "__main__":
    
    with open(args.config[0], "r") as f:
        wrong_data = json.load(f)

    out = process_wrong(wrong_data, args.config[0])
    if out:
        with open(args.config[0], "w") as f:
            json.dump(out, f, indent=2)
    else:
        os.remove(args.config[0])