import cv2
import json
import os
import subprocess
import inquirer
from common_util import parse_args, display_image, preinit_tensorflow
if __name__ == "__main__":
    args = parse_args()
    if len(args.config) != 2:
        print("Wrong data and output data must be specified with --config/-c")
        exit(1)
    preinit_tensorflow()
from common_dataset import read_image_from

def process_wrong(wrong_data, out):
    wrong_data.sort(key=lambda x: x["confidence"], reverse=True)
    for i, data in enumerate(wrong_data):
        image_path = data["image"]
        if not os.path.exists(image_path):
            continue
        image_name = os.path.basename(image_path)
        data_dir = os.path.dirname(image_path)
        current_label = os.path.basename(data_dir)
        data_dir = os.path.dirname(data_dir)
        predicted_label = data["predicted"]

        predicted_location = os.path.join(data_dir, predicted_label, image_name)
        predicted_partial_location = os.path.join(data_dir+".par", predicted_label, image_name)
        actual_partial_location = os.path.join(data_dir+".par", current_label, image_name)
        none_location = os.path.join(data_dir, "none", image_name)

        image = read_image_from(image_path).numpy()
        print(display_image(image))
        print()
        print(f"Prediction: {predicted_label} @ {data['confidence']}")
        print(f"Actual: {current_label}")
        print()
        choices = [
            "current label is correct",
            f"move image to {predicted_location}",
            f"move image to {predicted_partial_location}",
            f"move image to {actual_partial_location}",
            f"move image to {none_location}",
            "save for manual processing later"
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

    # for data in wrong_data:
    #     print(json.dumps(data, indent=2))
    #     image_name = subprocess.run(["grep", data["image"], data_path], stdout=subprocess.PIPE).stdout.decode("utf-8").split("#")[-1].strip() + ".png"
    #     data["image_name"] = image_name
    #     current_location = os.path.join(raw_path, data["actual"], image_name)
    #     predicted_location = os.path.join(raw_path, data["predicted"])
    #     predicted_location2 = os.path.join(raw_path+"_partial", data["predicted"])
    #     none_location = os.path.join(raw_path, "none")

    #     print(display_image(decode_image(data["image"])))
    #     choices = [
    #         "skip",
    #         f"move image to {predicted_location}",
    #         f"move image to {predicted_location2}",
    #         f"move image to {none_location}",
    #         "save to output"
    #     ]
    #     questions = [
    #         inquirer.List(
    #             'choice',
    #             message="Action?",
    #             choices=choices,
    #         )
    #     ]
    #     answers = inquirer.prompt(questions)
    #     if not answers:
    #         exit(1)
    #     choice = choices.index(answers["choice"])
    #     if choice == 0:
    #         continue
    #     elif choice == 1:
    #         os.makedirs(predicted_location, exist_ok=True)
    #         os.rename(current_location, os.path.join(predicted_location, image_name))
    #     elif choice == 2:
    #         os.makedirs(predicted_location2, exist_ok=True)
    #         os.rename(current_location, os.path.join(predicted_location2, image_name))
    #     elif choice == 3:
    #         os.makedirs(none_location, exist_ok=True)
    #         os.rename(current_location, os.path.join(none_location, image_name))
    #     elif choice == 4:
    #         out.append(data)


if __name__ == "__main__":
    
    with open(args.config[0], "r") as f:
        wrong_data = json.load(f)
    if os.path.exists(args.config[1]):
        with open(args.config[1], "r") as f:
            out = json.load(f)
    else:
        out = []
    process_wrong(wrong_data, out)
    if out:
        with open(args.config[1], "w") as f:
            json.dump(out, f, indent=2)