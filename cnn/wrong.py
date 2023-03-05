import cv2
import json
import os
import subprocess
import inquirer
from common import decode_image, parse_args, display_image

def process_wrong(wrong_data, out, data_path, raw_path):
    for data in wrong_data:
        print(json.dumps(data, indent=2))
        image_name = subprocess.run(["grep", data["image"], data_path], stdout=subprocess.PIPE).stdout.decode("utf-8").split("#")[-1].strip() + ".png"
        data["image_name"] = image_name
        current_location = os.path.join(raw_path, data["actual"], image_name)
        predicted_location = os.path.join(raw_path, data["predicted"])
        predicted_location2 = os.path.join(raw_path+"_partial", data["predicted"])
        none_location = os.path.join(raw_path, "none")

        print(display_image(decode_image(data["image"])))
        choices = [
            "skip",
            f"move image to {predicted_location}",
            f"move image to {predicted_location2}",
            f"move image to {none_location}",
            "save to output"
        ]
        questions = [
            inquirer.List(
                'choice',
                message="Action?",
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
            os.rename(current_location, os.path.join(predicted_location, image_name))
        elif choice == 2:
            os.makedirs(predicted_location2, exist_ok=True)
            os.rename(current_location, os.path.join(predicted_location2, image_name))
        elif choice == 3:
            os.makedirs(none_location, exist_ok=True)
            os.rename(current_location, os.path.join(none_location, image_name))
        elif choice == 4:
            out.append(data)


if __name__ == "__main__":
    args = parse_args()
    if len(args.json) != 2:
        print("Wrong data and output data must be specified with --json")
        exit(1)
    if len(args.data) != 1:
        print("Data must be specified with --data")
        exit(1)
    if len(args.raw) != 1:
        print("Raw data must be specified with --raw")
        exit(1)
    with open(args.json[0], "r") as f:
        wrong_data = json.load(f)
    if os.path.exists(args.json[1]):
        with open(args.json[1], "r") as f:
            out = json.load(f)
    else:
        out = []
    process_wrong(wrong_data, out, args.data[0], args.raw[0])
    if out:
        with open(args.json[1], "w") as f:
            json.dump(out, f, indent=2)