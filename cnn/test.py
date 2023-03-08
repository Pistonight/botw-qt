TF_LOGGING = False # Set to true to enable TensorFlow logging
import sys
import os
from common_util import import_labels, image_from_whole_frame, image_from_cropped_frame, get_image_score, is_score_valid, input_from_image
import cv2

# Quest banner
BANNER_V_START = 0.19
BANNER_V_END = 0.28
BANNER_H_START = 0.23
BANNER_H_END = 0.77

def test_image(should_crop, image_path, model_path):
    image = cv2.imread(image_path)
    if should_crop:
        image = image_from_whole_frame(image)
    else:
        image = image_from_cropped_frame(image)
    score = get_image_score(image)
    
    if not is_score_valid(score):
        print("Prediction: none")
        print("Reason: Filtered out by ratio of black pixels")
        return
    
    print("Initializing TensorFlow...")
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    from common_runner import ModelRunner
    runner = ModelRunner(model_path)
    quest_labels = import_labels()
    predicted, confidence = runner.run_one(input_from_image(image))
    print(f"Prediction: {quest_labels[predicted]}")
    print(f"Confidence: {confidence}")

if __name__ == "__main__":
    if not TF_LOGGING:
        # Set TensorFlow Verbosity to Error
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["AUTOGRAPH_VERBOSITY"] = "0"
    if len(sys.argv) < 3:
        print("Usage: python evaluate.py <model_path> <image_path> [--no-crop]")
        exit(1)
    test_image("--no-crop" not in sys.argv, sys.argv[2], sys.argv[1])