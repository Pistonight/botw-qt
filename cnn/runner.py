import tensorflow as tf
import numpy as np
import time

class ModelRunner:
    def __init__(self, model_path):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        time.sleep(0.5)
        print('\033[A\033[A\r')
        time.sleep(0.5)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        self.input_tensor_index = input_details[0]['index']
        self.output_tensor_index = output_details[0]['index']

        self.interpreter = interpreter

    def run_batch(self, image_batch):
        predicted_labels = []
        predicted_confidences = []
        for image in image_batch:
            predicted_idx, confidence = self.run_one(image)
            predicted_labels.append(predicted_idx)
            predicted_confidences.append(confidence)
        return predicted_labels, predicted_confidences

    def run_one(self, image):
        self.interpreter.set_tensor(self.input_tensor_index, np.array([image], dtype=np.float32))
        self.interpreter.invoke()
        scores = tf.nn.softmax(self.interpreter.get_tensor(self.output_tensor_index)[0])
        predicted_idx = np.argmax(scores)
        confidence = 100 * scores[predicted_idx]
        return predicted_idx, confidence.numpy().item()