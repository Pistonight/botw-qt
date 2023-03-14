import tensorflow as tf
import numpy as np

from common_util import INPUT_DIM

class Runner:
    def __init__(self, model_path):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        self.input_tensor_index = input_details[0]['index']
        self.output_tensor_index = output_details[0]['index']

        interpreter.allocate_tensors()
        self.interpreter = interpreter

    def run_image(self, image):
        self._execute(tf.constant([image], dtype=tf.float32))
        return self._extract()
    
    def _execute(self, input):
        self.interpreter.set_tensor(self.input_tensor_index, input)
        self.interpreter.invoke()
    
    def _extract(self):
        output = self.interpreter.get_tensor(self.output_tensor_index)[0]
        return _process_model_output(output)
    
class BatchRunner:
    def __init__(self, model_path, batch_size):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        self.input_tensor_index = input_details[0]['index']
        self.output_tensor_index = output_details[0]['index']
        if batch_size is not None:
            interpreter.resize_tensor_input(self.input_tensor_index, [batch_size, INPUT_DIM[1], INPUT_DIM[0], 1])

        interpreter.allocate_tensors()

        self.interpreter = interpreter
        self.batch_size = batch_size
    def run_batch(self, batch):
        run_size = len(batch)
        tensor = tf.constant(batch, dtype=tf.float32)
        if run_size < self.batch_size:
            tensor = tf.concat([tensor, tf.zeros([self.batch_size - run_size, INPUT_DIM[1], INPUT_DIM[0], 1], dtype=tf.float32)], axis=0)
        self._execute(tensor)
        return self._extract(run_size)
    
    def _execute(self, input):
        self.interpreter.set_tensor(self.input_tensor_index, input)
        self.interpreter.invoke()
    
    def _extract(self, size):
        output = self.interpreter.get_tensor(self.output_tensor_index)
        predicted_labels = []
        predicted_confidences = []
        for result in output[:size]:
            predicted_idx, confidence = _process_model_output(result)
            predicted_labels.append(predicted_idx)
            predicted_confidences.append(confidence)
        return predicted_labels, predicted_confidences
    
def _process_model_output(output_tensor):
    scores = tf.nn.softmax(output_tensor)
    predicted_idx = np.argmax(scores)
    confidence = 100 * scores[predicted_idx]
    return predicted_idx, confidence.numpy().item()

runner_instance = None
def init_runner_singleton(model_path, batch_size, lock):
    global runner_instance
    lock.acquire()
    # Print to the current line
    print("\033[A\033[A\r")
    runner_instance = BatchRunner(model_path, batch_size)
    lock.release()

def singleton_run_batch_with_paths(task):
    global runner_instance
    image_batch, label_batch, path_batch = task
    predicted_labels, predicted_confidences = runner_instance.run_batch(image_batch)
    out_image_paths = [ x.decode("utf-8") for x in path_batch ]
    actual_labels = [ x.item() for x in label_batch ]
    return out_image_paths, actual_labels, predicted_labels, predicted_confidences
