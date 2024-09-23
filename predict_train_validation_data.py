from utils import numpy_to_tensorflow_dataset
from PyQt5.QtCore import QThread, pyqtSignal

import numpy as np
import os
import tensorflow as tf


class PredictTrainValidationData(QThread):
    signal_predict_data_info = pyqtSignal(str)

    signal_predict_class_info = pyqtSignal(int, int)

    signal_predict_result = pyqtSignal(int)

    def __init__(self):
        super(PredictTrainValidationData, self).__init__()
        self.model = None
        self.model_path = None
        self.data_path = None

    def accept_signal_data_path(self, data_path):
        self.data_path = data_path

    def accept_signal_model_path(self, model_path):
        self.model_path = model_path
        self.signal_predict_data_info.emit(f"Load model from {self.model_path}")
        self.model = tf.keras.models.load_model(self.model_path)

    def predict_in_memory(self, data):
        train_data = numpy_to_tensorflow_dataset(data)

        pre = self.model.predict(train_data)

        result = list([0 for _ in range(pre.shape[1])])

        for single_slice_pre in pre:
            result[np.argmax(single_slice_pre)] += 1

        print(result)
        predict_class = np.argmax(result)

        return predict_class, result

    def run(self):
        try:
            self.signal_predict_data_info.emit(f"Load data from {self.data_path}")
            for file_dir in os.listdir(self.data_path):
                total_file_num = len(os.listdir(os.path.join(self.data_path, file_dir)))
                self.signal_predict_class_info.emit(int(file_dir), total_file_num)
                current_file_index = 0
                predict_right_num = 0
                for file in os.listdir(os.path.join(self.data_path, file_dir)):
                    predict_data = np.load(os.path.join(self.data_path, file_dir, file))
                    predict_result = self.predict_in_memory(predict_data)

                    if str(file_dir) == str(predict_result[0]):
                        sign = "√"
                        predict_right_num += 1
                    else:
                        sign = "x"
                    self.signal_predict_result.emit(predict_right_num)
                    current_file_index += 1
                    self.signal_predict_data_info.emit(f"{file_dir} {file} slice predict: {predict_result[1]}, predict "
                                                       f"class is  {predict_result[0]} {sign} {current_file_index}/"
                                                       f"{total_file_num}")

                self.signal_predict_data_info.emit(f"Class {file_dir} predict right percent "
                                                   f"{round(predict_right_num / total_file_num, 4) * 100}%\n")
                print(f"Class {file_dir} predict right percent {round(predict_right_num / total_file_num, 4) * 100}%\n")

            self.signal_predict_data_info.emit("Predict finished!")
            self.signal_predict_data_info.emit("Inner info: Predict finished!")

        except Exception as e:
            print(e)
