import os
import tensorflow as tf
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from utils import numpy_to_tensorflow_dataset, transport_wav_to_numpy


class PredictTestData(QThread):
    signal_predict_data_info = pyqtSignal(str)

    signal_predict_class_info = pyqtSignal(int, int)

    signal_predict_result = pyqtSignal(int)

    def __init__(self):
        super(PredictTestData, self).__init__()
        self.model = None
        self.model_path = None
        self.data_path = None

    def accept_signal_data_path(self, data_path):
        self.data_path = data_path

    def accept_signal_model_path(self, model_path):
        self.model_path = model_path
        self.signal_predict_data_info.emit(f"Load model from {self.model_path}")
        self.model = tf.keras.models.load_model(self.model_path)

    def predict_single_file(self, data):
        train_data = numpy_to_tensorflow_dataset(data)

        pre = self.model.predict(train_data)
        result = list([0 for _ in range(pre.shape[1])])

        for single_slice_pre in pre:
            result[np.argmax(single_slice_pre)] += 1

        predict_class = np.argmax(result)
        return predict_class, result

    def run(self) -> None:
        self.signal_predict_data_info.emit(f"Load data from {self.data_path}")
        # label_list = [0, 2, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 2]

        for file_dir in os.listdir(self.data_path):
            right_num = 0
            total_num = len(os.listdir(os.path.join(self.data_path, file_dir)))
            self.signal_predict_class_info.emit(int(file_dir), total_num)

            if total_num == 0:
                continue

            for file in os.listdir(os.path.join(self.data_path, file_dir)):
                file_type = file.split('.')[-1]
                if file_type != 'wav' and file_type != "WAV":
                    self.signal_predict_data_info.emit(f"Error: file {file} type is not wav!")
                    continue

                numpy_data = transport_wav_to_numpy(os.path.join(self.data_path, file_dir, file))

                predict_class, result = self.predict_single_file(numpy_data)
                # print(label_list[int(file_dir)], label_list[predict_class])
                if str(predict_class) == file_dir:
                    sign = "√"
                    right_num += 1
                else:
                    sign = "x"
                self.signal_predict_result.emit(right_num)

                print(f"{file} it's class is {file_dir}, predict class is {predict_class} {sign} {result}")
                self.signal_predict_data_info.emit(f"{file} it's class is {file_dir}, predict class is {predict_class} "
                                                   f"{sign} {result}")

            print(
                f"Class {file_dir} predict right num {right_num}, total num {total_num}, {round(right_num / total_num * 100, 2)}%\n")
            self.signal_predict_data_info.emit(f"Class {file_dir} predict right num {right_num}, total num {total_num},"
                                               f" {round(right_num / total_num * 100, 2)}%\n")


        self.signal_predict_data_info.emit("Predict finished!")
        self.signal_predict_data_info.emit("Inner info: Predict finished!")

