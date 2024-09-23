import tensorflow as tf
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from utils import transport_wav_to_numpy, numpy_to_tensorflow_dataset


class PredictUnknownData(QThread):
    signal_predict_data_info = pyqtSignal(str)

    signal_predict_result = pyqtSignal(list)

    def __init__(self):
        super(PredictUnknownData, self).__init__()
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

        return result

    def run(self) -> None:
        self.signal_predict_data_info.emit(f"Load data from {self.data_path}")

        numpy_data = transport_wav_to_numpy(self.data_path)

        result = self.predict_single_file(numpy_data)
        self.signal_predict_data_info.emit(self.data_path.split('/')[-1] + ' ' + str(result))

        self.signal_predict_result.emit(result)

        self.signal_predict_data_info.emit("Predict finished!")
        self.signal_predict_data_info.emit("Inner info: Predict finished!")

