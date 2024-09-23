# coding=utf-8

"""
processing data
"""
import sys

import os
from utils import *
import constants
from PyQt5.QtCore import QThread, pyqtSignal


class PreprocessDataClass(QThread):
    signal_preprocess_data = pyqtSignal(str)

    def __init__(self, pool):
        super(PreprocessDataClass, self).__init__()
        self.train_dir = None
        self.gif_file_url = None
        self.pool = pool

    def accept_signal_start(self, gif_file_url, train_dir):
        self.gif_file_url = gif_file_url
        self.train_dir = train_dir

    def single_class_worker(self, args):
        (file_in_path, file_dir, out_path) = args
        file_path = os.path.join(file_in_path, file_dir)

        files = os.listdir(file_path)
        np.random.shuffle(files)
        os.chdir(file_path)

        # 整个一类的所有预处理后的数据，集合成一个npy文件（shape为 n * 4096）
        single_class_feature = np.empty((0, 32768))

        for file in files:
            if "wav" not in file:
                continue
            single_file_feature = np.empty((0, 32768))
            print(file_dir, " ", file)
            f, r = soundfile.read(os.path.join(file_path, file))
            print("INFO: shape f ", f.shape)  # (246021,)

            if f.shape[0] <= 22050*60*20:
                f_sliced = pre_slice(f, hop_len=32768 * constants.hop_len_for_ds_data)
            else:
                print(f"Too Long {f.shape}")
                f_sliced = pre_slice(f, hop_len=32768 * 4)

            print("INFO: shape f_sliced ", f_sliced.shape)  # (14, 32768)
            for f_s in f_sliced:
                # (1, 32768)

                f_pre = np.reshape(f_s, (1, 32768))
                f_pre = normalization(f_pre)

                single_file_feature = np.concatenate((single_file_feature, f_pre))
            # ************************************************

            np.random.shuffle(single_file_feature)
            single_file_predict_feature = single_file_feature[
                                          :int(constants.ds_train_data_predict_ratio * single_file_feature.shape[0])]
            predict_save_path = os.path.join(out_path, "predict", file_dir)

            if not os.path.exists(predict_save_path):
                os.mkdir(predict_save_path)
            predict_save_path = os.path.join(out_path, "predict", file_dir, f"predict_{file}.npy")

            single_file_predict_feature[np.isnan(single_file_predict_feature)] = 0
            np.save(predict_save_path, single_file_predict_feature)

            single_file_feature = single_file_feature[
                                  int(constants.ds_train_data_predict_ratio * single_file_feature.shape[0]):]
            print(f"train and predict shape {single_file_predict_feature.shape[0]} {single_file_feature.shape[0]}")
            single_class_feature = np.concatenate((single_class_feature, single_file_feature))
            print("single_file_feature.shape =", single_file_feature.shape)
            print("single_class_feature.shape =", single_class_feature.shape)
            print("================single file processing finished==================")
            self.signal_preprocess_data.emit(f"{file_dir} {file} done!")

        print("single_class_feature.shape =", single_class_feature.shape)
        print("***************single class proecssing finished******************")

        single_class_feature[np.isnan(single_class_feature)] = 0

        len_audio = single_class_feature.shape[0]
        np.random.shuffle(single_class_feature)

        sup_train = single_class_feature[:int(0.8 * len_audio)]
        sup_train_save_path = os.path.join(out_path, "32768", "sup_train", f"sup_train_{file_dir}.npy")
        np.save(sup_train_save_path, sup_train)

        # sup_valid = single_class_feature[int(0.8 * len_audio):int(0.9 * len_audio)]
        sup_valid = single_class_feature[int(0.8 * len_audio):]
        sup_valid_save_path = os.path.join(out_path, "32768", "sup_valid", f"sup_valid_{file_dir}.npy")
        np.save(sup_valid_save_path, sup_valid)

        self.signal_preprocess_data.emit(f"Class {file_dir} done!\n")

        # return f"Class {file_dir} done!"

    def run(self):
        self.signal_preprocess_data.emit(f"Subprocess start1")
        print("Subprocess start1")
        arg_list = [(self.gif_file_url, file_dir, self.train_dir) for file_dir in os.listdir(self.gif_file_url)]
        self.pool.map(self.single_class_worker, arg_list)

        self.signal_preprocess_data.emit("Inner info: Predict finished!")

