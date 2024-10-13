import json

# import numpy as np
# from PyQt5.QtGui import QTextDocumentFragment
from PyQt5.QtWidgets import *
from ui import ColaSoftwareUi
from PyQt5.QtCore import Qt
from PyQt5.Qt import QTextCursor
import matplotlib.pyplot as plt
from generate_sup_and_predict_data import *
from ds_train import *
from predict_train_validation_data import PredictTrainValidationData
from predict_single_file import PredictTestData
from predict_unknown_data import PredictUnknownData
from predict_online_data import PredictOnlineData
import os


class ColaSoftware(ColaSoftwareUi, QWidget):
    def __init__(self, pool):
        super().__init__()
        self.steps_per_epoch = None
        self.total_class_number = None
        self.pool = pool
        self.move_Flag = False
        self.change_language("Chinese")

        self.preprocess_data_obj = PreprocessDataClass(self.pool)
        self.preprocess_data_obj.signal_preprocess_data.connect(
            self.get_sub_process_print_info
        )

        self.train_dir = ""
        self.is_training = False
        self.canvas_loss.draw()

        self.epoch_metric = None
        self.plot_x_data = None

        self.predict_result_x_bar = None
        self.predict_result_right_number_bar = None
        self.predict_result_total_number_bar = None

        self.model = SupervisedModule()
        self.model.train_plot_callback.on_epoch_end_signal.connect(
            self.get_epoch_end_info
        )
        self.model.train_plot_callback.on_batch_end_signal.connect(
            self.get_batch_end_info
        )
        self.model.train_plot_callback.on_test_batch_end_signal.connect(
            self.get_test_batch_end_info
        )
        self.model.train_plot_callback.on_train_end_signal.connect(
            self.get_train_end_info
        )

        self.model.train_plot_callback.on_epoch_begin_signal.connect(
            self.get_epoch_begin_info
        )

        self.model.train_info_signal.connect(self.get_sub_process_print_info)

        self.predict_train_data_obj = PredictTrainValidationData()
        self.predict_train_data_obj.signal_predict_data_info.connect(
            self.get_sub_process_print_info
        )
        self.predict_train_data_obj.signal_predict_result.connect(
            self.get_predict_result
        )
        self.predict_train_data_obj.signal_predict_class_info.connect(
            self.get_predict_class_info
        )

        self.predict_test_data_obj = PredictTestData()
        self.predict_test_data_obj.signal_predict_data_info.connect(
            self.get_sub_process_print_info
        )
        self.predict_test_data_obj.signal_predict_result.connect(
            self.get_predict_result
        )
        self.predict_test_data_obj.signal_predict_class_info.connect(
            self.get_predict_class_info
        )

        self.predict_unknown_data_obj = PredictUnknownData()
        self.predict_unknown_data_obj.signal_predict_data_info.connect(
            self.get_sub_process_print_info
        )
        self.predict_unknown_data_obj.signal_predict_result.connect(
            self.get_predict_unknown_result
        )

        self.buttonPreprocessData.clicked.connect(self.preprocess_data)
        self.buttonStartTrain.clicked.connect(self.start_ds_train)
        self.button_close.clicked.connect(self.close)
        self.buttonPredictTrainData.clicked.connect(self.start_predict_train_data)
        self.buttonPredictTestData.clicked.connect(self.start_predict_test_data)
        self.buttonPredictUnknownData.clicked.connect(self.start_predict_unknown_data)
        self.buttonOnlineTest.clicked.connect(self.start_predict_online_data)
        self.button_ssl_model_path.clicked.connect(self.change_model_path)
        # self.combox_select_model.activated[str].connect(self.select_model)
        # self.chatsQTextEdit.append(self.combox_select_model.currentText())

        self.show_training_every_epochs = 2
        self.show_training_every_epochs_current_numbers = 0
        self.show_test_every_epochs_current_numbers = 0

        self.tcp_client = None

    # def select_model(self, model_type):
    #     self.chatsQTextEdit.append(model_type)

    def change_model_path(self):
        dialog_title = (
            "Select model file" if self.language == "English" else "选择模型文件"
        )
        model_path, _ = QFileDialog.getOpenFileName(self, dialog_title, directory=".")
        if model_path == "":
            return
        print(model_path)
        self.set_model_path(model_path)

    def get_train_end_info(self):
        self.is_training = False
        self.buttonStartTrain.setText("训练模型")
        self.training_set_button_disabled(False)
        self.line_edit_learning_rate.setText(str(self.learning_rate))
        self.line_edit_learning_rate.setDisabled(False)

    def start_predict_online_data(self):
        if not os.path.exists(self.ssl_model_path):
            self.chatsQTextEdit.append("Error: Please check model path.")
            return

        self.predict_online_data_obj = PredictOnlineData()
        self.predict_online_data_obj.signal_predict_data_info.connect(
            self.get_sub_process_print_info
        )
        self.predict_online_data_obj.signal_predict_result.connect(
            self.get_predict_unknown_result
        )
        self.predict_online_data_obj.signal_predict_result.connect(
            self.get_predict_unknown_result
        )
        self.predict_online_data_obj.accept_signal_model_path(self.ssl_model_path)

        self.predict_online_data_obj.start()
        self.predicting_set_button_disabled(True)

    def start_predict_unknown_data(self):
        if not os.path.exists(self.ssl_model_path):
            self.chatsQTextEdit.append("Error: Please check model path.")
            return

        dialog_title = (
            "Select test file" if self.language == "English" else "选择测试文件"
        )
        test_file, _ = QFileDialog.getOpenFileName(self, dialog_title, directory=".")
        if test_file == "":
            return

        self.predict_unknown_data_obj.accept_signal_data_path(test_file)
        self.predict_unknown_data_obj.accept_signal_model_path(self.ssl_model_path)
        self.predict_unknown_data_obj.start()
        self.predicting_set_button_disabled(True)

    def get_predict_class_info(self, class_number, total_file_number):
        self.predict_result_x_bar.append(class_number)
        self.predict_result_total_number_bar.append(total_file_number)
        self.predict_result_right_number_bar.append(0)

    def get_data_folder_path(self):
        dialog_title = (
            "Select test folder" if self.language == "English" else "选择验证文件夹"
        )
        data_folder_url = QFileDialog.getExistingDirectory(
            self, dialog_title, directory="."
        )
        print(f"***{data_folder_url}---")
        if data_folder_url == "":
            return None

        self.total_class_number = len(os.listdir(data_folder_url))

        return data_folder_url

    def start_predict_train_data(self):
        if not os.path.exists(self.ssl_model_path):
            self.chatsQTextEdit.append("Error: Please check model path.")
            return

        test_data_folder_url = self.get_data_folder_path()
        if test_data_folder_url is None:
            return

        self.predict_result_x_bar = []
        self.predict_result_right_number_bar = []
        self.predict_result_total_number_bar = []

        self.predict_train_data_obj.accept_signal_data_path(test_data_folder_url)
        self.predict_train_data_obj.accept_signal_model_path(self.ssl_model_path)
        self.predict_train_data_obj.start()
        plt.clf()
        self.predicting_set_button_disabled(True)

    def start_predict_test_data(self):
        if not os.path.exists(self.ssl_model_path):
            self.chatsQTextEdit.append("Error: Please check model path.")
            return

        test_data_folder_url = self.get_data_folder_path()
        if test_data_folder_url is None:
            return

        self.predict_result_x_bar = []
        self.predict_result_right_number_bar = []
        self.predict_result_total_number_bar = []

        self.predict_test_data_obj.accept_signal_data_path(test_data_folder_url)
        self.predict_test_data_obj.accept_signal_model_path(self.ssl_model_path)
        self.predict_test_data_obj.start()
        plt.clf()
        self.predicting_set_button_disabled(True)

    def start_ds_train(self):
        if self.is_training is False:
            if self.line_edit_experiment_id.text() == "":
                self.get_sub_process_print_info("Experiment id cannot be empty!")
                return

            dialog_title = (
                "Select train folder"
                if self.language == "English"
                else "选择训练文件夹"
            )
            train_folder_url = QFileDialog.getExistingDirectory(
                self, dialog_title, directory="."
            )

            if train_folder_url == "":
                return
            if "32768" not in os.listdir(train_folder_url):
                self.get_sub_process_print_info("请检查目录结构！ 32768/ssl_train/")
                return

            self.buttonStartTrain.setText("停止训练模型")
            self.is_training = True

            self.chatsQTextEdit.append(f"Train folder is {train_folder_url}")
            self.chatsQTextEdit.append(f"Start Training")

            self.model.set_train_dir(train_folder_url)
            self.model.set_batch_size(int(self.line_edit_batch_size.text()))
            self.model.set_epochs(int(self.line_edit_epoch.text()))
            self.learning_rate = float(self.line_edit_learning_rate.text())
            self.model.set_learning_rate(self.learning_rate)
            self.model.set_experiment_id(self.line_edit_experiment_id.text())
            self.model.set_model_type(self.combox_select_model.currentText())
            self.chatsQTextEdit.append(
                "Info use model: " + self.combox_select_model.currentText()
            )
            self.line_edit_learning_rate.setDisabled(True)

            self.model.start()
            plt.clf()
            self.training_set_button_disabled(True)
            self.epoch_metric = [[] for _ in range(5)]
            self.plot_x_data = []

        elif self.is_training is True:
            self.is_training = False
            self.model.train_plot_callback.accept_stop_training_sign(True)
            self.buttonStartTrain.setText("训练模型")

    def delete_last_line_of_textedit(self):
        test_cursor = self.chatsQTextEdit.textCursor()
        test_cursor.movePosition(
            QTextCursor.MoveOperation.End, QTextCursor.MoveMode.MoveAnchor
        )
        test_cursor.movePosition(
            QTextCursor.MoveOperation.Up, QTextCursor.MoveMode.KeepAnchor
        )
        test_cursor.movePosition(
            QTextCursor.MoveOperation.EndOfLine, QTextCursor.MoveMode.KeepAnchor
        )
        # test_cursor.selection().toPlainText()
        test_cursor.deleteChar()
        self.chatsQTextEdit.setTextCursor(test_cursor)

    def get_epoch_end_info(self, epoch, logs):
        self.delete_last_line_of_textedit()

        show_text = f"Epoch {epoch}, "
        keys = list(logs.keys())
        for i, key in enumerate(keys):
            if "lr" in key:
                self.line_edit_learning_rate.setText(str(logs[key]))
                continue
            show_text += (
                key.replace("sparse_categorical_", "") + " " + "%.4f" % logs[key] + ", "
            )
            if float(logs[key]) > 5:
                self.chatsQTextEdit.append(show_text)
                return

        keys = list(logs.keys())
        self.plot_x_data.append(len(self.plot_x_data))
        show_text = f"Epoch {epoch}, "
        for i, key in enumerate(keys):
            if "lr" in key:
                continue
            show_text += (
                key.replace("sparse_categorical_", "") + " " + "%.4f" % logs[key] + ", "
            )
            self.epoch_metric[i].append(logs[key])

        plt.clf()
        plt.rcParams.update(
            {
                # "figure.facecolor": (1, 0.0, 0.0, 0.3),  # red   with alpha = 30%
                "axes.facecolor": (1, 1, 1, 0.6),  # green with alpha = 50%
            }
        )

        max_losses = max(max(self.epoch_metric[0]), max(self.epoch_metric[2]))
        self.figure_loss = plt.figure(dpi=100, figsize=(8, 3.2), num=1)
        plt.xlim(xmin=0, xmax=len(self.epoch_metric[0]))
        plt.ylim(0, max_losses * 1.1)
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        # ax = plt.gca()
        # ax.xaxis.set_label_coords(1.05, -0.025)

        plt_alpha = 1
        plt.plot(self.plot_x_data, self.epoch_metric[0], label=keys[0], alpha=plt_alpha)
        plt.plot(self.plot_x_data, self.epoch_metric[2], label=keys[2], alpha=plt_alpha)
        plt.legend()
        self.canvas_loss.draw()

        # accuracy
        plt.clf()
        self.figure_accuracy = plt.figure(dpi=100, figsize=(8, 3.2), num=2)
        plt.ylim(0, 1.01)
        plt.xlim(xmin=0, xmax=len(self.epoch_metric[0]))
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")

        plt.plot(self.plot_x_data, self.epoch_metric[1], label=keys[1], alpha=plt_alpha)
        plt.plot(self.plot_x_data, self.epoch_metric[3], label=keys[3], alpha=plt_alpha)
        plt.legend()
        self.canvas_accuracy.draw()

        show_text = show_text[:-2]
        self.chatsQTextEdit.append(show_text)

    def get_epoch_begin_info(self):
        self.chatsQTextEdit.append("-")

    def get_batch_end_info(self):
        if (
            self.show_training_every_epochs_current_numbers
            % self.show_training_every_epochs
            == 1
        ):
            self.chatsQTextEdit.insertPlainText("-")
        self.show_training_every_epochs_current_numbers += 1
        # print(self.show_training_every_epochs_current_numbers, self.show_training_every_epochs)

    def get_test_batch_end_info(self):
        if (
            self.show_test_every_epochs_current_numbers
            % int(
                self.show_training_every_epochs * int(self.line_edit_batch_size.text())
            )
            == 1
        ):
            self.chatsQTextEdit.insertPlainText("-")
        self.show_test_every_epochs_current_numbers += 1

    def get_predict_result(self, right_number):
        self.predict_result_right_number_bar[-1] = right_number

        # Number
        plt.clf()
        plt.rcParams.update(
            {
                # "figure.facecolor": (1, 0.0, 0.0, 0.3),  # red   with alpha = 30%
                "axes.facecolor": (1, 1, 1, 0.6),  # green with alpha = 50%
            }
        )

        self.figure_loss = plt.figure(dpi=100, figsize=(8, 3.2), num=1)
        plt.grid(b=True, axis="y")
        plt.ylabel("数量/个")
        plt.xlabel("类别")
        if self.total_class_number < 6:
            plt.xlim(-1, 6)

        plt.bar(
            self.predict_result_x_bar,
            self.predict_result_right_number_bar,
            label="Right number",
            color="green",
            alpha=1,
        )
        plt.bar(
            self.predict_result_x_bar,
            self.predict_result_total_number_bar,
            label="Total number",
            color="red",
            alpha=0.2,
        )

        plt.legend()
        self.canvas_loss.draw()

        # Percent
        plt.clf()
        self.figure_accuracy = plt.figure(dpi=100, figsize=(8, 3.2), num=2)
        if self.total_class_number < 6:
            plt.xlim(-1, 6)

        plt.grid(b=True, axis="y")
        plt.ylim(0, 101)
        plt.ylabel("正确百分比/%")
        plt.xlabel("类别")
        plt.bar(
            self.predict_result_x_bar,
            np.array(self.predict_result_right_number_bar)
            / np.array(self.predict_result_total_number_bar)
            * 100,
            label="Right percent",
            alpha=0.9,
        )

        plt.legend()
        self.canvas_accuracy.draw()

    def get_predict_unknown_result(self, result):

        # Number
        plt.clf()
        self.figure_loss = plt.figure(dpi=100, figsize=(8, 3.2), num=1)
        plt.grid(b=True, axis="y")
        plt.rcParams.update(
            {
                # "figure.facecolor": (1, 0.0, 0.0, 0.3),  # red   with alpha = 30%
                "axes.facecolor": (1, 1, 1, 0.6),  # green with alpha = 50%
            }
        )

        if len(result) < 6:
            plt.xlim(-1, 6)
        plt.ylabel("切片数量/个")
        plt.xlabel("类别")

        plt.bar(list(range(len(result))), result, label="Class slice number")

        plt.legend()
        self.canvas_loss.draw()

        # Percent
        plt.clf()
        self.figure_accuracy = plt.figure(dpi=100, figsize=(8, 3.2), num=2)
        if len(result) < 6:
            plt.xlim(-1, 6)

        plt.grid(b=True, axis="y")
        plt.ylim(0, 101)
        plt.ylabel("所占百分比/%")
        plt.xlabel("类别")

        plt.bar(
            list(range(len(result))),
            np.array(result) / np.sum(np.array(result)) * 100,
            label="Class slice percent",
        )

        plt.legend()
        self.canvas_accuracy.draw()

    def preprocess_data(self):
        dialog_title = (
            "Select train folder" if self.language == "English" else "选择训练文件夹"
        )
        gif_file_url = QFileDialog.getExistingDirectory(
            self, dialog_title, directory="D:/Datasets"
        )

        if gif_file_url == "" or gif_file_url is None:
            return
        # gif_file_url = "D:/Datasets/7_3_test_22050_train_npy_hop_10"
        self.chatsQTextEdit.setText(gif_file_url)
        self.train_dir = os.path.join(
            gif_file_url + "_train_npy_hop_" + str(constants.hop_len_for_ds_data)
        )
        if not os.path.exists(self.train_dir):
            os.mkdir(self.train_dir)
            os.mkdir(os.path.join(self.train_dir, "32768"))
            os.mkdir(os.path.join(self.train_dir, "predict"))
            os.mkdir(os.path.join(self.train_dir, "32768", "sup_train"))
            os.mkdir(os.path.join(self.train_dir, "32768", "sup_valid"))

        self.preprocess_data_obj.accept_signal_start(gif_file_url, self.train_dir)
        self.preprocess_data_obj.start()
        self.predicting_set_button_disabled(True)

    def get_sub_process_print_info(self, info_text):
        if info_text == "Stop training!":
            self.is_training = False

        if "IMPORTANT" in info_text:
            self.steps_per_epoch = int(info_text.split(" ")[1].split("=")[1])
            self.show_training_every_epochs = (self.steps_per_epoch * 1.25) // 40
            if self.show_training_every_epochs <= 1:
                self.show_training_every_epochs = 2

            self.chatsQTextEdit.append(f"Info: steps per epoch {self.steps_per_epoch}")
        elif "Inner info:" in info_text:
            if "Predict finished" in info_text:
                self.predicting_set_button_disabled(False)
        else:
            self.chatsQTextEdit.append(info_text)

    def keyPressEvent(self, event):
        if event.isAutoRepeat():
            return

        # print(f"press {event.key()}")
        # if event.key() == 32 and self.chatMethod == 2 and self.generating_answer is False:
        #     print("start recording main **********")
        #     self.generating_answer = True
        #     self.get_generated_answer(0)
        #
        #     self.threadLoop.accept_input_text_ready(True)

    def keyReleaseEvent(self, event):
        if event.isAutoRepeat():
            return

        # print(f"release {event.key()}")

        # if event.key() == 16777220 and self.chatMethod == 3:
        #     self.send_text()
        #
        # if event.key() == 32 and self.chatMethod == 2:
        #     self.threadLoop.accept_input_text_ready(False)

    def mouseReleaseEvent(self, QMouseEvent):
        self.move_Flag = False

    def mousePressEvent(self, evt):
        if evt.button() == Qt.LeftButton:
            self.move_Flag = True
            self.mouse_x = evt.globalX()
            self.mouse_y = evt.globalY()
            self.origin_x = self.x()
            self.origin_y = self.y()

    def mouseMoveEvent(self, evt):
        if self.move_Flag:
            move_x = evt.globalX() - self.mouse_x
            move_y = evt.globalY() - self.mouse_y

            dest_x = self.origin_x + move_x
            dest_y = self.origin_y + move_y
            self.move(dest_x, dest_y)

    def closeEvent(self, event=None):
        file = open(os.path.join(os.path.abspath("."), "config.json"), "w")
        # self.config["test"] = "test"
        self.config["batch_size"] = int(self.line_edit_batch_size.text())
        self.config["learning_rate"] = float(self.line_edit_learning_rate.text())
        self.config["epochs"] = int(self.line_edit_epoch.text())
        # self.config["ssl_model_path"] = self.label_ssl_model_path.text()
        self.config["experiment_id"] = self.line_edit_experiment_id.text()

        json.dump(self.config, file, indent=1)
        file.close()
        print("Call close event end!")
