# from PyQt5 import QtCore, QtWidgets
# from PyQt5.QtCore import *
# from PyQt5.QtGui import *

from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QMovie, QPixmap
from PyQt5.QtWidgets import QLabel, QWidget, QLineEdit, QPushButton, QScrollArea, QTextEdit, QGroupBox, QHBoxLayout, \
    QVBoxLayout, QComboBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from PIL import Image
import os
import json
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['KaiTi']  # 保证正常显示中文
mpl.rcParams['font.serif'] = ['KaiTi']  # 保证正常显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 保证负号正常显示


class ColaSoftwareUi(QWidget):
    def __init__(self):
        super().__init__()

        self.height = None
        self.width = None
        self.background_movie = None
        self.ssl_model_path = None
        self.epochs = None
        self.chat_content_width = None
        self.config = None
        self.background_gif_url = None
        self.language = None
        self.batch_size = None
        self.experiment_id = None
        self.learning_rate = None
        self.label_video = QLabel(self)

        self.load_config()

        # self.set_background_gif()
        self.set_total_background_color()

        # 聊天框
        self.scroll_area_chat_content = QScrollArea()
        self.widget_chat_content = QWidget()

        self.scroll_area_chat_content.setWidget(self.widget_chat_content)
        self.chatsQTextEdit = QTextEdit(self.widget_chat_content)
        # 发送框

        # 语音框
        self.figure_loss = plt.figure(dpi=100, figsize=(10, 5), num=1)
        self.figure_loss.patch.set_alpha(0.5)
        self.canvas_loss = FigureCanvas(self.figure_loss)
        plt.subplots_adjust(top=0.98, bottom=0.08, left=0.06, right=0.99)

        self.widget_canvas = QScrollArea()
        self.widget_canvas.setWidget(self.canvas_loss)

        self.figure_accuracy = plt.figure(dpi=100, figsize=(10, 5), num=2)
        self.figure_accuracy.patch.set_alpha(0.5)
        plt.subplots_adjust(top=0.98, bottom=0.08, left=0.06, right=0.99)
        # plt.subplots_adjust(top=2, bottom=1, left=0, right=1, hspace=0, wspace=0)

        self.canvas_accuracy = FigureCanvas(self.figure_accuracy)

        self.widget_canvas_accuracy = QScrollArea()
        self.widget_canvas_accuracy.setWidget(self.canvas_accuracy)

        # 设置框
        # 窗口和布局
        self.setting_widget = QWidget()
        self.setting_layout = QVBoxLayout()
        self.setting_widget.setLayout(self.setting_layout)

        self.groupBoxSetting = QGroupBox()
        self.buttonPreprocessData = QPushButton()
        self.buttonStartTrain = QPushButton()
        self.buttonOnlineTest = QPushButton()

        self.buttonPredictTrainData = QPushButton()
        self.buttonPredictTestData = QPushButton()
        self.buttonPredictUnknownData = QPushButton()

        # 操作框布局
        self.horizontalLayoutGroupSetting1 = QHBoxLayout()
        self.horizontalLayoutGroupSetting1.addWidget(self.buttonPreprocessData)
        self.horizontalLayoutGroupSetting1.addWidget(self.buttonStartTrain)
        self.horizontalLayoutGroupSetting1.addWidget(self.buttonOnlineTest)

        self.horizontalLayoutGroupSetting2 = QHBoxLayout()
        self.horizontalLayoutGroupSetting2.addWidget(self.buttonPredictTrainData)
        self.horizontalLayoutGroupSetting2.addWidget(self.buttonPredictTestData)
        self.horizontalLayoutGroupSetting2.addWidget(self.buttonPredictUnknownData)

        self.verticalLayoutSetting = QVBoxLayout()
        self.verticalLayoutSetting.addLayout(self.horizontalLayoutGroupSetting1)
        self.verticalLayoutSetting.addLayout(self.horizontalLayoutGroupSetting2)

        self.groupBoxSetting.setLayout(self.verticalLayoutSetting)

        # 参数框
        self.groupBoxParameters = QGroupBox()

        # 第一行参数
        self.label_epoch = QLabel()

        self.max_parameter_height = 50
        self.label_epoch.setMaximumHeight(self.max_parameter_height)
        self.label_epoch.setFixedWidth(80)

        self.line_edit_epoch = QLineEdit()
        self.line_edit_epoch.setMaximumHeight(self.max_parameter_height)
        self.line_edit_epoch.setFixedWidth(40)

        self.label_batch_size = QLabel()
        self.label_batch_size.setFixedWidth(120)

        self.line_edit_batch_size = QLineEdit()
        self.line_edit_batch_size.setMaximumHeight(self.max_parameter_height)
        self.line_edit_batch_size.setFixedWidth(40)

        self.label_learning_rate = QLabel()
        self.label_learning_rate.setFixedWidth(40)

        self.line_edit_learning_rate = QLineEdit()
        self.line_edit_learning_rate.setMaximumHeight(self.max_parameter_height)
        self.line_edit_learning_rate.setFixedWidth(90)

        self.label_select_model = QLabel()
        self.label_select_model.setFixedWidth(80)

        self.combox_select_model = QComboBox()
        self.combox_select_model.setMaximumHeight(self.max_parameter_height)
        self.combox_select_model.setFixedWidth(170)
        self.model_types = ['EfficientNetB0', 'EfficientNetV2B0', 'ResNet50', 'ResNet50V2', 'ResNet101',
                            'ConvNeXtTiny', 'ConvNeXtSmall', 'ConvNeXtBase', 'ConvNeXtLarge',
                            'DenseNet121', 'DenseNet169', 'InceptionV3', 'InceptionResNetV2',
                            'RegNetX002', 'RegNetX004', 'VGG19', 'Xception']
        self.combox_select_model.addItems(self.model_types)

        self.horizontalLayoutGroupParameter = QHBoxLayout()
        self.horizontalLayoutGroupParameter.addWidget(self.label_epoch)
        self.horizontalLayoutGroupParameter.addWidget(self.line_edit_epoch)

        self.horizontalLayoutGroupParameter.addWidget(self.label_batch_size)
        self.horizontalLayoutGroupParameter.addWidget(self.line_edit_batch_size)

        self.horizontalLayoutGroupParameter.addWidget(self.label_learning_rate)
        self.horizontalLayoutGroupParameter.addWidget(self.line_edit_learning_rate)

        self.horizontalLayoutGroupParameter.addWidget(self.label_select_model)
        self.horizontalLayoutGroupParameter.addWidget(self.combox_select_model)

        # temp_qlabel = QLabel()
        # temp_qlabel.setStyleSheet("background-color:rgba(0, 0, 0, 0);")
        # self.horizontalLayoutGroupParameter.addWidget(temp_qlabel)
        # self.horizontalLayoutGroupParameter.addWidget(temp_qlabel)

        # self.horizontalLayoutGroupParameter.addWidget(self.label_ds_experiment_id)
        # self.horizontalLayoutGroupParameter.addWidget(self.line_edit_experiment_id)

        # 第二行参数
        self.button_ssl_model_path = QPushButton()
        self.label_ssl_model_path = QLabel()
        self.button_ssl_model_path.setMaximumHeight(self.max_parameter_height)
        self.button_ssl_model_path.setFixedWidth(140)

        self.label_ssl_model_path.setMaximumHeight(self.max_parameter_height)
        self.label_ssl_model_path.setFixedWidth(160)

        self.label_ds_experiment_id = QLabel()
        self.label_ds_experiment_id.setMaximumHeight(self.max_parameter_height)
        self.label_ds_experiment_id.setFixedWidth(200)

        self.line_edit_experiment_id = QLineEdit()
        self.line_edit_experiment_id.setMaximumHeight(self.max_parameter_height)


        #
        # self.label_ds_experiment_id = QLabel()
        # self.line_edit_experiment_id = QLineEdit()
        #
        self.horizontalLayoutGroupParameter2 = QHBoxLayout()
        self.horizontalLayoutGroupParameter2.addWidget(self.button_ssl_model_path)
        self.horizontalLayoutGroupParameter2.addWidget(self.label_ssl_model_path)

        self.horizontalLayoutGroupParameter2.addWidget(self.label_ds_experiment_id)
        self.horizontalLayoutGroupParameter2.addWidget(self.line_edit_experiment_id)
        # self.horizontalLayoutGroupParameter2.addWidget(temp_qlabel)

        # self.horizontalLayoutGroupParameter2.addWidget(QLabel())
        # self.horizontalLayoutGroupParameter2.addWidget(QLabel())
        # self.horizontalLayoutGroupParameter2.addWidget(QLabel())

        #
        # self.horizontalLayoutGroupParameter2.addWidget(self.label_ds_experiment_id)
        # self.horizontalLayoutGroupParameter2.addWidget(self.line_edit_experiment_id)

        self.verticalLayoutParameters = QVBoxLayout()
        self.verticalLayoutParameters.addLayout(self.horizontalLayoutGroupParameter)
        self.verticalLayoutParameters.addLayout(self.horizontalLayoutGroupParameter2)

        self.groupBoxParameters.setLayout(self.verticalLayoutParameters)
        self.groupBoxParameters.setMaximumHeight(120)
        # 整个设置框
        self.setting_layout.addWidget(self.groupBoxSetting)
        self.setting_layout.addWidget(self.groupBoxParameters)

        # 总体布局
        self.verticalLayout = QVBoxLayout(self)
        self.verticalLayout.addWidget(self.label_video)

        # 聊天区域
        self.scroll_area_chat_content.setParent(self.label_video)

        # 关闭
        self.button_close = QPushButton(self.label_video)

        # 音频
        self.widget_canvas.setParent(self.label_video)
        self.widget_canvas_accuracy.setParent(self.label_video)

        # 选项
        self.setting_widget.setParent(self.label_video)

        self.update_geometry()
        self.update_style_sheet()
        self.update_parameters()
        self.set_model_path(self.ssl_model_path)

        # QtCore.QMetaObject.connectSlotsByName(self)

    def load_config(self):
        with open(os.path.join(os.path.abspath("."), "config.json")) as f:
            data = f.read()
        self.config = json.loads(data)
        self.language = self.config["language"]

    def set_background_gif(self):
        self.background_gif_url = self.config["background_gif_url"]  # "images/1k.gif"
        if not os.path.exists(self.background_gif_url):
            print(f"Path {self.background_gif_url} not exists.")
            self.width, self.height = 1920, 1080
        else:
            print("url exists")
            img = Image.open(self.background_gif_url)
            self.width, self.height = img.size
            img.close()
        print(f"width {self.width} {self.height}")
        if '.gif' in self.background_gif_url:
            self.background_movie = QMovie(self.background_gif_url)
            self.background_movie.start()
            self.label_video.setMovie(self.background_movie)
        else:
            pix = QPixmap(self.background_gif_url)
            self.label_video.setPixmap(pix)

    def set_total_background_color(self):
        self.width = 1920
        self.height = 1080
        self.label_video.resize(self.width, self.height)
        # self.label_video.setStyleSheet("background-color: rgba(0, 0, 0, 0.8)")
        # self.label_video.setStyleSheet("background-color: rgba(0, 0, 128, 1)")
        # self.label_video.setStyleSheet("QWidget {background-color: qlineargradient(x1: 0, x2: 1, stop: 0 #373B44, stop: 1 #4286f4)}")
        # self.label_video.setStyleSheet("QWidget {background-color: qlineargradient(x1: 0, x2: 1, stop: 0 #373B44, stop: 1 #4286f4)}")
        # self.label_video.setStyleSheet("QWidget {background-color: qlineargradient(x1:0, x2:0.3, x3:0.6, x4: 1,stop: 0 #667db6, stop: 0.3 #0082c8,stop:0.6 #0082c8,stop: 1 #667db6)}")
        # self.label_video.setStyleSheet("QWidget {background-color: qlineargradient(x1: 0, x2: 1, stop: 0 #000046, stop: 1 #1CB5E0)}")
        self.label_video.setStyleSheet(
            "QWidget {background-color: qlineargradient(x1: 0, x2: 1, stop: 0 #2b5876, stop: 1 #4e4376)}")

    def update_parameters(self):
        self.learning_rate = self.config["learning_rate"]
        self.line_edit_learning_rate.setText(str(self.learning_rate))
        self.batch_size = self.config["batch_size"]
        self.line_edit_batch_size.setText(str(self.batch_size))
        self.epochs = self.config["epochs"]
        self.line_edit_epoch.setText(str(self.epochs))
        self.experiment_id = self.config["experiment_id"]
        self.line_edit_experiment_id.setText(self.experiment_id)
        self.ssl_model_path = self.config["ssl_model_path"]

    def update_geometry(self):
        self.resize(self.width, self.height)

        self.button_setting_height = 50
        self.buttonPreprocessData.setMinimumSize(100, self.button_setting_height)
        self.buttonStartTrain.setMinimumSize(100, self.button_setting_height)
        self.buttonOnlineTest.setMinimumSize(100, self.button_setting_height)
        self.buttonPredictTrainData.setMinimumSize(100, self.button_setting_height)
        self.buttonPredictTestData.setMinimumSize(100, self.button_setting_height)
        self.buttonPredictUnknownData.setMinimumSize(100, self.button_setting_height)

        self.label_epoch.setAlignment(Qt.AlignCenter)
        self.label_learning_rate.setAlignment(Qt.AlignCenter)
        self.label_ds_experiment_id.setAlignment(Qt.AlignCenter)
        self.label_batch_size.setAlignment(Qt.AlignCenter)
        self.label_select_model.setAlignment(Qt.AlignCenter)

        left_widget_width = 800

        self.setting_widget.setGeometry(0, 0, left_widget_width, 340)

        self.scroll_area_chat_content.setGeometry(
            QRect(0, 360, left_widget_width, self.height - 360))

        self.canvas_height = (self.height - 30) // 2
        self.canvas_width = self.width - left_widget_width - 30
        self.widget_canvas.setGeometry(left_widget_width + 30, 0, self.canvas_width, self.canvas_height)
        self.widget_canvas_accuracy.setGeometry(left_widget_width + 30, self.canvas_height + 30, self.canvas_width, self.canvas_height)
        self.canvas_loss.resize(self.canvas_width, self.canvas_height)
        self.canvas_accuracy.resize(self.canvas_width, self.canvas_height)

        self.button_close.setGeometry(self.width - 40, 0, 40, 40)
        self.button_close.hide()
        self.widget_chat_content.resize(self.scroll_area_chat_content.size())

        self.chatsQTextEdit.resize(self.chatsQTextEdit.parent().size())

    def change_language(self, language):
        if language == "English":

            self.language = "English"
            self.config["language"] = self.language

        elif language == "Chinese":
            self.groupBoxSetting.setTitle("选项")
            self.buttonPreprocessData.setText("预处理")
            self.buttonOnlineTest.setText("在线推理")
            self.buttonStartTrain.setText("训练模型")
            self.groupBoxParameters.setTitle("参数")

            self.button_close.setText("关闭")
            self.label_epoch.setText("Epoch")
            self.label_batch_size.setText("Batch size")
            self.label_learning_rate.setText("LR")
            self.label_ds_experiment_id.setText("Experiment id")
            self.label_select_model.setText("Model")
            self.buttonPredictTrainData.setText("测试训练数据")
            self.buttonPredictTestData.setText("测试测试集数据")
            self.buttonPredictUnknownData.setText("推理")
            self.button_ssl_model_path.setText("Model Path")
            self.language = "Chinese"
            self.config["language"] = self.language

    def training_set_button_disabled(self, disable_sign):
        self.buttonPreprocessData.setDisabled(disable_sign)
        self.buttonOnlineTest.setDisabled(disable_sign)
        self.buttonPredictTrainData.setDisabled(disable_sign)
        self.buttonPredictTestData.setDisabled(disable_sign)
        self.buttonPredictUnknownData.setDisabled(disable_sign)

    def predicting_set_button_disabled(self, disable_sign):
        self.buttonPreprocessData.setDisabled(disable_sign)
        self.buttonStartTrain.setDisabled(disable_sign)
        self.buttonOnlineTest.setDisabled(disable_sign)
        self.buttonPredictTrainData.setDisabled(disable_sign)
        self.buttonPredictTestData.setDisabled(disable_sign)
        self.buttonPredictUnknownData.setDisabled(disable_sign)

    def set_model_path(self, model_path):
        self.ssl_model_path = model_path
        self.config["ssl_model_path"] = model_path
        self.label_ssl_model_path.setText(model_path.split('/')[-1])

    def update_style_sheet(self):
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)

        self.scroll_area_chat_content.setStyleSheet(
            "background-color: rgba(255, 255,255,0.2); width: 2px; border:4px; font-size:18px;")
        self.widget_canvas.setStyleSheet("background-color: rgba(255, 255,255,0.0); width: 4px; border:0px")
        self.widget_canvas_accuracy.setStyleSheet("background-color: rgba(255, 255,255,0.0); width: 4px; border:0px")

        self.widget_chat_content.setStyleSheet("QScrollBar:vertical {border: 1px solid #999999; background:blue;"
                                               " width:5px; margin: 0px 0px 0px 0px; }"
                                               "QScrollBar::handle:vertical {"
                                               "background: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
                                               "stop: 0 rgb(32, 47, 130), stop: 0.5 rgb(32, 47, 130), stop:1 rgb(32, 47, 130));"
                                               "min-height: 0px;}"
                                               )
        self.setting_widget.setStyleSheet("background-color: rgba(255, 255,255,0.5); font-size:20px; ")

        self.button_close.setStyleSheet("color: white; background-color: rgba(255,255,255,0.5); border:0px;")

        # Label Style Sheet
        self.label_epoch.setStyleSheet(
            "background-color:rgba(0, 0, 128, 0.5); font-size:22px;color: white; border-radius:3px;")
        self.label_learning_rate.setStyleSheet(
            "background-color:rgba(0, 0, 128, 0.5); font-size:22px;color: white; border-radius:3px;")
        self.label_ds_experiment_id.setStyleSheet(
            "background-color:rgba(0, 0, 128, 0.5); font-size:22px;color: white; border-radius:3px;")
        self.label_batch_size.setStyleSheet(
            "background-color:rgba(0, 0, 128, 0.5); font-size:22px;color: white; border-radius:3px;")
        self.button_ssl_model_path.setStyleSheet(
            "background-color:rgba(0, 0, 128, 0.5); font-size:22px;color: white; border-radius:3px;")
        self.label_select_model.setStyleSheet(
            "background-color:rgba(0, 0, 128, 0.5); font-size:22px;color: white; border-radius:3px;")

        # Button Style Sheet
        self.buttonPreprocessData.setStyleSheet('background-color:rgba(128, 128, 128, 0.5); font-size:22px;')
        self.buttonStartTrain.setStyleSheet('background-color:rgba(220, 20, 60, 0.5); font-size:22px;')
        self.buttonOnlineTest.setStyleSheet('background-color:rgba(255, 222, 173, 0.5); font-size:22px;')
        self.buttonPredictTrainData.setStyleSheet('background-color:rgba(0, 255, 255, 0.5); font-size:22px;')
        self.buttonPredictTestData.setStyleSheet('background-color:rgba(152, 251, 152, 0.5); font-size:22px;')
        self.buttonPredictUnknownData.setStyleSheet('background-color:rgba(138, 43, 226, 0.5); font-size:22px;')

        # self.buttonPreprocessData.setStyleSheet("QWidget {background-color: qlineargradient(x1: 0, x2: 1, stop: 0 #373B44, stop: 1 #4286f4)}")
        # self.buttonStartTrain.setStyleSheet("QWidget {background-color: qlineargradient(x1: 0, x2: 1, stop: 0 #f953c6, stop: 1 #b91d73)}")
        # self.buttonOnlineTest.setStyleSheet("QWidget {background-color: qlineargradient(x1: 0, x2: 1, stop: 0 #a8c0ff, stop: 1 #3f2b96)}")
        # self.buttonPredictTrainData.setStyleSheet("QWidget {background-color: qlineargradient(x1: 0, x2: 1, stop: 0 #56CCF2, stop: 1 #2F80ED)}")
        # self.buttonPredictTestData.setStyleSheet("QWidget {background-color: qlineargradient(x1: 0, x2: 1, stop: 0 #B3FFAB, stop: 1 #12FFF7)}")
        # self.buttonPredictUnknownData.setStyleSheet("QWidget {background-color: qlineargradient(x1: 0, x2: 1, stop: 0 #7F00FF, stop: 1 #E100FF)}")
