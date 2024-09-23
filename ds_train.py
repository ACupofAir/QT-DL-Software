# coding=utf-8

"""Supervised model for fine-tuning, random encoder and from scratch training."""
import functools
import tensorflow as tf
from tensorflow import keras
import data
import network
import numpy as np
import math
import os
import constants
from PyQt5.QtCore import QThread, pyqtSignal

os.environ['NUMEXPR_MAX_THREADS'] = '11'


class TrainPlotCallback(keras.callbacks.Callback, QThread):
    on_epoch_end_signal = pyqtSignal(int, dict)
    on_epoch_begin_signal = pyqtSignal()
    on_test_batch_end_signal = pyqtSignal()
    on_batch_end_signal = pyqtSignal()

    on_train_end_signal = pyqtSignal()

    def __init__(self):
        super(keras.callbacks.Callback, self).__init__()
        super(QThread, self).__init__()
        super(TrainPlotCallback, self).__init__()
        self.best_weights = None
        self.best = 0
        self.model_stop_training_sign = None

    def accept_stop_training_sign(self, stop_sign):
        self.model_stop_training_sign = stop_sign
        # if stop_sign is True:
        #     self.model.stop_training = True

    def on_train_batch_begin(self, batch, logs=None):
        # keys = list(logs.keys())
        # print("...Training: start of batch {}; got log keys: {}".format(batch, keys))
        self.on_batch_end_signal.emit()

    def on_test_batch_begin(self, batch, logs=None):
        self.on_test_batch_end_signal.emit()

    def on_epoch_begin(self, epoch, logs=None):
        self.on_epoch_begin_signal.emit()

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("val_sparse_categorical_accuracy")
        if np.less(self.best, current):
            self.best = current
            self.best_weights = self.model.get_weights()
        if self.model_stop_training_sign is True:
            self.model.stop_training = True
            self.model.set_weights(self.best_weights)
            self.best_weights = None
            self.best = 0
            self.model_stop_training_sign = False

        self.on_epoch_end_signal.emit(epoch, logs)

    def on_train_end(self, logs=None):
        # if self.stopped_epoch > 0:
        #     print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))
        self.on_train_end_signal.emit()


def prepare_standard_example(example):
    """Creates an example for supervised training."""
    x = example["audio"]

    x = data.extract_window(x)
    x = tf.math.l2_normalize(x, epsilon=1e-9)

    x = data.extract_log_mel_spectrogram(x)
    x = x[Ellipsis, tf.newaxis]

    y = example["label"]

    return x, y


class SupervisedModule(QThread):
    """Provides functionality for self-supervised source separation model."""
    train_info_signal = pyqtSignal(str)

    def __init__(self,
                 experiment_id='test',
                 data_dim=32768,
                 batch_size=64,
                 epochs=100,
                 learning_rate=0.001,
                 n_frames=100,
                 n_bands=64,
                 n_channels=1,
                 ssl_model_path=None,
                 contrastive_pooling_type='max',
                 contrastive_similarity_type=constants.SimilarityMeasure.DOT,
                 load_pretrained=False,
                 freeze_encoder=False,
                 contrastive_temperature=0.2,
                 contrastive_embedding_dim=512,
                 train_dir=None
                 ):
        """Initializes a supervised model object."""
        super(SupervisedModule, self).__init__()

        self.model_type = 'EfficientNetB0'
        self._model_path = os.path.join(os.path.abspath("."), "Model")
        self._experiment_id = experiment_id
        self._data_dim = data_dim
        self._batch_size = batch_size
        self._epochs = epochs
        self._learning_rate = learning_rate
        self._num_classes = None  # set by _prepare_downstream_task_data
        self._n_frames = n_frames
        self._n_bands = n_bands
        self._n_channels = n_channels
        self._shuffle_buffer = 1000
        self.steps_per_epoch = None

        self.train_plot_callback = TrainPlotCallback()

        self.contrastive_pooling_type = contrastive_pooling_type

        self.ssl_model_path = ssl_model_path
        self.contrastive_similarity_type = contrastive_similarity_type
        self.load_pretrained = load_pretrained
        self.freeze_encoder = freeze_encoder
        self.contrastive_temperature = contrastive_temperature
        self.contrastive_embedding_dim = contrastive_embedding_dim
        self.train_dir = train_dir

    def set_train_dir(self, train_dir):
        self.train_dir = train_dir

    def set_batch_size(self, batch_size):
        self._batch_size = batch_size

    def set_learning_rate(self, learning_rate):
        self._learning_rate = learning_rate

    def set_epochs(self, epochs):
        self._epochs = epochs

    def set_experiment_id(self, experiment_id):
        self._experiment_id = experiment_id

    def set_load_pretrained(self, load_pretrained):
        self.load_pretrained = load_pretrained

    def set_ssl_model_path(self, ssl_model_path):
        self.ssl_model_path = ssl_model_path

    def set_model_type(self, model_type):
        self.model_type = model_type

    def _prepare_downstream_task_data(self):
        """Get downstream task data."""
        train_data, test_data, self._num_classes, train_data_size = data.get_downstream_dataset(data_dim=self._data_dim,
                                                                                                shuffle_buffer=self._shuffle_buffer,
                                                                                                data_in_path=self.train_dir)
        self.steps_per_epoch = math.ceil(train_data_size / self._batch_size)
        self.train_info_signal.emit(f"IMPORTANT: steps_per_epoch={self.steps_per_epoch}")

        train_data = train_data.map(functools.partial(prepare_standard_example),
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE) \
            .batch(self._batch_size) \
            .prefetch(tf.data.experimental.AUTOTUNE)

        test_data = test_data.map(functools.partial(prepare_standard_example),
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE) \
            .batch(1) \
            .prefetch(tf.data.experimental.AUTOTUNE)

        return train_data, test_data

    def run(self):
        train_data, test_data = self._prepare_downstream_task_data()

        if self.load_pretrained is True:
            if self.ssl_model_path is None:
                raise ValueError("Self-supervised checkpoint id must not be None for loading pretrained model.")

            ssl_network = network.get_contrastive_network(
                embedding_dim=self.contrastive_embedding_dim,
                temperature=self.contrastive_temperature,
                similarity_type=self.contrastive_similarity_type,
                pooling_type=self.contrastive_pooling_type)

            ssl_network.compile(
                optimizer=tf.keras.optimizers.Adam(self._learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

            self.train_info_signal.emit(
                f"Info load pretrained model from {tf.train.latest_checkpoint(self.ssl_model_path)}")
            ssl_network.load_weights(tf.train.latest_checkpoint(self.ssl_model_path)).expect_partial()
            encoder = ssl_network.embedding_model.get_layer("encoder")
        else:
            '''
                efficient_net = tf.keras.applications.EfficientNetB0(
                include_top=False, weights=None, input_shape=input_shape, pooling=pooling)
        
                return tf.keras.Model(efficient_net.inputs, efficient_net.outputs, name="encoder")
            '''
            # encoder = network.get_efficient_net_encoder(
            #     input_shape=(None, self._n_bands, self._n_channels),
            #     pooling=self.contrastive_pooling_type)

            input_shape = (None, self._n_bands, self._n_channels)
            pooling = self.contrastive_pooling_type

            if self.model_type == 'EfficientNetB0':
                encoder = tf.keras.applications.EfficientNetB0(
                    include_top=False, weights=None, input_shape=input_shape, pooling=pooling)
            elif self.model_type == 'EfficientNetV2B0':
                encoder = tf.keras.applications.EfficientNetV2B0(
                    include_top=False, weights=None, input_shape=input_shape, pooling=pooling)
            elif self.model_type == 'ResNet50':
                encoder = tf.keras.applications.ResNet50(
                    include_top=False, weights=None, input_shape=input_shape, pooling=pooling)
            elif self.model_type == 'ResNet50V2':
                encoder = tf.keras.applications.ResNet50V2(
                    include_top=False, weights=None, input_shape=input_shape, pooling=pooling)
            elif self.model_type == 'ResNet101':
                encoder = tf.keras.applications.ResNet101(
                    include_top=False, weights=None, input_shape=input_shape, pooling=pooling)
            elif self.model_type == 'ConvNeXtTiny':
                encoder = tf.keras.applications.ConvNeXtTiny(
                    include_top=False, weights=None, input_shape=input_shape, pooling=pooling)
            elif self.model_type == 'ConvNeXtSmall':
                encoder = tf.keras.applications.ConvNeXtSmall(
                    include_top=False, weights=None, input_shape=input_shape, pooling=pooling)
            elif self.model_type == 'ConvNeXtBase':
                encoder = tf.keras.applications.ConvNeXtBase(
                    include_top=False, weights=None, input_shape=input_shape, pooling=pooling)
            elif self.model_type == 'ConvNeXtLarge':
                encoder = tf.keras.applications.ConvNeXtLarge(
                    include_top=False, weights=None, input_shape=input_shape, pooling=pooling)
            elif self.model_type == 'DenseNet121':
                encoder = tf.keras.applications.DenseNet121(
                    include_top=False, weights=None, input_shape=input_shape, pooling=pooling)
            elif self.model_type == 'DenseNet169':
                encoder = tf.keras.applications.DenseNet169(
                    include_top=False, weights=None, input_shape=input_shape, pooling=pooling)
            elif self.model_type == 'InceptionV3':
                encoder = tf.keras.applications.InceptionV3(
                    include_top=False, weights=None, input_shape=input_shape, pooling=pooling)
            elif self.model_type == 'InceptionResNetV2':
                encoder = tf.keras.applications.InceptionResNetV2(
                    include_top=False, weights=None, input_shape=input_shape, pooling=pooling)
            elif self.model_type == 'RegNetX002':
                encoder = tf.keras.applications.RegNetX002(
                    include_top=False, weights=None, input_shape=input_shape, pooling=pooling)
            elif self.model_type == 'RegNetX004':
                encoder = tf.keras.applications.RegNetX004(
                    include_top=False, weights=None, input_shape=input_shape, pooling=pooling)
            elif self.model_type == 'VGG19':
                encoder = tf.keras.applications.VGG19(
                    include_top=False, weights=None, input_shape=input_shape, pooling=pooling)
            elif self.model_type == 'Xception':
                encoder = tf.keras.applications.Xception(
                    include_top=False, weights=None, input_shape=input_shape, pooling=pooling)

        inputs = tf.keras.layers.Input(shape=(None, self._n_bands, self._n_channels))
        x = encoder(inputs)
        x = tf.keras.layers.ReLU()(x)

        self.train_info_signal.emit(f"Info class_num {self._num_classes}, activation {'None'}")

        outputs = tf.keras.layers.Dense(self._num_classes, activation=None, kernel_initializer="RandomNormal",
                                        bias_initializer="RandomNormal")(x)

        model = tf.keras.Model(inputs, outputs)

        # if freeze_encoder:
        #     print("ERROR ERROR ERROR freeze encoder")
        #     model.get_layer("encoder").trainable = False
        # else:
        #     print("SUCCESS SUCCESS SUCCESS not freeze encoder")

        model.compile(
            optimizer=tf.keras.optimizers.Adam(self._learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )

        model.summary()

        backup_path = os.path.join(self._model_path, self._experiment_id, "backup")
        # backend_restore_callback = tf.keras.callbacks.experimental.BackupAndRestore(backup_dir=backup_path,
        #                                                                             delete_checkpoint=False)
        backend_restore_callback = tf.keras.callbacks.experimental.BackupAndRestore(backup_dir=backup_path)

        self.train_info_signal.emit(f"Info backup path is {backup_path}")
        print(f"Info backup path is {backup_path}")

        log_dir = os.path.join(self._model_path, "log", self._experiment_id)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
        self.train_info_signal.emit(f"Info log_dir path is {log_dir}")
        print(f"Info log_dir path is {log_dir}")

        ckpt_path = os.path.join(self._model_path, self._experiment_id, "ckpt_{epoch}")
        self.train_info_signal.emit(f"Info ckpt_path path is {ckpt_path}")
        print(f"Info ckpt_path path is {ckpt_path}")

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path, save_weights_only=True,
                                                                       verbose=1,
                                                                       save_freq=self.steps_per_epoch * constants.ds_ckpt_save_freq_in_epoch)

        # lr_reduce_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_sparse_categorical_accuracy',
        #                                                           factor=0.9,
        #                                                           min_lr=self._learning_rate / 20,
        #                                                           verbose=1)

        lr_reduce_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                                  factor=0.9,
                                                                  min_lr=self._learning_rate / 20,
                                                                  verbose=1)

        import time
        start_time = time.time()
        model.fit(
            train_data,
            validation_data=test_data,
            epochs=self._epochs,
            verbose=1,
            callbacks=[
                lr_reduce_callback,
                model_checkpoint_callback,
                backend_restore_callback,
                tensorboard_callback,
                self.train_plot_callback
            ])
        self.train_info_signal.emit("Stop training!")

        end_time = time.time()
        seconds = int(end_time - start_time)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        days, hours = divmod(hours, 24)

        info_time = "INFO: use  %d days %02d:%02d:%02d" % (days, hours, minutes, seconds)

        self.train_info_signal.emit(info_time)
        print(info_time)

        model.save(os.path.join(self._model_path, self._experiment_id, self._experiment_id + '.h5'))
        self.train_info_signal.emit(
            f"INFO: save model to {os.path.join(self._model_path, self._experiment_id, self._experiment_id + '.h5')}")
