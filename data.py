# coding=utf-8

"""Provides helper data related functions."""
import os.path

import numpy as np
import tensorflow as tf

import constants


# data_in_path = "D:/COLA_ori/"
def get_self_supervised_data(data_dim, shuffle_buffer=1000, class_nums=5):
    """
    override
    自监督模型训练，加载ShipsEar数据
    """

    def _parse_example(audio, _):
        print("INFO audio shape ", audio.shape)  # (32768,)
        return {"audio": tf.cast(audio, tf.float32) / float(tf.int16.max)}

    data_in_path = constants.ssl_data_train_path

    print("Info load data from ", data_in_path)

    ssl_train_data_dir = os.path.join(data_in_path, "{}/ssl_train".format(data_dim))
    data = np.empty((0, data_dim))
    label = np.zeros(0)
    label_class_num = 0
    for class_num in range(class_nums):
        data_class = np.load(os.path.join(ssl_train_data_dir, "ssl_train_" + str(class_num + 1) + ".npy"))
        print("load data from ", os.path.join(ssl_train_data_dir, "ssl_train_" + str(class_num + 1) + ".npy"))
        data = np.concatenate((data, data_class))
        label_class = []
        for _ in range(data_class.shape[0]):
            label_class.append(label_class_num)
        label_class_num += 1
        label = np.concatenate((label, np.array(label_class)))

    dataset = tf.data.Dataset.from_tensor_slices((data, label))
    dataset = dataset.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
    dataset = dataset.map(_parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    print("INFO success label num ", label_class_num - 1)
    return dataset, len(label)


def get_downstream_dataset(data_dim=32768, shuffle_buffer=1000, class_nums=None, data_in_path=None):
    """
    override
    有监督下游任务训练，加载ShipsEar数据
    """
    if data_in_path is None:
        data_in_path = constants.ds_data_train_and_predict_path

    def _parse_example(audio, label):
        audio = tf.cast(audio, tf.float32) / float(tf.int16.max)
        return {"audio": audio, "label": label}

    print("Info load data from ", data_in_path)
    train_data_dir = os.path.join(data_in_path, "{}/sup_train".format(data_dim))

    if class_nums is None:
        class_nums = len(os.listdir(train_data_dir))

    train_data = np.empty((0, 32768))
    train_label = np.zeros(0)
    label_class_num = 0
    ratio = 1
    for class_num in range(class_nums):
        data_class = np.load(os.path.join(train_data_dir, "sup_train_" + str(class_num) + ".npy"))
        print("load data from ", os.path.join(train_data_dir, "sup_train_" + str(class_num) + ".npy"), '*'*20)

        len_used = int(ratio * data_class.shape[0])
        train_data = np.concatenate((train_data, data_class[:len_used]))
        label_class = []
        for _ in range(len_used):
            label_class.append(label_class_num)
        label_class_num += 1
        train_label = np.concatenate((train_label, np.array(label_class)))

    train_data_shape = train_data.shape[0]
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label))
    train_dataset = train_dataset.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
    train_dataset = train_dataset.map(_parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    #
    # test_data_dir = data_in_path + "{}/sup_valid".format(data_dim)
    test_data_dir = os.path.join(data_in_path, "{}/sup_valid".format(data_dim))

    test_data = np.empty((0, 32768))
    test_label = np.zeros(0)
    label_class_num = 0
    for class_num in range(class_nums):
        data_class = np.load(os.path.join(test_data_dir, "sup_valid_" + str(class_num) + ".npy"))
        print("load data from ", os.path.join(test_data_dir, "sup_valid_" + str(class_num) + ".npy"), '*'*20)

        len_used = int(ratio * data_class.shape[0])
        test_data = np.concatenate((test_data, data_class[:len_used]))
        label_class = []
        for _ in range(len_used):
            label_class.append(label_class_num)
        label_class_num += 1
        test_label = np.concatenate((test_label, np.array(label_class)))

    test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_label))
    test_dataset = test_dataset.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
    test_dataset = test_dataset.map(_parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return train_dataset, test_dataset, label_class_num, train_data_shape


def extract_log_mel_spectrogram(waveform, sample_rate=22050, frame_length=400, frame_step=160,
                                fft_length=1024, n_mels=64, fmin=60.0, fmax=7800.0):
    """
    提取log-mel谱图
    """

    stfts = tf.signal.stft(
        waveform,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length)
    spectrograms = tf.abs(stfts)
    print("INFO spectrograms shape ", spectrograms.shape)  # (98, 513)

    num_spectrogram_bins = stfts.shape[-1]
    print("INFO ==data.extract_log_mel_spectrogram==", "stfts.shape:", stfts.shape)  # (98, 513)
    # waveform:4096, frame_length=128, frame_step=64, fft_length=256 时，stfts.shape = (63, 129)
    # 20220510: waveform:2048, frame_length=64, frame_step=32, 后续shape保持不变
    # 20220516: waveform:16000, frame_length=400, frame_step=160, fft_length=1024

    lower_edge_hertz, upper_edge_hertz, num_mel_bins = fmin, fmax, n_mels

    # Returns a weight matrix that can be used to re-weight a Tensor
    # containing num_spectrogram_bins linearly sampled frequency information from [0, sample_rate / 2]
    # into num_mel_bins frequency information from [lower_edge_hertz, upper_edge_hertz] on the [mel scale][mel].

    # print("output shape of linear_to_mel_weight_matrix:", num_spectrogram_bins, num_mel_bins)
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins,
                                                                        num_spectrogram_bins,
                                                                        sample_rate,
                                                                        lower_edge_hertz,
                                                                        upper_edge_hertz)
    print("INFO liner to weight matrix shape ", linear_to_mel_weight_matrix.shape)  # (513, 64)
    print("==data.extract_log_mel_spectrogram==", "linear_to_mel_weight_matrix:", linear_to_mel_weight_matrix)
    # n_mels=64, fmin=60.0, fmax=7800.0 时，linear_to_mel_weight_matrix.shape = (129, 64)

    mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
    print("INFO mel spetrograms shape ", mel_spectrograms.shape)  # (98, 64)
    # spectrograms 和 linear_to_mel_weight_matrix 点乘后，mel_spectrograms.shape = (63, 64)

    # print(spectrograms.shape[:-1], linear_to_mel_weight_matrix.shape[-1:])  # (63,) (64,)
    # print(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))  # (63, 64)
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
    print("INFO mel new spetrograms shape ", mel_spectrograms.shape)  # (98, 64)

    mel_spectrograms = tf.clip_by_value(
        mel_spectrograms,
        clip_value_min=1e-5,
        clip_value_max=1e8)
    print("INFO mel spetrograms new new shape ", mel_spectrograms.shape)  # (98, 64)
    log_mel_spectrograms = tf.math.log(mel_spectrograms)
    print("INFO log mel spetrograms shape ", log_mel_spectrograms.shape)  # (98, 64)

    return log_mel_spectrograms


def extract_window(waveform, seg_length=16000):
    """
    Extracts a random segment from a waveform.
    如果seg_length < waveform长度，把waveform通过左右用0对称padding的方式，增长为seg_length长度
    如果seg_length > waveform长度，最后的random_crop从waveform中切seg_length的片段
    """
    print("success extract window")
    print(type(waveform))
    print(waveform.shape)
    padding = tf.maximum(seg_length - tf.shape(waveform)[0], 0)
    left_pad = padding // 2
    right_pad = padding - left_pad
    padded_waveform = tf.pad(waveform, paddings=[[left_pad, right_pad]])
    return tf.image.random_crop(padded_waveform, [seg_length])


def get_downstream_dataset_predict(train_data_dir):
    """
    override
    有监督下游任务训练，加载ShipsEar数据
    """

    def _parse_example(audio, label):
        audio = tf.cast(audio, tf.float32) / float(tf.int16.max)
        return {"audio": audio, "label": label}

    train_data = np.load(os.path.join(train_data_dir, "predict_data.npy"))
    train_label = np.zeros(train_data.shape[0])

    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label))
    # train_dataset = train_dataset.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
    train_dataset = train_dataset.map(_parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return train_dataset
