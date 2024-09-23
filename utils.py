import numpy as np
# import librosa
from librosa.util import frame
import tensorflow as tf
import soundfile
import functools


def normalization(data):
    """
    对数据进行归一化
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def pre_slice(signal, frame_len=32768, hop_len=16384, axis=0):
    """
    对信号进行分帧
    :param signal: 原始信号
    :param frame_len: 帧长，4096
    :param hop_len: 帧移，2048
    :param axis:输出数据的方向，默认0：每一行为一帧，-1：每一列为一帧
    :return: 由各帧组成的矩阵
    """
    # return librosa.util.frame(signal, frame_length=frame_len, hop_length=hop_len).transpose()
    return frame(signal, frame_length=frame_len, hop_length=hop_len).transpose()


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

    num_spectrogram_bins = stfts.shape[-1]

    lower_edge_hertz, upper_edge_hertz, num_mel_bins = fmin, fmax, n_mels
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins,
                                                                        num_spectrogram_bins,
                                                                        sample_rate,
                                                                        lower_edge_hertz,
                                                                        upper_edge_hertz)
    # n_mels=64, fmin=60.0, fmax=7800.0 时，linear_to_mel_weight_matrix.shape = (129, 64)

    mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
    # spectrograms 和 linear_to_mel_weight_matrix 点乘后，mel_spectrograms.shape = (63, 64)

    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

    mel_spectrograms = tf.clip_by_value(
        mel_spectrograms,
        clip_value_min=1e-5,
        clip_value_max=1e8)
    log_mel_spectrograms = tf.math.log(mel_spectrograms)

    return log_mel_spectrograms


def extract_window(waveform, seg_length=16000):
    """
    Extracts a random segment from a waveform.
    如果seg_length < waveform长度，把waveform通过左右用0对称padding的方式，增长为seg_length长度
    如果seg_length > waveform长度，最后的random_crop从waveform中切seg_length的片段
    """
    padding = tf.maximum(seg_length - tf.shape(waveform)[0], 0)
    left_pad = padding // 2
    right_pad = padding - left_pad
    padded_waveform = tf.pad(waveform, paddings=[[left_pad, right_pad]])
    return tf.image.random_crop(padded_waveform, [seg_length])


def prepare_standard_example(example):
    """Creates an example for supervised training."""
    _print_flag = False
    x = example["audio"]

    # # same training data and testing data
    x = extract_window(x)
    x = tf.math.l2_normalize(x, epsilon=1e-9)

    x = extract_log_mel_spectrogram(x)
    x = x[Ellipsis, tf.newaxis]

    return x


def transport_wav_to_numpy(file_path):
    single_file_feature = np.empty((0, 32768))
    f, r = soundfile.read(file_path)
    hop_len = 2
    if f.shape[0] > 22050 * 60 * 20:
        hop_len = 8

    f_sliced = pre_slice(f, hop_len=32768 * hop_len)

    for f_s in f_sliced:
        f_pre = np.reshape(f_s, (1, 32768))
        f_pre = normalization(f_pre)
        single_file_feature = np.concatenate((single_file_feature, f_pre))

    return single_file_feature


def parse_example(audio):
    audio = tf.cast(audio, tf.float32) / float(tf.int16.max)
    return {"audio": audio}


def numpy_to_tensorflow_dataset(numpy_data):
    train_dataset = tf.data.Dataset.from_tensor_slices((numpy_data))

    train_dataset = train_dataset.map(parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    train_data = train_dataset.map(functools.partial(prepare_standard_example),
                                   num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .batch(32) \
        .prefetch(tf.data.experimental.AUTOTUNE)

    return train_data
