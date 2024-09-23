import librosa
import multiprocessing
import os
import numpy as np
import soundfile


def resample_worker(args):
    data_path, output_path, file_dir, file, resample_rate = args
    data, sample = librosa.load(path=data_path, sr=None)
    print(f"{data_path} {sample} {data.shape}")
    if len(data.shape) > 1:
        print(f"{file} has more than one channel")
        for i in range(data.shape[1]):
            slice_data = data[:, i]

            # slice_data = slice_data.astype(np.float32)
            resample_data = librosa.resample(slice_data, sample, resample_rate)
            if i == 0:
                channel_name = "left"
            else:
                channel_name = "right"

            if i > 1:
                raise "Channels is large than 2!!!"
            soundfile.write(os.path.join(output_path, file_dir, file.split(".")[0] + f"_{channel_name}.wav"), resample_data, resample_rate)
            print(f"{file} {i} done !")
    else:
        resample_data = librosa.resample(data, sample, resample_rate)
        soundfile.write(os.path.join(output_path, file_dir, file), resample_data, resample_rate)
        print(f"{file} done !")


if __name__ == "__main__":

    file_in_path = "H:/Datasets/ShipsEar_Original"
    file_out_path = "H:/Datasets/ShipsEar_22050"

    file_in_path = "D:/Datasets/shipsEar_52734_train"
    file_out_path = "D:/Datasets/shipsEar_22050_train"


    resample_rate = 22050

    if not os.path.exists(file_out_path):
        os.mkdir(file_out_path)

    pool = multiprocessing.Pool(6)

    for file_dir in os.listdir(file_in_path):
        arg_list = []
        if not os.path.exists(os.path.join(file_out_path, file_dir)):
            os.mkdir(os.path.join(file_out_path, file_dir))
        for file in os.listdir(os.path.join(file_in_path, file_dir)):
            arg_list.append((os.path.join(file_in_path, file_dir, file), file_out_path, file_dir, file, resample_rate))
        pool.map(resample_worker, arg_list)


