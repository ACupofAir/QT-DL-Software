# coding=utf-8
import enum

cola_data_dim = 32768
# hop_len_for_ssl_data = 1
hop_len_for_ds_data = 1

# generate_ssl_data_pool_size = 4
generate_ds_data_pool_size = 20

ds_train_data_predict_ratio = 0.2
ds_ckpt_save_freq_in_epoch = 100


@enum.unique
class SimilarityMeasure(enum.Enum):
    """Look up for similarity measure in contrastive model."""

    DOT = "dot_product"

    BILINEAR = "bilinear_product"

