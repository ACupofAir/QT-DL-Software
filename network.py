# coding=utf-8

"""Network architecture."""
import tensorflow as tf

import constants


class DotProduct(tf.keras.layers.Layer):
    """Normalized dot product."""

    def call(self, anchor, positive):
        anchor = tf.nn.l2_normalize(anchor, axis=-1)
        positive = tf.nn.l2_normalize(positive, axis=-1)
        return tf.linalg.matmul(anchor, positive, transpose_b=True)


class BilinearProduct(tf.keras.layers.Layer):
    """Bilinear product."""

    def __init__(self, dim):
        super().__init__()
        self._dim = dim

    def build(self, _):
        self._w = self.add_weight(
            shape=(self._dim, self._dim),
            initializer="random_normal",
            trainable=True,
            name="bilinear_product_weight",
        )

    def call(self, anchor, positive):
        projection_positive = tf.linalg.matmul(self._w, positive, transpose_b=True)
        return tf.linalg.matmul(anchor, projection_positive)


class ContrastiveModel(tf.keras.Model):
    """Wrapper class for custom contrastive model."""

    def __init__(self, embedding_model, temperature, similarity_layer, similarity_type):
        super().__init__()
        self.embedding_model = embedding_model
        self._temperature = temperature
        self._similarity_layer = similarity_layer
        self._similarity_type = similarity_type

    def train_step(self, data):
        anchors, positives = data
        # data就是contrastive.train()中传入fit函数的dataset，在contrastive._prepare_example中已经转为anchor+positive的形式
        print("==network.ContrastiveModel.train_step==", "anchors.shape:", anchors.shape)
        print("==network.ContrastiveModel.train_step==", "positives.shape", positives.shape)

        with tf.GradientTape() as tape:
            inputs = tf.concat([anchors, positives], axis=0)
            embeddings = self.embedding_model(inputs, training=True)
            anchor_embeddings, positive_embeddings = tf.split(embeddings, 2, axis=0)

            # logits
            similarities = self._similarity_layer(anchor_embeddings, positive_embeddings)

            if self._similarity_type == constants.SimilarityMeasure.DOT:
                similarities /= self._temperature
            sparse_labels = tf.range(tf.shape(anchors)[0])

            # tf.print(sparse_labels)  # [0, 1, ..., batch_size - 1]
            # tf.print(tf.shape(similarities))  # [128, 128]
            # tf.print(tf.shape(sparse_labels))c  # [128]
            
            # 共batch_size个样本对，sparse_label的含义是“第i个样本对的label是i”
            # batch_size个anchor和batch_size个positive，计算similarity，得到[batch_size, batch_size]的similarity结果
            # 因此，similarity矩阵对角线上的值应该大，代表是正样本对，其它位置应当小，代表是负样本对
            # 因此，sparse_labels的值是range(0, batch_size)

            loss = self.compiled_loss(sparse_labels, similarities)  # compiled_loss(y, y_pred)
            loss += sum(self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(sparse_labels, similarities)
        return {m.name: m.result() for m in self.metrics}


def get_efficient_net_encoder(input_shape, pooling):
    """Wrapper function for efficient net B0."""
    efficient_net = tf.keras.applications.EfficientNetB0(
        include_top=False, weights=None, input_shape=input_shape, pooling=pooling)
    # imagenet None
    # To set the name `encoder` as it is used by supervised module for
    # to trainable value.
    return tf.keras.Model(efficient_net.inputs, efficient_net.outputs, name="encoder")


def get_contrastive_network(embedding_dim,
                            temperature,
                            pooling_type="max",
                            similarity_type=constants.SimilarityMeasure.DOT,
                            input_shape=(None, 64, 1)):
    """Creates a model for contrastive learning task."""
    inputs = tf.keras.layers.Input(input_shape)
    encoder = get_efficient_net_encoder(input_shape, pooling_type)
    x = encoder(inputs)  # encoder and h_x in the paper

    x = tf.keras.layers.Dense(embedding_dim, activation="linear")(x)  # projection head and z_x in the paper
    x = tf.keras.layers.LeakyReLU()(x)
    outputs = tf.keras.layers.Dense(embedding_dim, activation="linear")(x)

    # if similarity_type == constants.SimilarityMeasure.BILINEAR:
    #     outputs = tf.keras.layers.LayerNormalization()(outputs)
    #     outputs = tf.keras.layers.Activation("tanh")(outputs)   # !!!!!!!!!!!!!!!!! softmax should set from logit = True

    embedding_model = tf.keras.Model(inputs, outputs)  # model "②" in the paper
    print("embedding_model.summary()")
    embedding_model.summary()

    if similarity_type == constants.SimilarityMeasure.BILINEAR:
        embedding_dim = embedding_model.output.shape[-1]
        similarity_layer = BilinearProduct(embedding_dim)
    else:
        similarity_layer = DotProduct()

    return ContrastiveModel(embedding_model, temperature, similarity_layer, similarity_type)
