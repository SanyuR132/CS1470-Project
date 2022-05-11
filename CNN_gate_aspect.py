import tensorflow as tf
from tensorflow import keras


class CNN_Gate_Aspect_Text(tf.keras.Model):
    def __init__(self, feature_embeddings, aspect_embeddings, args):
        super(CNN_Gate_Aspect_Text, self).__init__()
        D = feature_embeddings.shape[1]  # embedding dimension
        C = 3  # no. of output classes
        Co = 100  # no. of kernels/filters

        self.feature_embedding_layer = tf.keras.layers.Embedding(len(feature_embeddings), D,
                                                                 embeddings_initializer=keras.initializers.Constant(feature_embeddings), trainable=True)

        self.aspect_embedding_layer = tf.keras.layers.Embedding(len(aspect_embeddings), D,
                                                                embeddings_initializer=keras.initializers.Constant(aspect_embeddings), trainable=True)

        self.conv_layer_11 = tf.keras.layers.Conv1D(
            Co, 3, input_shape=(None, D))
        self.conv_layer_12 = tf.keras.layers.Conv1D(
            Co, 4, input_shape=(None, D))
        self.conv_layer_13 = tf.keras.layers.Conv1D(
            Co, 5, input_shape=(None, D))

        self.conv_layer_21 = tf.keras.layers.Conv1D(
            Co, 3, input_shape=(None, D))
        self.conv_layer_22 = tf.keras.layers.Conv1D(
            Co, 4, input_shape=(None, D))
        self.conv_layer_23 = tf.keras.layers.Conv1D(
            Co, 5, input_shape=(None, D))

        self.fully_connected = tf.keras.layers.Dense(C)
        self.fc_aspect = tf.keras.layers.Dense(Co)

    def forward(self, feature, aspect):

        feature = self.feature_embedding_layer(feature)
        aspect_v = self.aspect_embedding_layer(aspect)
        aspect_v = tf.math.reduce_sum(
            aspect_v, 1) / tf.cast(tf.shape(aspect_v)[1], dtype=tf.float32)

        x = [tf.nn.tanh(self.conv_layer_11(feature)),
             tf.nn.tanh(self.conv_layer_12(feature)),
             tf.nn.tanh(self.conv_layer_13(feature))]

        y = [tf.nn.relu(self.conv_layer_21(feature) + self.fc_aspect(tf.expand_dims(aspect_v, 0))),
             tf.nn.relu(self.conv_layer_22(feature) +
                        self.fc_aspect(tf.expand_dims(aspect_v, 0))),
             tf.nn.relu(self.conv_layer_23(feature) + self.fc_aspect(tf.expand_dims(aspect_v, 0)))]

        x = [i * j for i, j in zip(x, y)]

        x0 = [tf.keras.layers.MaxPool1D(
            pool_size=int(tf.shape(i)[2]))(i) for i in x]

        x0 = [tf.reshape(i, (tf.shape(i)[0], -1)) for i in x]

        x0 = tf.concat(x0, 1)
        logits = self.fully_connected(x0)
        probs = tf.nn.softmax(logits)

        # print(tf.shape(logit))

        return probs, x, y
