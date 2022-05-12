import tensorflow as tf
from tensorflow import keras


class CNN_Gate_Aspect_Text(tf.keras.Model):
    def __init__(self, feature_embeddings, aspect_embeddings, args):
        super(CNN_Gate_Aspect_Text, self).__init__()
        self.args = args
        emb_dim = self.args.embedding_dim  # embedding dimension
        num_classes = 3  # no. of output classes
        num_filters = self.args.num_filters  # no. of kernels/filters
        kern_sizes = self.args.kernel_sizes

        self.feature_embedding_layer = tf.keras.layers.Embedding(len(feature_embeddings), emb_dim,
                                                                 embeddings_initializer=keras.initializers.Constant(feature_embeddings), trainable=True)

        self.aspect_embedding_layer = tf.keras.layers.Embedding(len(
            aspect_embeddings), emb_dim, embeddings_initializer=keras.initializers.Constant(aspect_embeddings), trainable=True)

        # not meant to be sequential layers
        self.conv1_layers = [tf.keras.layers.Conv1D(
            num_filters, kern_size,  input_shape=(None, emb_dim)) for kern_size in kern_sizes]

        self.conv2_layers = [tf.keras.layers.Conv1D(
            num_filters, kern_size, input_shape=(None, emb_dim)) for kern_size in kern_sizes]

        self.fully_connected = tf.keras.layers.Dense(num_classes)
        self.fc_aspect = tf.keras.layers.Dense(num_filters)

    def forward(self, feature, aspect):

        feature = self.feature_embedding_layer(feature)
        aspect_v = self.aspect_embedding_layer(aspect)
        aspect_v = tf.math.reduce_sum(
            aspect_v, 0) / tf.cast(tf.shape(aspect_v)[1], dtype=tf.float32)

        x = [tf.nn.tanh(conv_layer(tf.transpose(feature, perm=[0, 2, 1])))
             for conv_layer in self.conv1_layers]

        y = [tf.nn.relu(conv_layer(feature) + self.fc_aspect(tf.expand_dims(aspect_v, 0)))
             for conv_layer in self.conv2_layers]

        x = [i * j for i, j in zip(x, y)]

        x = [tf.keras.layers.MaxPool1D(
            pool_size=int(tf.shape(i)[2]))(i) for i in x]

        x = [tf.reshape(i, (tf.shape(i)[0], -1)) for i in x]

        x = tf.concat(x, 1)
        logits = self.fully_connected(x0)
        probs = tf.nn.softmax(logits)

        # print(tf.shape(logit))

        return probs, x, y
