import tensorflow as tf
from tensorflow import keras


class CNN_Gate_Aspect_Text(tf.keras.Model):
    def __init__(self, feature_embeddings, aspect_embeddings, args):
        super(CNN_Gate_Aspect_Text, self).__init__()
        self.args = args
        emb_dim = args.embedding_dim  # embedding dimension
        num_classes = 3  # no. of output classes
        num_filters = self.args.num_filters  # no. of kernels/filters
        kern_sizes = self.args.kernel_sizes

        self.feature_embedding_layer = tf.keras.layers.Embedding(len(feature_embeddings), emb_dim,
                                                                 embeddings_initializer=keras.initializers.Constant(feature_embeddings), trainable=True)

        self.aspect_embedding_layer = tf.keras.layers.Embedding(len(
            aspect_embeddings), emb_dim, embeddings_initializer=keras.initializers.Constant(aspect_embeddings), trainable=True)

        self.conv1_layers = [tf.keras.layers.Conv1D(
            num_filters, kern_size,  input_shape=(None, emb_dim)) for kern_size in kern_sizes]

        self.conv2_layers = [tf.keras.layers.Conv1D(
            num_filters, kern_size, input_shape=(None, emb_dim)) for kern_size in kern_sizes]

        self.conv3_layer = tf.keras.layers.Conv1D(
            num_filters, 3,  input_shape=(None, emb_dim))

        self.fully_connected = tf.keras.layers.Dense(num_classes)
        self.fc_aspect = tf.keras.layers.Dense(num_filters)

    def forward(self, feature, aspect):

        feature = self.feature_embedding_layer(feature)
        print(f'sentence matrix size = {feature.shape}')
        aspect_v = self.aspect_embedding_layer(aspect)
        print(f'aspect_v initial shape = {aspect_v.shape}')

        aspect_v = tf.nn.relu(self.conv3_layer(aspect_v))
        print(f'aspect_v shape after conv = {aspect_v.shape}')
        # check axis == 1 \\ reduce_max does the max pooling
        aspect_v = tf.math.reduce_max(aspect_v, 1)
        print(f'aspect_v shape after max-over-time = {aspect_v.shape}')

        x = [tf.nn.tanh(conv_layer(feature))
             for conv_layer in self.conv1_layers]  # size = [batch_size x num_filters (output channels) x sentence_length (shortened by convolution)]
        print(f'sentence matrix size after conv = {x[0].shape}')

        y = [tf.nn.relu(conv_layer(feature) + self.fc_aspect(tf.expand_dims(aspect_v, 1)))
             for conv_layer in self.conv2_layers]

        x = [i*j for i, j in zip(x, y)]

        x = [tf.math.reduce_max(i, 1)
             for i in x]  # max_pooling using reduce_max

        x = tf.concat(x, 1)

        x = tf.nn.dropout(x, 0.5)  # dropout - is 0.5 too high?

        logits = self.fully_connected(x)
        probs = tf.nn.softmax(logits)
        return probs, x, y
