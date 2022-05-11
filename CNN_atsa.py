import tensorflow as tf
from tensorflow import keras

class CNN_Gate_Aspect_Text(tf.keras.model):
    def __init__(self, feature_embeddings, aspect_embeddings):
        D = feature_embeddings.shape[1] ## embedding dimension
        C = 4 ## no. of output classes
        Co = 3 ## no. of kernels/filters

        self.feature_embedding_layer = tf.keras.layers.Embedding(len(feature_embeddings), D, 
        embeddings_initializer= keras.initializers.Constant(feature_embeddings), trainable = True)

        self.aspect_embedding_layer = tf.keras.layers.Embedding(len(aspect_embeddings), D, 
        embeddings_initializer= keras.initializers.Constant(aspect_embeddings), trainable = True)

        self.conv_layer_11 = tf.nn.conv1d(D, Co, 3)
        self.conv_layer_12 = tf.nn.conv1d(D, Co, 4)
        self.conv_layer_13 = tf.nn.conv1d(D, Co, 5)

        self.conv_layer_21 = tf.nn.conv1d(D, Co, 3)
        self.conv_layer_22 = tf.nn.conv1d(D, Co, 4)
        self.conv_layer_23 = tf.nn.conv1d(D, Co, 5)

        self.conv_layer_31 = tf.nn.conv1d(D, Co, 3, padding = 2)
        self.conv_layer_32 = tf.nn.conv1d(D, Co, 4, padding = 2)
        self.conv_layer_33 = tf.nn.conv1d(D, Co, 5, padding = 2)

        self.dropout = tf.nn.dropout(0.2)

        self.fully_connected = tf.keras.layers.Dense(C)
        self.fc_aspect = tf.keras.layers.Dense(Co)

    def forward(self, feature, aspect):

        feature = self.feature_embedding_layer(feature)
        aspect_v = self.aspect_embedding_layer(aspect)
        aspect_v = aspect_v.sum(1) / aspect_v.size(1)

        aa = [tf.nn.relu(self.conv_layer_31(aspect_v.transpose(1,2))),
              tf.nn.relu(self.conv_layer_32(aspect_v.transpose(1,2))),
              tf.nn.relu(self.conv_layer_33(aspect_v.transpose(1,2)))]

        x = [tf.nn.tanh(self.conv_layer_11(feature.transpose(1,2))),
            tf.nn.tanh(self.conv_layer_12(feature.transpose(1,2))),
            tf.nn.tanh(self.conv_layer_13(feature.transpose(1,2)))]

        y =  [tf.nn.relu(self.conv_layer_21(feature.transpose(1,2)) + self.fc_aspect(aspect_v)),
              tf.nn.relu(self.conv_layer_22(feature.transpose(1,2)) + self.fc_aspect(aspect_v)),
              tf.nn.relu(self.conv_layer_23(feature.transpose(1,2)) + self.fc_aspect(aspect_v))]
        
        x = [i*j for i, j in zip(x, y)]
        x = [tf.keras.layers.MaxPooling1D(i, i.size(2)).squeeze(2) for i in x]

        x = tf.cat(x, 1)
        x = self.dropout(x)  
        logit = self.fc1(x) 
        return logit, x, y