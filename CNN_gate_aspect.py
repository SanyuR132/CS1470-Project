import tensorflow as tf
import numpy as np
from tensorflow import keras

class CNN_Gate_Aspect_Text(tf.keras.Model):
    def __init__(self, embeddings):
        D = embeddings.shape[1] ## embedding dimension
        C = 4 ## no. of output classes
        Co = 3 ## no. of kernels/filters

        self.embedding_layer = tf.keras.layers.Embedding(len(embeddings), D, 
        embeddings_initializer= keras.initializers.Constant(embeddings), trainable = True)

        self.conv_layer_11 = tf.nn.conv1d(D, Co, 3)
        self.conv_layer_12 = tf.nn.conv1d(D, Co, 4)
        self.conv_layer_13 = tf.nn.conv1d(D, Co, 5)

        self.conv_layer_21 = tf.nn.conv1d(D, Co, 3)
        self.conv_layer_22 = tf.nn.conv1d(D, Co, 4)
        self.conv_layer_23 = tf.nn.conv1d(D, Co, 5)

        self.fully_connected = tf.keras.layers.Dense(C)
        self.fc_aspect = tf.keras.layers.Dense(Co)

    def forward(self, feature, aspect):
        
        aspect_v = aspect_v.sum(1) / aspect_v.size(1)

        x = [tf.nn.tanh(self.conv_layer_11(feature.transpose(1,2))),
            tf.nn.tanh(self.conv_layer_12(feature.transpose(1,2))),
            tf.nn.tanh(self.conv_layer_13(feature.transpose(1,2)))]

        y =  [tf.nn.relu(self.conv_layer_21(feature.transpose(1,2)) + self.fc_aspect(aspect_v)),
              tf.nn.relu(self.conv_layer_22(feature.transpose(1,2)) + self.fc_aspect(aspect_v)),
              tf.nn.relu(self.conv_layer_23(feature.transpose(1,2)) + self.fc_aspect(aspect_v))]

        x = [i*j for i,j in zip(x,y)]

        x0 = [tf.keras.layers.MaxPooling1D(i, i.size(2)).squeeze(2) for i in x]

        logit = self.fully_connected(x0)
        return logit, x, y