import tensorflow as tf
import numpy as np
import math

class CNN_Gate_Aspect_Text(tf.keras.Model):
    def __init__(self, args):
        D = args.embed_dim
        C = args.class_num

        Co = args.kernel_num

        self.conv_layer_11 = tf.nn.conv1d(D, Co, 3)
        self.conv_layer_12 = tf.nn.conv1d(D, Co, 4)
        self.conv_layer_13 = tf.nn.conv1d(D, Co, 5)

        self.conv_layer_21 = tf.nn.conv1d(D, Co, 3)
        self.conv_layer_22 = tf.nn.conv1d(D, Co, 4)
        self.conv_layer_23 = tf.nn.conv1d(D, Co, 5)

        self.fully_connected = tf.nn.linear(C)
        self.fc_aspect = tf.nn.linear(Co)

    def forward(self, feature, aspect):
        
        aspect_v = aspect_v.sum(1) / aspect_v.size(1)

        x = tf.nn.tanh(self.conv_layer_11(feature))
        x = tf.nn.tanh(self.conv_layer_12(x))
        x = tf.nn.tanh(self.conv_layer_13(x))

        y =  tf.nn.relu(self.conv_layer_21(feature) + self.fc_aspect(aspect_v).unsqueeze(2))
        y =  tf.nn.relu(self.conv_layer_22(y) + self.fc_aspect(aspect_v).unsqueeze(2))
        y =  tf.nn.relu(self.conv_layer_23(y) + self.fc_aspect(aspect_v).unsqueeze(2))

        x = [i*j for i,j in zip(x,y)]

        x0 = [tf.keras.layers.MaxPooling1D(i, i.size(2)).squeeze(2) for i in x]

        logit = self.fully_connected(x0)
        return logit, x, y


