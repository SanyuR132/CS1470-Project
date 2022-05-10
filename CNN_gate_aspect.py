import tensorflow as tf
import numpy as np
import math

class CNN_Gate_Aspect_Text(nn.Module):
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

        x = tf.nn.tanh(self.conv_layer_11(feature))
        x = tf.nn.tanh(self.conv_layer_12(x))
        x = tf.nn.tanh(self.conv_layer_13(x))

        y =  tf.nn.relu(self.conv_layer_21(feature) + self.fc_aspect(aspect))
        y =  tf.nn.relu(self.conv_layer_22(y) + self.fc_aspect(aspect))
        y =  tf.nn.relu(self.conv_layer_23(y) + self.fc_aspect(aspect))