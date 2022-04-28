
import tensorflow as tf
from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense, SeparableConv2D
import tensorflow_addons as tfa

import hyperparameters as hp


class YourModel(tf.keras.Model):

    def __init__(self):
        super(YourModel, self).__init__()

        self.optimizer = tfa.optimizers.AdamW(learning_rate = 0.001, weight_decay = 0.3)

        self.architecture = [SeparableConv2D(filters=8, kernel_size=(7, 7), padding='same',
                   strides=(1, 1), activation='gelu'),
                   MaxPool2D(pool_size=(2, 2)),
                   SeparableConv2D(filters=16, kernel_size=(7, 7), padding='same',
                   strides=(1, 1), activation='gelu'),
                   SeparableConv2D(filters=32, kernel_size=(7, 7), padding='same',
                   strides=(1, 1), activation='gelu'),
                   MaxPool2D(pool_size=(2, 2)),
                   SeparableConv2D(filters=64, kernel_size=(7, 7), padding='same',
                   strides=(1, 1), activation='gelu'),
                   MaxPool2D(pool_size=(2, 2)),
                   SeparableConv2D(filters=128, kernel_size=(7, 7), padding='same',
                   strides=(1, 1), activation='gelu'),
                   MaxPool2D(pool_size=(2, 2)),
                   Flatten(),
                   Dropout(0.1),
                   Dense(128, activation='gelu'),
                   Dense(64,activation = 'gelu'),
                   Dense(10, activation = 'softmax')
                   ]

    def call(self, x):
        """ Passes input image through the network. """

        for layer in self.architecture:
            x = layer(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """

        # TODO: Select a loss function for your network (see the documentation
        #       for tf.keras.losses)
        lossfn = tf.keras.losses.SparseCategoricalCrossentropy()
        return lossfn(labels, predictions)


