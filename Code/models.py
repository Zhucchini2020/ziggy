
import tensorflow as tf
from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense, SeparableConv2D, BatchNormalization

import hyperparameters as hp


class YourModel(tf.keras.Model):

    def __init__(self):
        super(YourModel, self).__init__()

        self.optimizer = tf.optimizers.Adam()

        self.architecture = [Conv2D(filters=8, kernel_size=(3, 3), padding='same',
                   strides=(1, 1), activation='relu'),
                   BatchNormalization(),
                   MaxPool2D(pool_size=(2, 2)),
                   Conv2D(filters=16, kernel_size=(3, 3), padding='same',
                   strides=(1, 1), activation='relu'),
                   MaxPool2D(pool_size=(2, 2)),
                   BatchNormalization(),
                   Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                   strides=(1, 1), activation='relu'),
                   BatchNormalization(),
                   MaxPool2D(pool_size=(2, 2)),
                   Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                   strides=(1, 1), activation='relu'),
                   BatchNormalization(),
                   MaxPool2D(pool_size=(2, 2)),
                   Conv2D(filters=128, kernel_size=(3, 3), padding='same',
                   strides=(1, 1), activation='relu'),
                   BatchNormalization(),
                   MaxPool2D(pool_size=(2, 2)),
                   Flatten(),
                   Dropout(0.2),
                   Dense(128,activation='relu'),
                   Dense(64,activation='relu'),
                   Dense(10, activation = 'softmax')]
        

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

class VGGModel(tf.keras.Model):
    def __init__(self):
        super(VGGModel, self).__init__()

        # TODO: Select an optimizer for your network (see the documentation
        #       for tf.keras.optimizers)

        self.optimizer = tf.keras.optimizers.Adam()
        # Don't change the below:

        self.vgg16 = [
            # Block 1
            Conv2D(64, 3, 1, padding="same",
                   activation="relu", name="block1_conv1", ),
            Conv2D(64, 3, 1, padding="same",
                   activation="relu", name="block1_conv2"),
            MaxPool2D(2, name="block1_pool"),
            # Block 2
            Conv2D(128, 3, 1, padding="same",
                   activation="relu", name="block2_conv1"),
            Conv2D(128, 3, 1, padding="same",
                   activation="relu", name="block2_conv2"),
            MaxPool2D(2, name="block2_pool"),
            # Block 3
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv1"),
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv2"),
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv3"),
            MaxPool2D(2, name="block3_pool"),
            # Block 4
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv1"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv2"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv3"),
            MaxPool2D(2, name="block4_pool"),
            # Block 5
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv1"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv2"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv3"),
            MaxPool2D(2, name="block5_pool")
        ]

        # TODO: Make all layers in self.vgg16 non-trainable. This will freeze the
        #       pretrained VGG16 weights into place so that only the classificaiton
        #       head is trained.
        for layer in self.vgg16:
               layer.trainable = False
        # TODO: Write a classification head for our 15-scene classification task.

        self.head = [Flatten(),
        Dense(256, activation = 'relu'),
        Dense(128, activation = 'relu'),
        Dense(64, activation = 'relu'), 
        Dense(15, activation = 'softmax')]

        # Don't change the below:
        self.vgg16 = tf.keras.Sequential(self.vgg16, name="vgg_base")
        self.head = tf.keras.Sequential(self.head, name="vgg_head")

    def call(self, x):
        """ Passes the image through the network. """

        x = self.vgg16(x)
        x = self.head(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for model. """

        # TODO: Select a loss function for your network (see the documentation
        #       for tf.keras.losses)

        lossfn = tf.keras.losses.SparseCategoricalCrossentropy()
        return lossfn(labels, predictions)
