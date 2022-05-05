"""
Homework 5 - CNNs
CS1430 - Computer Vision
Brown University
"""

import os
import sys
import argparse
import re
from datetime import datetime
import tensorflow as tf

import hyperparameters as hp
from models import YourModel, VGGModel
from preprocess import Dataset
from skimage.transform import resize
from tensorboard_utils import \
        ImageLabelingLogger, ConfusionMatrixLogger, CustomModelSaver

from skimage.io import imread
from skimage.segmentation import mark_boundaries
from matplotlib import pyplot as plt
import numpy as np
import random
import shutil

from keras.applications.vgg16 import VGG16

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--load-vgg',
        default='vgg16_imagenet.h5',
        help='''Path to pre-trained VGG-16 file (only applicable to
        task 3).''')
    parser.add_argument(
        '--task',
        required=True,
        choices=['1', '3'],
        help='''Which task of the assignment to run -
        training from scratch (1), or fine tuning VGG-16 (3).''')
    parser.add_argument(
        '--data',
        default= os.path.join(os.path.normpath(os.getcwd() + os.sep + os.pardir), "Data\\generated_spectrograms\\"), 
        help='Location where the dataset is stored.')
    parser.add_argument(
        '--load-checkpoint',
        default=None,
        help='''Path to model checkpoint file (should end with the
        extension .h5). Checkpoints are automatically saved when you
        train your model. If you want to continue training from where
        you left off, this is how you would load your weights.''')
    parser.add_argument(
        '--confusion',
        action='store_true',
        help='''Log a confusion matrix at the end of each
        epoch (viewable in Tensorboard). This is turned off
        by default as it takes a little bit of time to complete.''')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='''Skips training and evaluates on the test set once.
        You can use this to test an already trained model by loading
        its checkpoint.''')
    parser.add_argument(
        '--testsplit',
        action='store_true',
        help='''Moves files from the training directory to the test directory''')

    return parser.parse_args()

def train(model, datasets, checkpoint_path, logs_path, init_epoch):
    """ Training routine. """

    # Keras callbacks for training
    callback_list = [
        tf.keras.callbacks.TensorBoard(
            log_dir=logs_path,
            update_freq='batch',
            profile_batch=0),
        ImageLabelingLogger(logs_path, datasets),
        CustomModelSaver(checkpoint_path, hp.max_num_weights)
    ]

    # Include confusion logger in callbacks if flag set
    if ARGS.confusion:
        callback_list.append(ConfusionMatrixLogger(logs_path, datasets))

    # Begin training
    model.fit(
        x=datasets.train_data,
        validation_data=datasets.test_data,
        epochs=hp.num_epochs,
        batch_size=None,
        callbacks=callback_list,
        initial_epoch=init_epoch,
    )


def test(model, test_data):
    """ Testing routine. """

    # Run model on test set
    model.evaluate(
        x=test_data,
        verbose=1,
    )


def main():
    """ Main function. """
    
    time_now = datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")
    init_epoch = 0
    if ARGS.testsplit:
        print("Splitting spectrograms from training into testing (150 per genre)")
        source = os.path.join(ARGS.data, "train")
        dest = os.path.join(ARGS.data, "test")
        for genre in ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]:
            genreSource = os.path.join(source, genre)
            genreDest = os.path.join(dest, genre)
            files = os.listdir(genreSource)
            for file_name in random.sample(files, 150):
                shutil.move(os.path.join(genreSource, file_name), genreDest)
        

    # If loading from a checkpoint, the loaded checkpoint's directory
    # will be used for future checkpoints
    if ARGS.load_checkpoint is not None:
        ARGS.load_checkpoint = os.path.abspath(ARGS.load_checkpoint)

        # Get timestamp and epoch from filename
        regex = r"(?:.+)(?:\.e)(\d+)(?:.+)(?:.h5)"
        init_epoch = int(re.match(regex, ARGS.load_checkpoint).group(1)) + 1
        timestamp = os.path.basename(os.path.dirname(ARGS.load_checkpoint))

    # If paths provided by program arguments are accurate, then this will
    # ensure they are used. If not, these directories/files will be
    # set relative to the directory of run.py
    if os.path.exists(ARGS.data):
        ARGS.data = os.path.abspath(ARGS.data)
    # Run script from location of run.py
    
    os.chdir(sys.path[0])

    datasets = Dataset(ARGS.data)

    if ARGS.task == '1':
        model = YourModel()
        model(tf.keras.Input(shape=(128,87,1)))
        checkpoint_path = "checkpoints" + os.sep + \
            "your_model" + os.sep + timestamp + os.sep
        logs_path = "logs" + os.sep + "your_model" + \
            os.sep + timestamp + os.sep

        # Print summary of model
        model.summary()
    else:
        model = VGGModel()

        checkpoint_path = "checkpoints" + os.sep + \
            "vgg_model" + os.sep + timestamp + os.sep
        logs_path = "logs" + os.sep + "vgg_model" + \
            os.sep + timestamp + os.sep
        model(tf.keras.Input(shape=(224,224,3)))

        # Print summaries for both parts of the model
        model.vgg16.summary()
        model.head.summary()

        # Load base of VGG model
        model.vgg16.load_weights(ARGS.load_vgg, by_name=True)
    
    # Load checkpoints
    if ARGS.load_checkpoint is not None:
        model.load_weights(ARGS.load_checkpoint, by_name=False)
        

    # Make checkpoint directory if needed
    if not ARGS.evaluate and not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # Compile model graph
    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["sparse_categorical_accuracy"])

    if ARGS.evaluate:
        test(model, datasets.test_data)
    else:
        train(model, datasets, checkpoint_path, logs_path, init_epoch)
    # Splice out different parts of input songs, generate type of song, mode it and test on actual songs
    # Maybe even a web interface?? (mebe)
    # model.load_weights() with a weights file
    # model.predict()
    # 


# Make arguments global
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
ARGS = parse_args()

main()
