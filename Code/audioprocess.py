import tensorflow as tf
import numpy as np
import hyperparameters as hp
import librosa
import skimage
import sklearn.preprocessing
import matplotlib.pyplot as plt
import os

def spectrogrammify(path):
    split_path = path.split("\\")
    folder_path = ".\\Data\\generated_spectograms\\" + split_path[-2]
    save_path = folder_path + "\\" + split_path[-1] + "_"
    if not os.path.isdir(folder_path): os.makedirs(".\\Data\\generated_spectograms\\" + split_path[-2])

    audio, sampling_rate = librosa.load(path, sr=hp.sample_rate, mono=True)
    total_samples = len(audio)
    result = []
    for i in range(total_samples // (sampling_rate * hp.clip_length)):
        clip = audio[i*sampling_rate:(i+1)*sampling_rate]
        spec = librosa.feature.melspectrogram(y=clip, sr=sampling_rate, fmax=hp.max_frequency)
        spec = np.log(spec + 1e-8) # avoiding log0, necessary step because amplitude scales logarithmically
        img = sklearn.preprocessing.minmax_scale(spec, feature_range=(0,255))
        img = np.flip(img, axis=0).astype("uint8") # for data interpretation purposes, the spectrogram displays upside down by default

        skimage.io.imsave(save_path + str(i) + ".png", img)
        
        
        result.append(img)

    return np.array(result)

data_branch = ".\\Data\\genres_original"

for dir in os.listdir(data_branch):
    newdir = os.path.join(data_branch, dir)
    for filename in os.listdir(newdir):
        file = os.path.join(newdir, filename)
        print(file)
        a = spectrogrammify(file)

#consider removing hiphop38, it behaves weird
#python cant decode jazz54, removed from dataset