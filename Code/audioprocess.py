import tensorflow as tf
import numpy as np
import hyperparameters as hp
import librosa
import skimage
import sklearn.preprocessing
import matplotlib.pyplot as plt
import os

def spectrogrammify(path):
    split_path = path.split("/")
    folder_path = "./Data/generated_spectograms/" + split_path[-2]
    save_path = folder_path + "/" + split_path[-1] + "_"
    if not os.path.isdir(folder_path): os.makedirs("./Data/generated_spectograms/" + split_path[-2])

    audio, sampling_rate = librosa.load(path, sr=hp.sample_rate, mono=True)
    total_samples = len(audio)
    result = []
    for i in range(total_samples // (sampling_rate * hp.clip_length)):
        clip = audio[i*sampling_rate:(i+1)*sampling_rate]
        spec = librosa.feature.melspectrogram(y=clip, sr=sampling_rate, fmax=hp.max_frequency)
        spec = np.log(spec + 1e-8) # avoiding log0, necessary step because amplitude scales logarithmically
        img = sklearn.preprocessing.minmax_scale(spec, feature_range=(0,255))
        img = np.flip(img, axis=0).astype("uint8")

        skimage.io.imsave(save_path + str(i) + ".png", img)
        
        
        result.append(img)

    return np.array(result)


s = spectrogrammify("./Data/genres_original/hiphop/hiphop.00000.wav")

#load librosa song through audio, sampling_rate = librosa.load(path, mono=True)