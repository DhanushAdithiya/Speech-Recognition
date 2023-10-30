import os
import numpy as np
import librosa
from HiddenMarkovModelTrainer import HiddenMarkovModelTrainer
import pickle


def generate_model(input_folder):
    hmm_models = []
    for dirname in os.listdir(input_folder):
        subfolder = os.path.join(input_folder, dirname)
        if not os.path.isdir(subfolder):
            continue
        label = subfolder[subfolder.rfind("/") + 11:]
        X = np.array([])
        y_words = []
        for fname in [x for x in os.listdir(subfolder) if x.endswith('.mp3')][:-1]:  
            # print(filename)
            filepath = os.path.join(subfolder, fname)
            x, sr = librosa.load(filepath, sr=44100,duration=3.0) 
            # print("TIME", librosa.get_duration(y= x,sr=sr))
            mfcc_features = librosa.feature.mfcc(y=x, sr=sr)
            if len(X) == 0:
                X = mfcc_features[:, :30]
            else:
                X = np.append(X, mfcc_features[:, :30], axis=0)
            y_words.append(label)

            print('X.shape =', X.shape)

        hmm_trainer = HiddenMarkovModelTrainer()
        hmm_trainer.train(X)
        with open(f"../model/{label}.pkl", "wb")as f: pickle.dump(hmm_trainer, f)
        hmm_models.append((hmm_trainer, label))
        hmm_trainer = None
    return hmm_models

