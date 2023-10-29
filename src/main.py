from fileinput import filename
from re import A, I
import numpy as np
from scipy.io import wavfile
import librosa
import os
from hmmlearn import hmm
import pyaudio
import pickle
import wave
import sys


class HiddenMarkovModelTrainer(object):
    def __init__(self, model_name="Gaussian", n_components=6, cov_type="full", n_iter=10000):
        self.model_name = model_name
        self.n_components = n_components
        self.cov_type = cov_type
        self.n_iter = n_iter
        self.models = []

        if self.model_name == "Gaussian":
            self.model = hmm.GaussianHMM(
                
                n_components=self.n_components, covariance_type=self.cov_type, n_iter=self.n_iter)
        else:
            raise Exception("Invalid Model")

    
    def train(self, X):
        np.seterr(all="ignore")
        self.models.append(self.model.fit(X))

    def acc_score(self, input):
        return self.model.score(input)


CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1 if sys.platform == "darwin" else 2
RATE = 44100
RECORD_SECONDS = 3


# with wave.open("output.mp3", "wb") as wf:
#     p = pyaudio.PyAudio()
#     wf.setnchannels(CHANNELS)
#     wf.setsampwidth(p.get_sample_size(FORMAT))
#     wf.setframerate(RATE)


#     stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True)

#     print("RECORDING")
#     for _ in range(0, RATE // CHUNK * RECORD_SECONDS):
#         wf.writeframes(stream.read(CHUNK))
#     print("DONE")


#     stream.close()
#     p.terminate()
    

def generate_model(input_folder):
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


hmm_models = []
if not os.listdir("../model"):
    generate_model("../Training2")
else:
    for file in os.listdir("../model"):
        filename = file[:-4]
        model = pickle.load(open(f"../model/{file}", "rb"))
        hmm_models.append((model, filename))


test_audio = "../test/knock.mp3"
x, sr = librosa.load(test_audio, duration = 1.0)
input_mfcc = librosa.feature.mfcc(y=x, sr=sr)
input_mfcc = input_mfcc[:,:30]


scores = []
for item in hmm_models:
    hmm_model, label = item

    score = hmm_model.acc_score(input_mfcc)
    scores.append(score)
index = np.array(scores).argmax()

print("PRED:", hmm_models[index][1])

