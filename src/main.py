import numpy as np
from scipy.io import wavfile
import librosa
import os
from hmmlearn import hmm


class HiddenMarkovModelTrainer(object):
    def __init__(self, model_name="Gaussian", n_components=4, cov_type="diag", n_iter=1000):
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


labels = ["Aristotle", "Diogeners", "Epictetus",
          "Plato", "Pythagoras", "Sorcrates", "Thalse"]

input_folder = "training"

hmm_models = []

for dirname in os.listdir(input_folder):
    subfolder = os.path.join(input_folder, dirname)
    if not os.path.isdir(subfolder):
        continue
    label = subfolder[subfolder.rfind("/") + 1:]
    X = np.array([])
    y_words = []
    for filename in [x for x in os.listdir(subfolder) if x.endswith('.wav')]:
        filepath = os.path.join(subfolder, filename)
        x, sr = librosa.load(filepath, sr=44100)
        mfcc_features = librosa.feature.mfcc(y=x, sr=sr)
        if len(X) == 0:
            X = mfcc_features[:, :15]
        else:
            X = np.append(X, mfcc_features[:, :15], axis=0)
        y_words.append(label)

        print('X.shape =', X.shape)

    hmm_trainer = HiddenMarkovModelTrainer()
    hmm_trainer.train(X)
    hmm_models.append((hmm_trainer, label))


test_audio = "test/test3.wav"
x, sr = librosa.load(test_audio)
input_mfcc = librosa.feature.mfcc(y=x, sr=sr)
input_mfcc = input_mfcc[:,:15]

scores = []
for item in hmm_models:
    hmm_model, label = item

    score = hmm_model.acc_score(input_mfcc)
    scores.append(score)
index = np.array(scores).argmax()
print("TRUE:ARISTOTLE ")
print("PRED:", hmm_models[index][1])