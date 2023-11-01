import librosa
import numpy as np


def predict(test_audio, hmm_models):
    x, sr = librosa.load(test_audio, duration = 2.0)
    input_mfcc = librosa.feature.mfcc(y=x, sr=sr)
    input_mfcc = input_mfcc[:,:30]


    scores = []
    for item in hmm_models:
        hmm_model, label = item

        score = hmm_model.acc_score(input_mfcc)
        scores.append(score)
    # print("SCORES", np.array(scores))
    # print("MAX", np.array(scores).max() / 10**4)
    index = np.array(scores).argmax()   

    # return hmm_models[index][1]
 
    if np.array(scores).max() / 10**4 >  -0.34 :
        return hmm_models[index][1]
    else:
        return None


