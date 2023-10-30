from hmmlearn import hmm
import numpy as np

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
