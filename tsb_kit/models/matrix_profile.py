#import matrixprofile as mp
import stumpy
import numpy as np

class MatrixProfile():
    def __init__(self, window = 100):
        self.window = window
        self.model_name = 'MatrixProfile'

    def fit(self, X, y=None):
        """Fit detector. y is ignored in unsupervised methods.
        Parameters
        ----------
        X : numpy array of shape (n_samples, )
            The input samples.
        y : Ignored
            Not used, present for API consistency by convention.
        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.profile = stumpy.stump(X,m=self.window)
        #self.profile = mp.compute(X, windows=self.window)
        self.decision_scores_ = self.profile[:,0]#['mp']
        return self
    
    #def top_k_discords(self, k=5):
    #    discords = mp.discover.discords(self.profile, exclusion_zone=self.window//2, k=k)
    #    return discords['discords']
