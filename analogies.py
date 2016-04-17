from ann import ANN
import featureVector
import numpy as np

class Analogies:
    def __init__(self, imageA, imageA1):
        self.A = imageA
        self.A1 = imageA1

        # get dim
        self.ann = ANN(dim)
        # [for px in A, get featureVector]
        # add these feature vectors to ann
        self.ann.addVectors()
        self.s = {}
        self.K = 2


    def getAnalogy(self, imageB):
        self.B = imageB
        self.B1 = np.array(self.B.shape)

    def bestMatch(self,q):
        p_app = bestApproximateMatch(q)
        p_coh = bestCoherenceMatch(q)
        # d_app
        # d_coh
        if d_coh <= d_app*(1+0.5*self.K):
            return p_coh
        else:
            return p_app

    def bestApproximateMatch(self,q):
        # v = feature at q
        return self.ann.query(v)

    def bestCoherenceMatch(self,q):
        # in nbd of q
        
