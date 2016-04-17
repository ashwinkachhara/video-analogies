from ann import ANN
import featureVector

class Analogies:
    def __init__(self, imageA, imageA1):
        self.A = imageA
        self.A1 = imageA1

        # get dim
        self.ann = ANN(dim)
        # [for px in A, get featureVector]
        # add these feature vectors to ann
        self.ann.addVectors()


    def getAnalogy(self, imageB):
        self.B = imageB
