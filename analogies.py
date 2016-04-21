from ann import ANN
import featureVector
import numpy as np

class Analogies:
    def __init__(self, imageA, imageA1):
        self.A = imageA
        self.A1 = imageA1
        # [for px in A, get featureVector]
        fvs = featureVector.getAllFeatureVectors(self.A,self.A1)
        # get dim
        print(len(fvs),len(fvs[0]))
        dim = len(fvs[0])
        self.ann = ANN(dim)
        # add these feature vectors to ann
        self.ann.addVectors(fvs)
        print("populated the ANN")
        self.s = {}
        self.K = 2

    def XYToLinear(self, x, y, img):
        return x*img[1]+y
    def LinearToXY(self, p, img):
        return p/img[1], p%img[1]

    def getAnalogy(self, imageB):
        self.B = imageB
        self.B1 = np.zeros(self.B.shape)

        self.ashape = self.A.shape
        self.bshape = self.B.shape

        # self.A = self.A.flatten()
        # self.A1 = self.A1.flatten()
        # self.B = self.B.flatten()
        # self.B1 = self.B1.flatten()

        print("initialized")
        print(self.bshape)
        for i in range(self.bshape[0]):
            for j in range(self.bshape[1]):
                idx = self.XYToLinear(i,j,self.bshape)
                if idx%100000 == 0:
                    print("loop",idx)
                index = self.bestMatch(idx)
                x,y = self.LinearToXY(index,self.ashape)
                if x>=self.ashape[0] or y>=self.ashape[1]:
                    print "xy out of bounds",x,y,index
                a1y,a1i,a1q = featureVector.getPixelAsYIQ(self.A1,x,y)
                by,bi,bq = featureVector.getPixelAsYIQ(self.B,i,j)
                featureVector.setPixelFromYIQ([a1y,bi,bq],self.B1,i,j)
                # self.B1[i,j] = self.A1[x,y]

        # for idx,elem in enumerate(self.B):
        #     if idx%100000 == 0:
        #         print("loop",idx)


        # self.A = self.A.reshape(self.ashape)
        # self.A1 = self.A1.reshape(self.ashape)
        # self.B = self.B.reshape(self.bshape)
        # self.B1 = self.B1.reshape(self.bshape)

        return self.B1


    def bestMatch(self,q):
        p_app = self.bestApproximateMatch(q)
        return p_app
        (p_coh, d_coh) = self.bestCoherenceMatch(q)
        # # d_app
        # # d_coh
        # if d_coh <= d_app*(1+0.5*self.K):
        #     return p_coh
        # else:
        #     return p_app

    def bestApproximateMatch(self, q):
        # v = feature at q
        x,y = self.LinearToXY(q,self.bshape)
        if x>=self.bshape[0] or y>=self.bshape[1]:
            print "out of bounds",x,y,q
        v = featureVector.getFeatureVectorForRowCol(self.B.reshape(self.bshape),self.B1.reshape(self.bshape),x,y)
        return self.ann.query(v)

    def bestCoherenceMatch(self, q):
        # in nbd of q
        x,y = self.LinearToXY(q,self.bshape)
        if x>=self.bshape[0] or y>=self.bshape[1]:
            print "out of bounds",x,y,q
        fvq = featureVector.getFeatureVectorForRowCol(self.B.reshape(self.bshape),self.B1.reshape(self.bshape),x,y)
        minNeighbor = None
        minDiff = 13*99 # A random high value
        #top 3 neighbors
        for l in range(-1, 2):
            i = (x - 1) % self.bshape[0]
            j = (y + l) % self.bshape[1]
            fvij = featureVector.getFeatureVectorForRowCol(self.B.reshape(self.bshape),self.B1.reshape(self.bshape),i,j)
            diff = self.getDiff(fvq, fvij)
            if diff < minDiff:
                minDiff = diff
                minNeighbor = self.XYToLinear(i, j, self.B1)
        #left neighbor
        i = x;
        j = (y-1)%self.bshape[1]
        fvij = featureVector.getFeatureVectorForRowCol(self.B.reshape(self.bshape),self.B1.reshape(self.bshape),i,j)
        diff = self.getDiff(fvq, fvij)
        if diff < minDiff:
            minDiff = diff
            minNeighbor = self.XYToLinear(i, j, self.B1)
        return (minNeighbor, minDiff)



    def getDiff(fv1, fv2);
        diffVec = np.array(fv1) - np.array(fv2)
        diffVec = diffVec**2
        diff = sum(diffVec)
        return diff




