from ann import ANN
import featureVector
import numpy as np

class Analogies:
    debug = True
    def __init__(self, imageA, imageA1):
        self.A = imageA
        self.A1 = imageA1

    def quietMode(self):
        self.debug = False

    def annFromFVs(self):
        # [for px in A, get featureVector]
        fvs = featureVector.getAllFeatureVectors(self.A,self.A1)
        # get dim
        print(len(fvs),len(fvs[0]))
        dim = len(fvs[0])
        self.ann = ANN(dim)
        # add these feature vectors to ann
        self.ann.addVectors(fvs)
        if self.debug:
            print("populated the ANN")
        self.ann.save()

    def annFromFile(self, fsize):
        self.ann = ANN(fsize)
        self.ann.load("analogies.ann")
        if self.debug:
            print("Loaded the ANN")

    def XYToLinear(self, x, y, imgshape):
        return x*imgshape[1]+y
    def LinearToXY(self, p, imgshape):
        return p/imgshape[1], p%imgshape[1]

    def getRandomImageFrom(self, outputShape, imageRef):
        image = np.zeros(outputShape)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                x = random.randint(0,imageRef.shape[0]-1)
                y = random.randint(0,imageRef.shape[1]-1)
                image[i][j] = imageRef[x][y]
                self.s[ self.XYToLinear(i,j,image.shape) ] = self.XYToLinear(x,y,imageRef.shape)
        # print("b1:",image)
        return image

    def getRandomPatchImage(self):
        alpha = 0.5
        image = np.array(self.B)
        for i in range(0,image.shape[0],10):
            for j in range(0,image.shape[1],10):
                x = random.randint(0, self.A1.shape[0]-11)
                y = random.randint(0, self.A1.shape[1]-11)
                for k in range(0,10):
                    for l in range(0,10):
                        if i+k < image.shape[0] and j+l < image.shape[1]:
                            image[i+k][j+l] = self.A1[x+k][y+l]
                            self.s[self.XYToLinear(i+k,j+l,image.shape)] = self.XYToLinear(x+k,y+l,self.A1.shape)
                        else:
                            pass
        image = cv2.addWeighted(image, alpha, self.B, 1-alpha, 0)
        print("Done intermediate")
        cv2.imwrite("Output/inter1.jpg",image)
        return image

    def getAnalogy(self, imageB):
        self.K = 0.5
        self.B = imageB
        self.B1 = self.getRandomImageFrom(self.B.shape, self.A1) # self.getRandomPatchImage() # np.zeros(self.B.shape)
        self.s = {}
        for i in range(self.B.size/3):
            if i >= self.A.size/3:
                self.s[i] = self.A.size/3-1
            else:
                self.s[i] = i

        self.ashape = self.A.shape
        self.bshape = self.B.shape

        # self.A = self.A.flatten()
        # self.A1 = self.A1.flatten()
        # self.B = self.B.flatten()
        # self.B1 = self.B1.flatten()

        numcoh = 0
        numapp = 0

        if self.debug:
            print("initialized")
            print self.A.size/3,self.B.size/3
        for i in range(self.bshape[0]):
            for j in range(self.bshape[1]):
                idx = self.XYToLinear(i,j,self.bshape)
                if idx%5000 == 0 and self.debug:
                    print("loop",idx)
                index, which = self.bestMatch(idx)
                self.s[idx] = index
                x,y = self.LinearToXY(index,self.ashape)
                self.s[idx] = index
                if which=="app":
                    numapp = numapp+1
                else:
                    numcoh = numcoh+1
                # print x,y
                # if x>=self.ashape[0] or y>=self.ashape[1]:
                #     print "xy out of bounds",x,y,index
                # if which == "app":
                a1y,a1i,a1q = featureVector.getPixelAsYIQ(self.A1,x,y)
                by,bi,bq = featureVector.getPixelAsYIQ(self.B,i,j)
                featureVector.setPixelFromYIQ([a1y,bi,bq],self.B1,i,j)
                # else:
                #     a1y,a1i,a1q = featureVector.getPixelAsYIQ(self.B,x,y)
                #     by,bi,bq = featureVector.getPixelAsYIQ(self.B,i,j)
                #     featureVector.setPixelFromYIQ([a1y,bi,bq],self.B1,i,j)
                # self.B1[i,j] = self.A1[x,y]

        # for idx,elem in enumerate(self.B):
        #     if idx%100000 == 0:
        #         print("loop",idx)


        # self.A = self.A.reshape(self.ashape)
        # self.A1 = self.A1.reshape(self.ashape)
        # self.B = self.B.reshape(self.bshape)
        # self.B1 = self.B1.reshape(self.bshape)
        if self.debug:
            print numcoh,numapp
        return self.B1


    def bestMatch(self,q):
        p_app, d_app = self.bestApproximateMatch(q)
        # return p_app, "app"
        p_coh, d_coh = self.bestCoherenceMatch(q)
        # # d_app
        # # d_coh
        if d_coh < d_app*(1+0.5*self.K):
            return p_coh, "coh"
        else:
            return p_app, "app"

    def bestApproximateMatch(self, q):
        # v = feature at q
        x,y = self.LinearToXY(q,self.bshape)
        if x>=self.bshape[0] or y>=self.bshape[1]:
            print "out of bounds",x,y,q
        v = featureVector.getFeatureVectorForRowCol(self.B.reshape(self.bshape),self.B1.reshape(self.bshape),x,y)
        p = self.ann.query(v)
        x,y = self.LinearToXY(p,self.ashape)
        v2 = featureVector.getFeatureVectorForRowCol(self.A.reshape(self.ashape),self.A1.reshape(self.ashape),x,y)
        return p, self.getDiff(v,v2)

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
            r = self.XYToLinear(i,j,self.bshape)
            p = self.s[r]
            i,j = self.LinearToXY(p, self.ashape)
            if i>=self.ashape[0] or j>=self.ashape[1]:
                print "neighbor out of bounds",i,j,p
            fvij = featureVector.getFeatureVectorForRowCol(self.A.reshape(self.ashape),self.A1.reshape(self.ashape),i,j)
            diff = self.getDiff(fvij, fvq)
            if diff < minDiff:
                minDiff = diff
                minNeighbor = self.XYToLinear(i, j, self.ashape)
        #left neighbor
        i = x;
        j = (y-1)%self.bshape[1]
        r = self.XYToLinear(i,j,self.bshape)
        p = self.s[r]
        i,j = self.LinearToXY(p, self.ashape)
        fvij = featureVector.getFeatureVectorForRowCol(self.A.reshape(self.ashape),self.A1.reshape(self.ashape),i,j)
        diff = self.getDiff(fvij, fvq)
        if diff < minDiff:
            minDiff = diff
            minNeighbor = self.XYToLinear(i, j, self.ashape)
        # print(minNeighbor)
        return minNeighbor, minDiff



    def getDiff(self, fv1, fv2):
        diffVec = np.array(fv1) - np.array(fv2)
        diffVec = diffVec**2
        diff = sum(diffVec)
        return diff
