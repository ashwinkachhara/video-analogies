import cv2
import numpy as np
import sys
import os
import traceback

#from ann import ANN
import featureVector
from analogies import Analogies

# try:
# print "Image Analogies"
bfilename = sys.argv[1]
print bfilename
imageA = cv2.imread("Input/a.jpg")
imageA1 = cv2.imread("Input/a1.jpg")
imageB = cv2.imread(bfilename)
# fvs = featureVector.getAllFeatureVectors(imageA, imageA1)
# print "FVS:", len(fvs), "FVS[0]:", len(fvs[0])#, "FVS[0][0]:",len(fvs[0][0])
# fv = featureVector.getFeatureVectorForRowCol(imageB, imageB,0,0)
# print "Feature Vector:",len(fv)
analogies = Analogies(imageA,imageA1)
analogies.quietMode()
analogies.annFromFile(13)
# analogies.annFromFVs()
imageB1 = analogies.getAnalogy(imageB)

cv2.imwrite("FramesOutput/"+bfilename.split("/")[1],imageB1)
