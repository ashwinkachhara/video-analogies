import cv2
import numpy as np
import sys
import os
import traceback

#from ann import ANN
import featureVector
from analogies import Analogies

# try:
print "Video Quilting"
imageA = cv2.imread("Input/a.jpg")
imageA1 = cv2.imread("Input/a1.jpg")
imageB = cv2.imread("Input/b.jpg")
# fvs = featureVector.getAllFeatureVectors(imageA, imageA1)
# print "FVS:", len(fvs), "FVS[0]:", len(fvs[0])#, "FVS[0][0]:",len(fvs[0][0])
# fv = featureVector.getFeatureVectorForRowCol(imageB, imageB,0,0)
# print "Feature Vector:",len(fv)
analogies = Analogies(imageA,imageA1)
# analogies.annFromFile(13)
analogies.annFromFVs()
imageB1 = analogies.getAnalogy(imageB)

cv2.imwrite("Output/b1.jpg",imageB1)

print "Done"
# except:
#     tb = sys.exc_info()[2]
#     tbinfo = traceback.format_tb(tb)[0]
#     pymsg = "PYTHON ERRORS:\nTraceback Info:\n" + tbinfo + \
#         "\nError Info:\n     " + \
#             str(sys.exc_type) + ": " + str(sys.exc_value) + "\n"
#     print pymsg
