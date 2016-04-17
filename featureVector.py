# Compute Feature Vector

import numpy as np
import cv2


def getFeatureVector(unfiltered, filtered):
    yiqUnFiltered = np.zeros((unfiltered.shape[0], unfiltered.shape[1],
                              unfiltered.shape[2]), dtype=np.float32)
    for i in range(unfiltered.shape[0]):
        for j in range(unfiltered.shape[1]):
            print(str(unfiltered[i][j]))
    #for (x, y), value in np.ndenumerate(unfiltered):
    #    print(x, " ", y, " ", str(value))
    #    # yiqUnFiltered[x][y] = value
