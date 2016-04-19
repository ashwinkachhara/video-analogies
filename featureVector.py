# Compute Feature Vector

import numpy as np

NONCAUSAL = 0
CAUSAL = 1
R = Y = 0
G = I = 1
B = Q = 2

def getAllFeatureVectors(unfiltered, filtered):
    fvUnfiltered = getFeatureVectors(unfiltered, NONCAUSAL)
    # print("unfilter:"+str(fvUnfiltered[0]))
    fvFiltered = getFeatureVectors(filtered, CAUSAL)
    # print("filtered:"+str(fvFiltered[0]))
    # join fvFiltered at the end of each row of fvUnfiltered
    completeFv = []
    for i in range(len(fvUnfiltered)):
        completeFv.append(fvUnfiltered[i] + fvFiltered[i])
    # print("complete:"+str(completeFv[0]))
    return completeFv


def getFeatureVectors(image, type=NONCAUSAL):
    rows = image.shape[0]
    cols = image.shape[1]
    channels = image.shape[2]

    yiqImage = np.zeros((rows, cols, channels), dtype=np.float32)
    yiqImage = image / 255.0

    yiqMatrix = np.array([[0.299, 0.587, 0.114],
                          [0.596, -0.274, -0.322],
                          [0.211, -0.523, 0.312]], np.float32)
    for i in range(rows):
        for j in range(cols):
            pixel = np.array(yiqImage[i][j])
            yiqImage[i][j] = np.dot(yiqMatrix, pixel)
    fvs = []
    if type == CAUSAL:
        # top 3 neighbor pixels and 1 left pixel
        for i in range(rows):
            for j in range(cols):
                fv = []
                for l in range(-1, 2):
                    pix = yiqImage[(i - 1) % rows][(j + l) % cols]
                    fv.append(pix[Y])
                    fv.append(pix[I])
                    fv.append(pix[Q])
                pix2 = yiqImage[i][(j - 1) % cols]
                fv.append(pix2[Y])
                fv.append(pix2[I])
                fv.append(pix2[Q])
                fvs.append(fv)
    else:
        # NON CAUSAL ( entire 3x3 neighbourhood)
        for i in range(rows):
            for j in range(cols):
                fv = []
                for k in range(-1, 2):
                    for l in range(-1, 2):
                        pix = yiqImage[(i + k) % rows][(j + l) % cols]
                        fv.append(pix[Y])
                        fv.append(pix[I])
                        fv.append(pix[Q])
                fvs.append(fv)
    return fvs
    # cv2.imshow('check', yiqImage)
    '''
    for i in range(unfiltered.shape[0]):
        for j in range(unfiltered.shape[1]):
            #print(str(unfiltered[i][j]))
            r = unfiltered[i][j][0]/255.0
            g = unfiltered[i][j][1]/255.0
            b = unfiltered[i][j][2]/255.0
            yiqUnFiltered[i][j] = colorsys.rgb_to_yiq(r, g, b)
            print("i,j:",i,",",j,str(yiqUnFiltered[i][j]))

    #for (x, y), value in np.ndenumerate(unfiltered):
    #    print(x, " ", y, " ", str(value))
    #    # yiqUnFiltered[x][y] = value
    '''
