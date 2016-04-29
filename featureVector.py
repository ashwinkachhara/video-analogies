# Compute Feature Vector

import numpy as np
import cv2

NONCAUSAL = 0
CAUSAL = 1
R = Y = 0
G = I = 1
B = Q = 2

yiqMatrix = np.array([[0.299, 0.587, 0.114],
                      [0.596, -0.274, -0.322],
                      [0.211, -0.523, 0.312]], np.float32)

rgbMatrix = np.array([[1, 0.956, 0.621],
                      [1, -0.272, -0.647],
                      [1, -1.106, 1.703]], np.float32)

'''
return yiq as a tuple (y,i,q)
'''
def getPixelAsYIQ(image, row, col):
    pixel = np.array(image[row][col], np.float32)
    pixel = pixel / 255.0
    yiqPixel = np.dot(yiqMatrix, pixel)
    return (yiqPixel[0], yiqPixel[1], yiqPixel[2])

'''
yiq is an array of form [y, i ,q]
return image with rgb value set for the given row, col
'''
def setPixelFromYIQ(yiq, image, row, col):
    yiqPix = np.array(yiq,np.float32)
    rgbPixel = np.dot(rgbMatrix, yiqPix)
    rgbPixel = rgbPixel * 255
    image[row][col] = rgbPixel
    return image

def getFeatureVectorForRowCol(unfiltered, filtered, row, col):
    rows = unfiltered.shape[0]
    cols = unfiltered.shape[1]
    channels = unfiltered.shape[2]

    unfiltlaplacian = cv2.cvtColor(unfiltered,cv2.COLOR_BGR2GRAY)
    unfiltlaplacian = cv2.Laplacian(unfiltlaplacian,cv2.CV_32F)
    filtlaplacian = cv2.cvtColor(filtered,cv2.COLOR_BGR2GRAY)
    filtlaplacian = cv2.Laplacian(filtlaplacian,cv2.CV_32F)

    fv = []
    # unfiltered image - non causal type ( entire 3x3 neighborhood)
    for k in range(-1, 2):
        for l in range(-1, 2):
            pixel = np.array(
                unfiltered[(row + k) % rows][(col + l) % cols], np.float32)
            pixel = pixel / 255.0
            yiqPixel = np.dot(yiqMatrix, pixel)
            fv.append(yiqPixel[Y])
            fv.append(unfiltlaplacian[(row + k) % rows][(col + l) % cols])
            # fv.append(yiqPixel[I])
            # fv.append(yiqPixel[Q])

    # filtered image - causal ( top 3 and left 1 neighbors only)
    # top 3 neighbors
    for l in range(-1, 2):
        pixel = np.array(
            filtered[(row - 1) % rows][(col + l) % cols], np.float32)
        pixel = pixel / 255.0
        yiqPixel = np.dot(yiqMatrix, pixel)
        fv.append(yiqPixel[Y])
        fv.append(filtlaplacian[(row - 1) % rows][(col + l) % cols])
        # fv.append(yiqPixel[I])
        # fv.append(yiqPixel[Q])
    # left neighbor
    pixel2 = np.array(
        filtered[row][(col - 1) % cols], np.float32)
    pixel2 = pixel2 / 255.0
    yiqPixel2 = np.dot(yiqMatrix, pixel2)
    fv.append(yiqPixel2[Y])
    fv.append(filtlaplacian[row][(col - 1) % cols])
    # fv.append(yiqPixel2[I])
    # fv.append(yiqPixel2[Q])
    # print("Fv:"+str(fv))
    return fv


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

    imlaplacian = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    imlaplacian = cv2.Laplacian(imlaplacian,cv2.CV_32F)

    yiqImage = np.zeros((rows, cols, channels), dtype=np.float32)
    yiqImage = image / 255.0
    '''
    yiqMatrix = np.array([[0.299, 0.587, 0.114],
                          [0.596, -0.274, -0.322],
                          [0.211, -0.523, 0.312]], np.float32)
    '''
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
                    fv.append(imlaplacian[(i - 1) % rows][(j + l) % cols])
                    # v.append(pix[I])
                    # fv.append(pix[Q])
                pix2 = yiqImage[i][(j - 1) % cols]
                fv.append(pix2[Y])
                fv.append(imlaplacian[i][(j - 1) % cols])
                # fv.append(pix2[I])
                # fv.append(pix2[Q])
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
                        fv.append(imlaplacian[(i + k) % rows][(j + l) % cols])
                        # fv.append(pix[I])
                        # fv.append(pix[Q])
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
