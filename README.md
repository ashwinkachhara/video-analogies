# Video-Quilting

# FeatureVector.py Usage

To get all the feature vectors for a image pair

fvs = featureVector.getAllFeatureVectors(imageA, imageA1)

To get feature vector for a particular pixel in imageB and imageB1

fv = featureVector.getFeatureVectorForRowCol(imageB, imageB1,0,0) 

where 0,0 is the row,col.

imageB1 is initially assumed to be imageB since we dont have data for pixels at the first row.

