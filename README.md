# Video-Analogies

# Usage for GIFs/Videos
Extract frames from GIF/Video

$> time find FramesInput/ -maxdepth 1 -mindepth 1 -name "*.png" | parallel python image-analogies.py {}

the time command is just to find runtime
Replace FramesInput/ by directory name containing the frames.
The output frames will be stored in a directory called FramesOutput/

# Notes on GIFs/Videos
For converting sequence to GIFs, I found Image-Magick to be the most convenient

$> convert FramesOutput/* out.gif

For converting GIF to Image Sequence,

$> convert out.gif FramesInput/target.png

For Videos avconv/ffmpeg may be more convenient.


# FeatureVector.py Usage

To get all the feature vectors for a image pair

fvs = featureVector.getAllFeatureVectors(imageA, imageA1)

To get feature vector for a particular pixel in imageB and imageB1

fv = featureVector.getFeatureVectorForRowCol(imageB, imageB1,0,0) 

where 0,0 is the row,col.

imageB1 is initially assumed to be imageB since we dont have data for pixels at the first row.

