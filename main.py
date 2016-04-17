import cv2
import numpy as np
import sys
import os

try:
	print "Video Quilting"
except:
	tb = sys.exc_info()[2]
    tbinfo = traceback.format_tb(tb)[0]
    pymsg = "PYTHON ERRORS:\nTraceback Info:\n" + tbinfo + \
        "\nError Info:\n     " + \
            str(sys.exc_type) + ": " + str(sys.exc_value) + "\n"
    msgs = "ARCPY ERRORS:\n" + arcpy.GetMessages(2) + "\n"
    arcpy.AddError(msgs)
    arcpy.AddError(pymsg)
    print msgs
    print pymsg
    arcpy.AddMessage(arcpy.GetMessages(1))
    print arcpy.GetMessages(1)