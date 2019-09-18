# Usage
# python detect_faces.py --image images/001.jpg


# Import Needed Packages
from __future__ import print_function
from helpers.faceDetection import FaceDetector
import argparse
import cv2


# Construct Argument Parser And Parse Arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path To Image")
args = vars(ap.parse_args())


# Path To Face Cascade
cascade = "haarcascade_frontalface_default.xml"

# Load Image, Convert To Gray Scale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Init Face Detector Object
fd = FaceDetector(cascade)
faceRects = fd.detect(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
print("[INFO] Found {} Face(s).".format(len(faceRects)))

# Iterate Over Coords And Draw Bounding Box
for (x,y,w,h) in faceRects:
	cv2.rectangle(image, (x,y), (x+w,y+h), (0,0, 255), 2)
	
# Display Image With Bounding Box
cv2.imshow("Faces", image)
cv2.waitKey(0)