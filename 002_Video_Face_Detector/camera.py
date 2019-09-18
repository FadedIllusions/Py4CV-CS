# Usage
# python camera.py
# python camera.py --video example_video.mp4
#
# Defaults To Onboard Webcam Without Video File Specified
# Press 'q' To Quit


# Import Needed Packages
from helpers.faceDetection import FaceDetector
import argparse
import cv2


# Define Aspect-Aware Resizing
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
	dim = None
	(h,w) = image.shape[:2]
	
	if width is None and height is None:
		return image
	
	if width is None:
		r = height/float(h)
		dim = (int(w*r), height)
	else:
		r = width/float(w)
		dim = (width, int(h*r))
		
	resized = cv2.resize(image, dim, interpolation=inter)
	
	return resized


# Construct Argument Parser
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="Path To (Optional) Video File")
args = vars(ap.parse_args())


# Path To Face Cascade
cascade = "haarcascade_frontalface_default.xml"

# Init Face Detector Object
fd = FaceDetector(cascade)


# If Video Not Specified, Use Onboard Video Capture Device
if not args.get("video", False):
	camera = cv2.VideoCapture(0)
else:
	camera = cv2.VideoCapture(args["video"])
	

# Print Info To User
print("[INFO] Press 'q' To Quit...")
	
#  Iterate Over Video Frames
while True:
	(grabbed, frame) = camera.read()
	
	# If Cannot Grab Frame From Provided Video, Exit Iteration
	if args.get("video") and not grabbed:
		break
		
	frame = resize(frame, width=400)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	faceRects = fd.detect(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
	frameClone= frame.copy()
	
	for (fX,fY,fW,fH) in faceRects:
		cv2.rectangle(frameClone, (fX,fY), (fX+fW,fY+fH), (0,0,255), 2)
		
	cv2.imshow("Face", frameClone)
	
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break
		
camera.release()
cv2.destroyAllWindows()