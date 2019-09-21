# Usage
# python eyetracking.py
# python eyetracking.py --video video_file.avi
#
# Press 'q' To Quit


# Import Needed Packages
from helpers.eyetracker import EyeTracker
import argparse
import imutils
import cv2


# Construct Argument Parser
ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', help="Path To (Optional) Video File")
args = vars(ap.parse_args())


# Define Cascade Paths
faceCascade = "haarcascade_frontalface_default.xml"
eyeCascade = "haarcascade_eye.xml"

# Init Eye Tracker Object
et = EyeTracker(faceCascade, eyeCascade)

# If Video Not Specified, Use Onboard Video Capture Device
if not args.get("video", False):
	camera = cv2.VideoCapture(0)
else:
	camera = cv2.VideoCapture(args["video"])
	
# Iterate Camera Frames
while True:
	(grabbed,frame) = camera.read()
	
	# If Unable To Read Frames From Video, Break Loop
	if args.get("video") and not grabbed:
		break
		
	# Resize Frame For Processing Speed, Convert To Grayscale	
	frame = imutils.resize(frame, width=400)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	# Assign Rects
	rects = et.track(gray)
	
	# Iterate Through Received Rects
	for rect in rects:
		# Draw Rectangle On Frame
		cv2.rectangle(frame, (rect[0],rect[1]), (rect[2],rect[3]), (0,0,255), 2)
		
	# Display Frame
	cv2.imshow("Tracking", frame)
	
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break
		
# Release Video Stream And Clean Up
camera.release()
cv2.destroyAllWindows()