# Usage
# python track.py
# python track.py --video video/001.mp4
#
# Press 'q' To Quit


# Import Needed Packages
import numpy as np
import argparse
import time
import cv2


# Construct Argument Parser
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="Path To (Optional) Video File")
args = vars(ap.parse_args())

# Lower And Upper Color Ranges (BGR)
lower = np.array([100,25,0], dtype="uint8")
upper = np.array([255,128,50], dtype="uint8")

# If Video Not Specified, Use Onboard Video Capture Device
if not args.get("video", False):
	camera = cv2.VideoCapture(0)
else:
	camera = cv2.VideoCapture(args["video"])

# Iterate Through Video Feed
while True:
	(grabbed, frame) = camera.read()
	
	# If No Feed, Break
	if not grabbed:
		break
		
	# Define And Blur (Detection) Color	
	color = cv2.inRange(frame, lower, upper)
	color = cv2.GaussianBlur(color, (3,3), 0)
	
	# Find Contours Within (Color) Range
	(cnts, _) = cv2.findContours(color.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	# If Contours Found... Create Rectangle Around Object
	if len(cnts)>0:
		cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
		
		rect = np.int32(cv2.boxPoints(cv2.minAreaRect(cnt)))
		cv2.drawContours(frame, [rect], -1, (0,255,0), 2)
		
	cv2.imshow("Tracking", frame)
	cv2.imshow("Binary", color)
	
	time.sleep(0.025)
	
	if cv2.waitKey(1) &0xFF == ord("q"):
		break
		
camera.release()
cv2.destroyAllWindows()