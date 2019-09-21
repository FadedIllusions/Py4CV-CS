# Import Needed Package
import cv2


class EyeTracker:
	def __init__(self, faceCascadePath, eyeCascadePath):
		self.faceCascade = cv2.CascadeClassifier(faceCascadePath)
		self.eyeCascade = cv2.CascadeClassifier(eyeCascadePath)
		
	def track(self, image):
		# Use Cascade To Detect Face
		faceRects = self.faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5,
													  minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)
		# Init Variable Within Which To Store Face And Eye Rectangle Coors
		rects=[]
		
		# Iterate Through Faces
		for(fX,fY,fW,fH) in faceRects:
			# Append Face Coors To Rects Variable
			faceROI=image[fY:fY+fH, fX:fX+fW]
			rects.append((fX,fY,fX+fW,fY+fH))
			
			# Use Cascade To Detect Eyes
			eyeRects = self.eyeCascade.detectMultiScale(faceROI, scaleFactor=1.1, minNeighbors=10,
														minSize=(20,20), flags=cv2.CASCADE_SCALE_IMAGE)
			
			# Iterate Through Eyes And Append Coords To Rects Variable
			for(eX,eY,eW,eH) in eyeRects:
				rects.append((fX+eX,fY+eY,fX+eX+eW,fY+eY+eH))
				
		return rects