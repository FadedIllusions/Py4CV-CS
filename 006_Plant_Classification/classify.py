# Usage
# python classify.py --images dataset/images --masks dataset/masks


# Import Needed Packages
from __future__ import print_function
from helpers import RGBHistogram
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np
import argparse
import glob
import cv2


# Construct Argument Parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="Path To Image Dataset")
ap.add_argument("-m", "--masks", required=True, help="Path To Image Masks")
args = vars(ap.parse_args())


# Define Paths
imagePaths = sorted(glob.glob(args["images"]+"/*.png"))
maskPaths = sorted(glob.glob(args["masks"]+"/*.png"))

# Init Data And Target Containers
data = []
target = []

# Init Descriptor / Histogram Object
desc = RGBHistogram([8,8,8])


# Iterate Over Images And Masks
for (imagePath,maskPath) in zip(imagePaths,maskPaths):
	# Load Images And Mask, Convert Masks To Grayscale
	image = cv2.imread(imagePath)
	mask = cv2.imread(maskPath)
	mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
	
	# Describe Features
	features = desc.describe(image,mask)
	
	# Update Lists
	data.append(features)
	target.append(imagePath.split("_")[-2])
	
# Filter Target Names And Encode
targetNames = np.unique(target)
le = LabelEncoder()
target = le.fit_transform(target)

# Create Train/Test Split
(trainData, testData, trainTarget, testTarget) = train_test_split(data,target,test_size=0.3,random_state=42)

# Train Classifier
model = RandomForestClassifier(n_estimators=25, random_state=84)
model.fit(trainData,trainTarget)

# Evaluate Classifier
print(classification_report(testTarget,model.predict(testData),target_names=targetNames))

# Iterate Through Sample Images
for i in np.random.choice(np.arange(0,len(imagePaths)),10):
	# Assign Image/Mask Paths
	imagePath = imagePaths[i]
	maskPath = maskPaths[i]
	
	# Load Image/Mask
	image = cv2.imread(imagePath)
	mask = cv2.imread(maskPath)
	mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
	
	# Describe Image
	features = desc.describe(image,mask)
	
	# Predict/Classify Image
	flower = le.inverse_transform(model.predict([features]))[0]
	
	# Print Image Path And Prediction
	print(imagePath)
	print("This Flower Is A {}".format(flower.upper()))

	# Display Image
	cv2.imshow("Image", image)
	cv2.waitKey(0)
	
# Cleanup, If Needed
cv2.destroyAllWindows()