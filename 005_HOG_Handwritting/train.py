# Usage
# python train.py --dataset data/digits.csv --model models/svm.cpickle


# Import Needed Packages
from sklearn.svm import LinearSVC
from helpers.hog import HOG
from helpers import dataset
import argparse
import joblib


# Construct Argument Parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path To Dataset File")
ap.add_argument("-m", "--model", required=True, help="Path To Where Model Will Be Stored")
args = vars(ap.parse_args())

# Load Dataset And Init Data Matrix
(digits,target) = dataset.load_digits(args["dataset"])
data = []

# Init HOG Descriptor
hog = HOG(orientations=18, pixelsPerCell=(10,10), cellsPerBlock=(1,1), transform=True)

# Iterate Through Images
for image in digits:
	# Deskew And Center Image
	image = dataset.deskew(image,20)
	image = dataset.center_extent(image,(20,20))
	
	# Describe Image And Update Data Matrix
	hist = hog.describe(image)
	data.append(hist)
	
# Train Model
model = LinearSVC(random_state=42)
model.fit(data,target)

# Dump Model To File
joblib.dump(model, args["model"])