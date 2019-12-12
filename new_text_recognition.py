#import packages
from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import argparse #for command line arguments
import cv2

def decode_predictions(scores, geometry):
	# get the num Rows and Cols from scores
	(numRows, numCols) = scores.shape[2:4]
	# initialize our bounding boxes and confidence score lists
	rects = []
	confidences = []

	# loop over scores and associated bounding box geometry
	for y in range(0, numRows):
		#extract scores
		scoresData = scores[0, 0, y]

		#extract geometric bounding box coordinates
		corner0 = geometry[0, 0, y]
		corner1 = geometry[0, 1, y]
		corner2 = geometry[0, 2, y]
		corner3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# loop over columns of data
		for x in range(0, numCols):
			#ignore scores that are too low
			if scoresData[x] < args["min_confidence"]:
				continue

			#compute offset factor (feature maps 4x smaller than input image)
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			#extract and compute angle data
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			#compute bounding box dimensions
			height = corner0[x] + corner2[x]
			width = corner1[x] + corner3[x]

			#compute start and end (x, y) coords for bounding box
			#note boxes will account for rotated text, but box not rotated
			endX = int(offsetX + (cos * corner1[x]) + (sin * corner2[x]))
			endY = int(offsetY - (sin * corner1[x]) + (cos * corner2[x]))
			startX = int(endX - width)
			startY = int(endY - height)

			#add coordinates to rects
			rects.append((startX, startY, endX, endY))
			#add score to confidence list
			confidences.append(scoresData[x])

	#return tuple of bounding boxes and confidence scores
	return (rects, confidences)


#argument parsing
ap = argparse.ArgumentParser()
ap.add_argument()
