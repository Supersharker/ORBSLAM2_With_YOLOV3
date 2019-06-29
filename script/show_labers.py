import time
import cv2
import numpy as np

# load the class label names
CLASSES = open("/home/cl/Desktop/coco_user/coco.names").read().strip().split("\n")
COLORS = open("/home/cl/Desktop/coco_user/color.txt").read().strip().split("\n")
COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]

# print COLORS
# print CLASSES

size  = 15

# initialize the legend visualization
legend = np.zeros(((len(CLASSES) * size) + 25, 300, 3), dtype="uint8")
# loop over the class names + colors
for (i, (className, color)) in enumerate(zip(CLASSES, COLORS)):
	# draw the class name + color on the legend
	color = [int(c) for c in color]
	cv2.putText(legend, className, (5, (i * size) + 17),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
		# def putText(img, text, org, fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin)
	cv2.rectangle(legend, (100, (i * size)), (300, (i * size) + size),
		tuple(color), -1)
	print color

cv2.imshow("Legend", legend)
cv2.waitKey(0)
# if cv2.waitKey(1) & 0xFF == ord('q'):
# 	exit
