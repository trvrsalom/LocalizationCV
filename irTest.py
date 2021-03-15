import argparse
import cv2
import time
import imutils
import grip
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("input", help="path to the input file");
args = vars(ap.parse_args())

vs = cv2.VideoCapture(args["input"]);

time.sleep(2.0)

pipeline = grip.GripPipeline()

while True:
	frame = vs.read()
	frame = frame[1] if args.get("input", False) else frame
	if frame is None:
		vs.set(cv2.CAP_PROP_POS_FRAMES, 0)
		continue
	inp = imutils.resize(frame, width=600)
	pipeline.process(inp)
	out = pipeline.find_blobs_output
	#cv2.imshow("input", inp)
	outImage = np.copy(inp)
	if len(out) <= 2: # Valid number of points detected
		pos = [out[i].pt for i in range(len(out))]
		size = [out[0].size for i in range(len(out))]
		for i in range(len(out)):
			cv2.circle(outImage,
			           (int(pos[i][0]), int(pos[i][1])),
			           int(size[i]), (255, 255, 255), 2);
		#cv2.imshow("output", outImage)
	else:
		print("Too many markers found.");
	vert = np.concatenate((inp, outImage), axis=0)
	cv2.imshow("Results", vert)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

vs.release()
cv2.destroyAllWindows()