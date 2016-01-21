import cv2
import csfm
import numpy as np

image = cv2.imread('test.png', cv2.CV_LOAD_IMAGE_COLOR)
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
points, desc = csfm.hahog(img.astype(np.float32) / 255, 0, False, True)

imgFloat = img.astype(np.float32) / 255
for x in imgFloat[0,0:10]:
    print "{:10.4f}".format(x)

points, desc = csfm.hahog(img.astype(np.float32) / 255, 0.01, 10, 0, False, True)

cv.imshow('dst_rt', img)
cv.waitKey(0)
cv.destroyAllWindows()
