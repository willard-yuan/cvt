import cv2
import numpy as np
import argparse

class FastestDet():
    def __init__(self, confThreshold=0.3, nmsThreshold=0.4):
        self.classes = list(map(lambda x: x.strip(), open('coco.names',
                                                          'r').readlines()))  ###这个是在coco数据集上训练的模型做opencv部署的，如果你在自己的数据集上训练出的模型做opencv部署，那么需要修改self.classes
        self.inpWidth = 512
        self.inpHeight = 512
        self.net = cv2.dnn.readNet('FastestDet.onnx')
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.H, self.W = 32, 32
        self.grid = self._make_grid(self.W, self.H)

    def _make_grid(self, nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)

    def postprocess(self, frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []
        for detection in outs:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId] * detection[0]
            if confidence > self.confThreshold:
                center_x = int(detection[1] * frameWidth)
                center_y = int(detection[2] * frameHeight)
                width = int(detection[3] * frameWidth)
                height = int(detection[4] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                # confidences.append(float(confidence))
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold).flatten()
        for i in indices:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            frame = self.drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)
        return frame

    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=2)

        label = '%.2f' % conf
        label = '%s:%s' % (self.classes[classId], label)

        # Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255,255,255), cv.FILLED)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=1)
        return frame

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def detect(self, srcimg):
        blob = cv2.dnn.blobFromImage(srcimg, 1 / 255.0, (self.inpWidth, self.inpHeight))
        self.net.setInput(blob)
        pred = self.net.forward(self.net.getUnconnectedOutLayersNames())[0]
        pred[:, 3:5] = self.sigmoid(pred[:, 3:5])  ###w,h
        pred[:, 1:3] = (np.tanh(pred[:, 1:3]) + self.grid) / np.tile(np.array([self.W,self.H]), (pred.shape[0], 1)) ###cx,cy
        srcimg = self.postprocess(srcimg, pred)
        return srcimg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str, default='data/3.jpg', help="image path")
    parser.add_argument('--confThreshold', default=0.8, type=float, help='class confidence')
    parser.add_argument('--nmsThreshold', default=0.35, type=float, help='nms iou thresh')
    args = parser.parse_args()

    srcimg = cv2.imread(args.imgpath)
    model = FastestDet(confThreshold=args.confThreshold, nmsThreshold=args.nmsThreshold)
    srcimg = model.detect(srcimg)

    winName = 'Deep learning object detection in OpenCV'
    cv2.namedWindow(winName, 0)
    cv2.imshow(winName, srcimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()