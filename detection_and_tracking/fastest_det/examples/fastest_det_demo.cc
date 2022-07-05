#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

class FastestDet
{
public:
	FastestDet(float confThreshold, float nmsThreshold);
	void detect(Mat& frame);
private:
	const int inpWidth = 512;
	const int inpHeight = 512;
	vector<string> class_names;
	int num_class;

	float confThreshold;
	float nmsThreshold;
	Net net;
	void drawPred(float conf, int left, int top, int right, int bottom, Mat& frame, int classid);
};

FastestDet::FastestDet(float confThreshold, float nmsThreshold)
{
	this->confThreshold = confThreshold;
	this->nmsThreshold = nmsThreshold;

	this->net = readNet("../model/FastestDet.onnx");
	ifstream ifs("../model/coco.names");
	string line;
	while (getline(ifs, line)) this->class_names.push_back(line);
	this->num_class = class_names.size();
}

void FastestDet::drawPred(float conf, int left, int top, int right, int bottom, Mat& frame, int classid)   // Draw the predicted bounding box
{
	//Draw a rectangle displaying the bounding box
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255), 2);

	//Get the label for the class name and its confidence
	string label = format("%.2f", conf);
	label = this->class_names[classid] + ":" + label;

	//Display the label at the top of the bounding box
	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = max(top, labelSize.height);
	//rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
}

inline float sigmoid(float x)
{
	return 1.0 / (1 + expf(-x));
}

void FastestDet::detect(Mat& frame)
{
	Mat blob = blobFromImage(frame, 1 / 255.0, Size(this->inpWidth, this->inpHeight), Scalar(0, 0, 0), false, false);
	this->net.setInput(blob);
	vector<Mat> outs;
	this->net.forward(outs, this->net.getUnconnectedOutLayersNames());

	int num_proposal = outs[0].size[0];
	int nout = outs[0].size[1];
	/////generate proposals
	vector<float> confidences;
	vector<Rect> boxes;
	vector<int> classIds;

	int i = 0, j = 0, row_ind = 0; ///box_score, xmin,ymin,xamx,ymax,class_score
	const int num_grid_x = 32;
	const int num_grid_y = 32;
	float* pdata = (float*)outs[0].data;

	for (i = 0; i < num_grid_y; i++)
	{
		for (j = 0; j < num_grid_x; j++)
		{
			Mat scores = outs[0].row(row_ind).colRange(5, nout);
			Point classIdPoint;
			double max_class_socre;
			// Get the value and location of the maximum score
			minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
			max_class_socre *= pdata[0];
			if (max_class_socre > this->confThreshold)
			{
				const int class_idx = classIdPoint.x;
				float cx = (tanh(pdata[1]) + j) / (float)num_grid_x;  ///cx
				float cy = (tanh(pdata[2]) + i) / (float)num_grid_y;   ///cy
				float w = sigmoid(pdata[3]);   ///w
				float h = sigmoid(pdata[4]);  ///h

				cx *= float(frame.cols);
				cy *= float(frame.rows);
				w *= float(frame.cols);
				h *= float(frame.rows);

				int left = int(cx - 0.5 * w);
				int top = int(cy - 0.5 * h);

				confidences.push_back((float)max_class_socre);
				boxes.push_back(Rect(left, top, int(w), int(h)));
				classIds.push_back(class_idx);
			}
			row_ind++;
			pdata += nout;
		}
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	vector<int> indices;
	dnn::NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		this->drawPred(confidences[idx], box.x, box.y,
			box.x + box.width, box.y + box.height, frame, classIds[idx]);
	}
}

int main()
{
	FastestDet FastestDet_model(0.8, 0.35);
	string imgpath = "../data/3.jpg";
	Mat srcimg = imread(imgpath);
	FastestDet_model.detect(srcimg);

	static const string kWinName = "Deep learning object detection in OpenCV";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, srcimg);
	waitKey(0);
	destroyAllWindows();
}
