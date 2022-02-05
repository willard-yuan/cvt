#include <iostream>

#include <opencv2/opencv.hpp>

#include "../include/pupil_tracker.h"
#include "../include/cvx.h"
#include "../include/eye_center.h"
#include "../include/erase_specular.h"


using namespace std;
using namespace cv;

void imshowscale(const std::string& name, cv::Mat& m, double scale)
{
    cv::Mat res;
    cv::resize(m, res, cv::Size(), scale, scale, cv::INTER_NEAREST);
    cv::imshow(name, res);
}

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
    if ( event == EVENT_LBUTTONDOWN )
    {
        cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
        auto k = cv::waitKey(0);
        if(k == 's'){
            return;
        }
    }else if ( event == EVENT_LBUTTONUP ){
    }
}

int main(int argc, char* argv[])
{
    //test_centermap();
    //test_erase_specular();

    //读取视频
	//cv::VideoCapture cap("/Users/willard/data_sets/eyeVideo/pupillab.mp4");
    cv::VideoCapture cap("/Users/willard/data_sets/eyeVideo/ee8.avi");
    if(!cap.isOpened()){
        std::cout << "Unable to open the camera\n";
        std::exit(-1);
    }

    cv::Mat frame;

    int i = 0;
    while(true) {
        cap >> frame;
        if(frame.empty()) {
            std::cout << "Can't read frames from your camera\n";
            break;
        }

        cout << "frame:  " << ++i << endl;

        pupiltracker::findPupilEllipse_out out;
        pupiltracker::findPupilEllipse_out outDebug;

        double fScale = 1;
        int r_min = 60;

        pupiltracker::TrackerParams params;
        //params.Radius_Min = (int)r_min*fScale;
        //params.Radius_Max = (int)2*r_min*fScale; // r_max = 2*r_min，不一定要满足这样的关系，这里这么做是为了减少调参的数目
        params.Radius_Min = (int)1*fScale;
        params.Radius_Max = (int)90*fScale; // r_max = 2*r_min，不一定要满足这样的关系，这里这么做是为了减少调参的数目

        //params.Radius_Min = (int)20*fScale;
        //params.Radius_Max = (int)40*fScale;

        params.CannyBlur = 1;
        params.CannyThreshold1 = 20;
        params.CannyThreshold2 = 40;
        params.StarburstPoints = 0;

        params.PercentageInliers = 30;
        //params.InlierIterations = 2;
        params.InlierIterations = 1;
        params.ImageAwareSupport = true;
        params.EarlyTerminationPercentage = 95;
        params.EarlyRejection = true;
        params.Seed = -1;

        cv::Mat m = frame;

        // 缩放图片
        cv::imshow("Haar Pupil", m);
        cv::Size2f dsize = Size(m.cols*fScale, m.rows*fScale);
        cv::resize(m, m, dsize);
        imshow("Haar Pupil1", m);
        //cv::waitKey();

        bool eraseSpecular = false;
        pupiltracker::findPupilEllipse(params, m, out, eraseSpecular);
        //pupiltracker::findPupilEllipse(params, m, outDebug, eraseSpecular);

        // 显示Haar响应得到的眼睛区域
        namedWindow("Haar Pupil", WINDOW_NORMAL);
        cv::moveWindow("Haar Pupil", 1020, 10);
        imshowscale("Haar Pupil", out.mHaarPupil, 3);
        cv::resizeWindow("Haar Pupil", 300, 300);

        namedWindow("Pupil", WINDOW_NORMAL);
        cv::moveWindow("Pupil", 1020, 450);
        imshowscale("Pupil", out.mPupil, 3);
        cv::resizeWindow("Pupil", 300, 300);

        // 显示经过阈值化后的瞳孔区域
        namedWindow("Thresh Pupil",WINDOW_NORMAL);
        cv::moveWindow("Thresh Pupil", 50, 10);
        imshowscale("Thresh Pupil", out.mPupilThresh, 3);
        cv::resizeWindow("Thresh Pupil", 300, 300);

        //显示轮廓
        namedWindow("Edges", WINDOW_NORMAL);
        cv::moveWindow("Edges", 50, 450);
        imshowscale("Edges", out.mPupilEdges, 3);
        cv::resizeWindow("Edges", 300, 300);

        cv::Point2f localPos = out.pPupil;
        //画十字架
        cv::line(frame,cv::Point(localPos.x,0),cv::Point(localPos.x,frame.rows),CV_RGB(255,255,0),2,cv::LINE_AA,0);
        cv::line(frame,cv::Point(0,localPos.y),cv::Point(frame.cols,localPos.y),CV_RGB(255,255,0),2,cv::LINE_AA,0);

        cv::Rect rmin(cv::Point(out.pPupil.x-params.Radius_Min, out.pPupil.y-params.Radius_Min),cv::Point(out.pPupil.x+params.Radius_Min, out.pPupil.y+params.Radius_Min));
        cv::Rect rmax(cv::Point(out.pPupil.x-params.Radius_Max, out.pPupil.y-params.Radius_Max),cv::Point(out.pPupil.x+params.Radius_Max, out.pPupil.y+params.Radius_Max));
        cv::rectangle(frame,rmin,Scalar(255,0,255),2,8,0);
        cv::rectangle(frame,rmax,Scalar(255,0,255),2,8,0);

        // 画椭圆中心
        pupiltracker::cvx::cross(frame, out.pPupil, 2, pupiltracker::cvx::rgb(255, 0, 255));

        // 画椭圆
        cv::ellipse(m, out.elPupil, pupiltracker::cvx::rgb(255, 0, 0), 2);
        cv::ellipse(frame, outDebug.elPupil, pupiltracker::cvx::rgb(0, 0, 255), 2);

        namedWindow("Pupil Tracking", cv::WINDOW_NORMAL);
        cv::moveWindow("Pupil Tracking", 360, 200);
        cv::imshow("Pupil Tracking", frame);

        cv::waitKey(1);

        // 设置鼠标暂停
        setMouseCallback("Pupil Tracking", CallBackFunc, NULL);

        if (cv::waitKey(10) != -1)
            break;
    }
    return 0;
}

