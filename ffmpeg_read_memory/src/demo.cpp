#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "baseUtils.h"
#include "videocapture.h"

using namespace std;

void test(const string filename)
{
    VideoCapture cap;
    
    // 从文件读取
    //cap.open(filename);
    
    // 从内存中读取
    vector<unsigned char> video_buf;
    load_file(filename, video_buf);
    cap.open(video_buf.data(), video_buf.size());
    
    if (!cap.isOpened())
    {
        cout << "open capture failed!" << endl;
        return;
    }

    int h = (int)cap.get(VideoCapture::CAP_PROP_FRAME_HEIGHT);
    int w = (int)cap.get(VideoCapture::CAP_PROP_FRAME_WIDTH);
    double fps = cap.get(VideoCapture::CAP_PROP_FPS);
    int n = std::max((int)cap.get(VideoCapture::CAP_PROP_FRAME_COUNT), 10000);
    double frame_time = 1000.0 / fps;   // ms
    
    cv::Mat frame(h, w, CV_8UC3);
    int t0 = 0, t1 = 0;
    int wait_time = 0;
    int num_true_frame = 0;
    for(int i = 0; i < n; i++)
    {
        t0 = cv::getTickCount();
        if(cap.read(frame))
        {
            cv::imshow("frame", frame);
            cv::waitKey(wait_time);
            ++ num_true_frame;
        }
        t1 = cv::getTickCount();
        wait_time = int(frame_time - (t1 -t0)*1000/cv::getTickFrequency());
    }
}

int main()
{
    test("/Users/willard/Downloads/16693098594_org.mp4");
}
