#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "baseUtils.h"
#include "videocapture.h"

extern "C" {
#include "libavutil/avutil.h"
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libswscale/swscale.h"
#include <libavutil/imgutils.h>
}

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
    double frame_time = n / fps;   // ms
    
    cv::Mat frame(h, w, CV_8UC3);
    int num_true_frame = 0;
    for(int i = 0; i < n; i++)
    {
        if(cap.read(frame))
        {
            cv::imshow("frame", frame);
            cv::waitKey(1);
            ++ num_true_frame;
        } else {
            if (num_true_frame > 100) {
                std::cout << "finished reading" << std::endl;
                break;
            }
        }
    }
}

int main()
{
    test("/Users/willard/Movies/dog.mp4");
}
