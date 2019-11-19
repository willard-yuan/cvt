#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "../include/baseUtils.h"
#include "../include/videocapture.h"

#include "../include/common.hpp"
#include "../include/GHT.hpp"

extern "C" {
#include "libavutil/avutil.h"
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libswscale/swscale.h"
#include <libavutil/imgutils.h>
}

using namespace std;

const int kShorterEdge = 300;
const float distRatio = 0.9;
const int numThrehold = 10;
const cv::Ptr<cv::Feature2D> detector = cv::xfeatures2d::SIFT::create();

struct localFeatInfo
{
    std::vector<cv::KeyPoint> img_kpts;
    cv::Mat img_descs;
};

localFeatInfo extract(cv::Mat &img)
{
    localFeatInfo siftInfo;
    detector->detect(img, siftInfo.img_kpts);
    
    if ((int)siftInfo.img_kpts.size() < 0){
        std::cout << "number of image keypoints: " << siftInfo.img_kpts.size() << std::endl;
    }
    detector->compute(img, siftInfo.img_kpts, siftInfo.img_descs);
    return siftInfo;
}

void test(const string templePath,
          const string videoDir,
          const string saveDir)
{
    VideoCapture cap;
    
    GHT SVValtor;
    cv::FlannBasedMatcher matcher;
    std::vector<cv::KeyPoint> qeK, obK;
    std::vector<std::vector<cv::DMatch>> matches;
    std::vector<cv::DMatch> good_matches;
    
    cv::Mat obectImg = cv::imread(templePath);
    int maxlen = std::max(obectImg.rows, obectImg.cols);
    cv::Mat resizedObjectImg;
    resizedObjectImg = obectImg;
    localFeatInfo objectImgSiftInfo = extract(resizedObjectImg);
    
    std::vector<std::string> videosPath = globVector(videoDir);
    
    for (int k = 0; k <  videosPath.size(); k++)
    {
        std::string filename = videosPath.at(k);
        // 从文件读取
        //cap.open(filename);
    
        // 从内存中读取
        vector<unsigned char> video_buf;
        load_file(filename, video_buf);
        cap.open(video_buf.data(), video_buf.size());
    
        if (!cap.isOpened())
        {
            cout << "open capture failed!" << endl;
            continue;
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
                ++ num_true_frame;
                std::cout << num_true_frame << "_ith frame" << std::endl;
                
                matches.clear();
                good_matches.clear();
            
                cv::Mat resizedImg;
                ResizeImage(frame, resizedImg, kShorterEdge, true);
                localFeatInfo resizedImgSiftInfo = extract(resizedImg);
            
                if (resizedImgSiftInfo.img_kpts.size() < 3)
                {
                    cout << "num key points of the frame is less than 3!" << endl;
                    continue;
                }
                
                matcher.knnMatch(objectImgSiftInfo.img_descs, resizedImgSiftInfo.img_descs, matches, 2);
                for (size_t i = 0; i < matches.size(); i++)
                {
                    if (matches[i][0].distance < distRatio*matches[i][1].distance)
                    {
                        good_matches.push_back(matches[i][0]);
                        qeK.push_back(objectImgSiftInfo.img_kpts[matches[i][0].queryIdx]);
                        obK.push_back(resizedImgSiftInfo.img_kpts[matches[i][0].trainIdx]);
                    }
                }
            
                int numInliners = SVValtor.getInliers(objectImgSiftInfo.img_kpts, resizedImgSiftInfo.img_kpts, good_matches);
                if (numInliners < numThrehold)
                {
                    cout << "num key points of the frame is less than 3!" << endl;
                    continue;
                }
                std::vector<cv::DMatch> matchedInliners = SVValtor.getSVFMatches();
            
                qeK.clear();
                obK.clear();
                for (size_t i = 0; i < matchedInliners.size(); i++)
                {
                    qeK.push_back(objectImgSiftInfo.img_kpts[matchedInliners[i].queryIdx]);
                    obK.push_back(resizedImgSiftInfo.img_kpts[matchedInliners[i].trainIdx]);
                }
            
                cv::Mat matchedImage = drawMatch(resizedObjectImg, resizedImg, qeK, obK);
                if (matchedImage.empty())
                {
                    continue;
                }

                cv::resize(matchedImage, matchedImage, cv::Size(matchedImage.cols/1, matchedImage.rows/1));
                //cv::imshow("GHT Matched inlines", matchedImage);
                //cv::waitKey(1);
                cv::imwrite(saveDir + "/" + removeExtension(getBaseName(filename, "/")) + "_" + std::to_string(num_true_frame) +  ".jpg", frame);
            } else {
                if (num_true_frame > 100)
                {
                    std::cout << "finished reading!" << std::endl;
                    break;
                }
            }
        }
    }
}

int main()
{
    const string videoDir = "./vdata/*";
    const string templePath = "./data/test_1.jpg";
    const string saveDir = "./saveTmp";
    test(templePath, videoDir, saveDir);
}
