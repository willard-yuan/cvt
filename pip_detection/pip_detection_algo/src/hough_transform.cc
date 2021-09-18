#include "../include/hough_transform.h"

namespace mmu {
    HoughTransform::HoughTransform() {
        hough_transform_threshold_max = 1000;
        hough_transform_threshold = 300;
        type = STANDARD;
        imgScale = 0.4;
    }
    
    LineDetectInfo HoughTransform::detectLines(cv::Mat & input_frame, const std::string & frameId){
        std::vector<cv::Vec2f> lines;
        cv::HoughLines(input_frame, lines, 1, CV_PI/180, hough_transform_threshold, 0, 0);
        
        std::vector<std::pair<cv::Point, cv::Point>> lines_h;
        std::vector<std::pair<cv::Point, cv::Point>> lines_v;   
        std::vector<float> ycoords;
        std::vector<float> xcoords;
        
        for(int i = 0; i < MIN(lines.size(), 100); i++) {
            float rho = lines[i][0];
            float theta = lines[i][1];
            cv::Point pt1, pt2;
            double a = cos(theta), b = sin(theta);
            double x0 = a*rho, y0 = b*rho;
            pt1.x = cvRound(x0 + 10000*(-b));
            pt1.y = cvRound(y0 + 10000*(a));
            pt2.x = cvRound(x0 - 10000*(-b));
            pt2.y = cvRound(y0 - 10000*(a));
            theta = atan( (double)(pt2.y - pt1.y)/(pt2.x - pt1.x) );
            float degree = theta*180/CV_PI;
            if(fabs(degree) < 3 && pt1.y >= 10){
                lines_h.push_back(std::pair<cv::Point, cv::Point>(pt1, pt2));
                ycoords.push_back(pt1.y);
            }
            if(fabs(degree) < 92 && fabs(degree) > 88 && pt1.x >= 10){
                lines_v.push_back(std::pair<cv::Point, cv::Point>(pt1, pt2));
                xcoords.push_back(pt1.x);
            }
        }
        
        sort(ycoords.begin(), ycoords.end());
        ycoords.erase(unique(ycoords.begin(), ycoords.end()), ycoords.end());
        
        sort(xcoords.begin(), xcoords.end());
        xcoords.erase(unique(xcoords.begin(), xcoords.end()), xcoords.end());
        
        LineDetectInfo dInfo;
        dInfo.xSelectcoords = xcoords;
        dInfo.ySelectcoords = ycoords;
        
        return dInfo;
    } 
} //end namespce mmu
