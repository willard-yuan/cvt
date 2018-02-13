
#include "common.hpp"

void drawMatch(cv::Mat &srcImg, cv::Mat &objImg, const std::vector<cv::KeyPoint> &srcPts, const std::vector<cv::KeyPoint> &dstPts)
{
    
    bool opinion = true;
    
    // Create a image for displaying mathing keypoints
    cv::Size sz = cv::Size(srcImg.size().width + objImg.size().width, srcImg.size().height + objImg.size().height);
    cv::Mat matchingImage = cv::Mat::zeros(sz, CV_8UC3);
    
    // Draw camera frame
    cv::Mat roi1 = cv::Mat(matchingImage, cv::Rect(0, 0, srcImg.size().width, srcImg.size().height));
    srcImg.copyTo(roi1);
    // Draw original image
    cv::Mat roi2;
    if (opinion)
    {
        roi2 = cv::Mat(matchingImage, cv::Rect(srcImg.size().width, srcImg.size().height, objImg.size().width, objImg.size().height));
    }else{
        roi2 = cv::Mat(matchingImage, cv::Rect(srcImg.size().width, 0, objImg.size().width, objImg.size().height));
    }
    objImg.copyTo(roi2);
    
    // Draw line between nearest neighbor pairs
    std::vector<cv::KeyPoint> tmpKps;
    for (int i = 0; i < (int)srcPts.size(); ++i) {
        cv::Point2f pt1 = srcPts[i].pt;
        cv::Point2f pt2 = dstPts[i].pt;
        cv::Point2f from = pt1;
        cv::Point2f to;
        
        cv::KeyPoint tmp;
        tmp.pt = from;
        tmpKps.push_back(tmp);
        
        if (opinion){
            to = cv::Point(srcImg.size().width + pt2.x, srcImg.size().height + pt2.y);
            cv::KeyPoint tmp = srcPts[i];
            tmp.pt = to;
            tmpKps.push_back(tmp);
        }else{
            to = cv::Point(srcImg.size().width + pt2.x, pt2.y);
            cv::KeyPoint tmp = srcPts[i];
            tmp.pt = to;
            tmpKps.push_back(tmp);
        }
        cv::line(matchingImage, from, to, cv::Scalar(0, 255, 255), 1);
    }
    cv::drawKeypoints(matchingImage, tmpKps, matchingImage);
    // Display mathing image
#if 1
    cv::resize(matchingImage, matchingImage, cv::Size(matchingImage.cols/2, matchingImage.rows/2));
#endif
    cv::namedWindow( "Display frame",CV_WINDOW_AUTOSIZE);
    cv::imshow("Matched Points", matchingImage);
    cv::waitKey(0);
}

int ResizeImage(const cv::Mat& raw_img, cv::Mat& img, const int kShorterEdge, bool resize) {
    if(!resize)
    {
        img = raw_img;
        return 0;
    }
    double ratio;
    if(raw_img.rows < raw_img.cols) {
        ratio = double(kShorterEdge / double(raw_img.rows));
    }else {
        ratio = double(kShorterEdge / double(raw_img.cols));
    }
    cv::Rect my_roi(raw_img.cols >> 3, raw_img.rows >> 3,  (raw_img.cols * 3) >> 2, (raw_img.rows *3) >> 2);
    cv::Mat croped_img = raw_img(my_roi);
    cv::resize(croped_img, img, cv::Size(0,0), ratio, ratio);
    return 0;
}

bool idxdist(cv::DMatch first, cv::DMatch second)
{
    if (first.trainIdx == second.trainIdx)
        return first.distance < second.distance;
    return first.trainIdx < second.trainIdx;
}
