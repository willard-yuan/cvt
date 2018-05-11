
#include <stdio.h>
#include <time.h>
#include <glob.h>
#include "svf_array.hpp"

int main(int argc, char** argv) {
    
    cv::Ptr<cv::Feature2D> detector = cv::xfeatures2d::SIFT::create();
    cv::FlannBasedMatcher siftMatcher;
    
    cv::Mat im1 = cv::imread("/Users/willard/Pictures/hnsw_queries/bg5.jpg", 1);
    if (im1.empty()) return 0;
    std::vector<cv::KeyPoint> kpts1;
    cv::Mat img1Descs;
    detector->detect(im1, kpts1);
    if (kpts1.size() < 3) {
        std::cout << "too few feature points: " << std::endl;
        return 0;
    }
    detector->compute(im1, kpts1, img1Descs);
    rootSift(img1Descs);
    // 转成1维, 1维转为OpenCV SIFT格式
    float *img1DescsArray = new float[img1Descs.rows*135+1];
    cvtSIFT2Array(img1Descs, kpts1, img1DescsArray);
    cv::Mat img1DescsTest;
    std::vector<cv::KeyPoint> kpts1Test;
    cvtARRAY2CVMat(img1DescsTest, kpts1Test, img1DescsArray);
    
    
    cv::Mat im2 = cv::imread("/Users/willard/Pictures/hnsw_queries/bg5.jpg", 1);
    if (im2.empty()) return 0;
    std::vector<cv::KeyPoint> kpts2;
    cv::Mat img2Descs;
    detector->detect(im2, kpts2);
    if (kpts2.size() < 3) {
        std::cout << "too few feature points: " << std::endl;
        return 0;
    }
    detector->compute(im2, kpts2, img2Descs);
    rootSift(img2Descs);
    // 转成1维, 1维转为OpenCV SIFT格式
    float *img2DescsArray = new float[img1Descs.rows*135+1];
    cv::Mat img2DescsTest;
    std::vector<cv::KeyPoint> kpts2Test;
    cvtSIFT2Array(img2Descs, kpts2, img2DescsArray);
    cvtARRAY2CVMat(img2DescsTest, kpts2Test, img2DescsArray);
    
    // 几何重排
    std::vector<cv::DMatch> refineMatches = getInliers(img1DescsArray, img2DescsArray);
    
    cv::Mat img_matches;
    cv::drawMatches(im1, kpts1Test, im2, kpts2Test, refineMatches, img_matches);
    cv::imshow("match", img_matches);
    cv::waitKey();
    
    return 0;
}
