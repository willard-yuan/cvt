
#include <stdio.h>
#include <time.h>
#include <glob.h>
#include "svf_array.hpp"


int main(int argc, char** argv) {
    
    float siftRatio = 0.8;
    
    cv::Ptr<cv::Feature2D> detector = cv::xfeatures2d::SIFT::create();
    cv::FlannBasedMatcher siftMatcher;
    
    cv::Mat im1 = cv::imread("/Users/willard/Pictures/0509/WechatIMG81.jpeg", 1);
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

    cv::Mat im2 = cv::imread("/Users/willard/Pictures/0509/WechatIMG82.jpeg", 1);
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
    
    // 几何重排
    std::vector<std::vector<cv::DMatch>> matches;
    std::vector<cv::DMatch> goodMatches2;
    siftMatcher.knnMatch(img1Descs, img2Descs, matches, 2);
    for (size_t i = 0; i < matches.size(); i++){
        if (matches[i][0].distance < siftRatio*matches[i][1].distance){
            goodMatches2.push_back(matches[i][0]);
        }
    }
    std::vector<cv::DMatch> refineMatches = getInliers(kpts1, kpts2, goodMatches2);
    
    cv::Mat img_matches;
    cv::drawMatches(im1, kpts1, im2, kpts2, refineMatches, img_matches);
    cv::imshow("match", img_matches);
    cv::waitKey();
    
    return 0;
}
