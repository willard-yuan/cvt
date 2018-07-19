
#include <stdio.h>
#include <time.h>
#include <glob.h>
#include "svf.hpp"

uint64_t constexpr mix(char m, uint64_t s)
{
    return ((s<<7) + ~(s>>3)) + ~m;
}

uint64_t constexpr hash(const char * m)
{
    return (*m) ? mix(*m,hash(m+1)) : 0;
}

int main(int argc, char** argv) {
    
    const char* method = "1nn";
    float siftRatio = 0.85;
    float sift_near_thresh = 0.65;
    sift_near_thresh *= 512*512*sift_near_thresh;
    bool removeRepeat = true;
    bool useRootSIFT = true;
    
    cv::Ptr<cv::Feature2D> detector = cv::xfeatures2d::SIFT::create();
    cv::FlannBasedMatcher siftMatcher;
    
    cv::Mat im1 = cv::imread("/Users/willard/svf/50521531988253_.pic.jpg", 1);
    if (im1.empty()) return 0;
    std::vector<cv::KeyPoint> kpts1;
    cv::Mat img1Descs;
    detector->detect(im1, kpts1);
    if (kpts1.size() < 3) {
        std::cout << "too few feature points: " << std::endl;
        return 0;
    }
    detector->compute(im1, kpts1, img1Descs);

    cv::Mat im2 = cv::imread("/Users/willard/svf/50531531988253_.pic.jpg", 1);
    if (im2.empty()) return 0;
    std::vector<cv::KeyPoint> kpts2;
    cv::Mat img2Descs;
    detector->detect(im2, kpts2);
    if (kpts2.size() < 3) {
        std::cout << "too few feature points: " << std::endl;
        return 0;
    }
    detector->compute(im2, kpts2, img2Descs);
    
    if (useRootSIFT){
        rootSift(img1Descs);
        rootSift(img2Descs);
    }
    
    // 几何重排
    std::vector<std::vector<cv::DMatch>> matches;
    std::vector<cv::DMatch> goodMatches2;
    siftMatcher.knnMatch(img1Descs, img2Descs, matches, 2);
    for (size_t i = 0; i < matches.size(); i++){
        switch(hash(method))
        {
            case hash("1nn"):
                if (matches[i][0].distance < sift_near_thresh){
                    goodMatches2.push_back(matches[i][0]);
                }
            case hash("2nn"):
                if (matches[i][0].distance < siftRatio*matches[i][1].distance){
                    goodMatches2.push_back(matches[i][0]);
                }
        }
    }
    std::vector<cv::DMatch> refineMatches = getInliers(kpts1, kpts2, goodMatches2, removeRepeat);
    
    cv::Mat img_matches;
    cv::drawMatches(im1, kpts1, im2, kpts2, refineMatches, img_matches);
    //cv::resize(img_matches, img_matches, cv::Size(int(0.5*img_matches.rows), int(0.5*img_matches.cols)));
    cv::imshow("match", img_matches);
    cv::waitKey();
    
    return 0;
}
