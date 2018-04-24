
#include "SVF.hpp"
#include "common.hpp"

const int kShorterEdge = 300;
const float distRatio = 0.8;

struct localFeatInfo
{
    std::vector<cv::KeyPoint> img_kpts;
    cv::Mat img_descs;
};

localFeatInfo extract(cv::Mat &img)
{
    localFeatInfo siftInfo;
    cv::Ptr<cv::Feature2D> detector = cv::xfeatures2d::SIFT::create();
    
    detector->detect(img, siftInfo.img_kpts);
    
    if ((int)siftInfo.img_kpts.size() < 0){
        std::cout << "number of image keypoints: " << siftInfo.img_kpts.size() << std::endl;
    }
    detector->compute(img, siftInfo.img_kpts, siftInfo.img_descs);
    return siftInfo;
}

std::vector<cv::DMatch> delRepeat(std::vector<cv::DMatch> & rawMatches)
{
    std::vector<cv::DMatch> goodMatches;
    if(rawMatches.size() <= 0) return goodMatches;
    std::sort(rawMatches.begin(), rawMatches.end(), idxdist);
    
    goodMatches.push_back(rawMatches[0]);
    int base = rawMatches[0].trainIdx;
    for(int i = 1; i < rawMatches.size(); ++i)
    {
        if(base != rawMatches[i].trainIdx){
            base = rawMatches[i].trainIdx;
            goodMatches.push_back(rawMatches[i]);
        }
    }
    return goodMatches;
}

int main(int argc, const char * argv[]) {
    
    cv::Mat img1 = cv::imread("/Users/willard/Pictures/111/1446522480_120.jpg");
    cv::Mat Resizeimg1;
    ResizeImage(img1, Resizeimg1, kShorterEdge, true);
    localFeatInfo siftInfo_1 = extract(Resizeimg1);
    
    cv::Mat img2 = cv::imread("/Users/willard/Pictures/111/4917623158_60.jpg");
    cv::Mat Resizeimg2;
    ResizeImage(img2, Resizeimg2, kShorterEdge, true);
    localFeatInfo siftInfo_2 = extract(Resizeimg2);
    
    cv::FlannBasedMatcher matcher;
    std::vector<cv::KeyPoint> qeK, obK;
    std::vector<std::vector<cv::DMatch>> matches;
    std::vector<cv::DMatch> good_matches2;
    
    matcher.knnMatch(siftInfo_1.img_descs, siftInfo_2.img_descs, matches, 2);
    
    for (size_t i = 0; i < matches.size(); i++){
        if (matches[i][0].distance < distRatio*matches[i][1].distance){
            good_matches2.push_back(matches[i][0]);
            qeK.push_back(siftInfo_1.img_kpts[matches[i][0].queryIdx]);
            obK.push_back(siftInfo_2.img_kpts[matches[i][0].trainIdx]);
        }
    }
    
    drawMatch(Resizeimg1, Resizeimg2, qeK, obK);
    
    SVF SVValtor;
    int num_inliners = SVValtor.getInliers(siftInfo_1.img_kpts, siftInfo_2.img_kpts, good_matches2);
    std::vector<cv::DMatch> vali_matches = SVValtor.getSVFMatches();
    
    qeK.clear();
    obK.clear();
    for (size_t i = 0; i < vali_matches.size(); i++)
    {
        qeK.push_back(siftInfo_1.img_kpts[vali_matches[i].queryIdx]);
        obK.push_back(siftInfo_2.img_kpts[vali_matches[i].trainIdx]);
    }
    
    drawMatch(Resizeimg1, Resizeimg2, qeK, obK);
    printf("number of inliners: %d\n", num_inliners);
    return 0;
}
