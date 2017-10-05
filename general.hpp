/* author: yongyuan.name */

#ifndef general_hpp
#define general_hpp

#include <vector>
#include <iostream>
#include <armadillo>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>


// generate ramdom color
static cv::Scalar randomColor(cv::RNG& rng)
{
    int icolor = (unsigned)rng;
    return cv::Scalar(icolor&0xFF, (icolor>>8)&0xFF, (icolor>>16)&0xFF);
}


// ref: https://gist.github.com/thorikawa/3398619
void plotMatches(const cv::Mat &srcColorImage,
                 const cv::Mat &dstColorImage,
                 std::vector<cv::Point2f> &srcPoints,
                 std::vector<cv::Point2f> &dstPoints)
{
    // Create a image for displaying mathing keypoints
    cv::Size sz = cv::Size(srcColorImage.size().width + dstColorImage.size().width, srcColorImage.size().height + dstColorImage.size().height);
    cv::Mat matchingImage = cv::Mat::zeros(sz, CV_8UC3);
    
    // Draw camera frame
    cv::Mat roi1 = cv::Mat(matchingImage, cv::Rect(0, 0, srcColorImage.size().width, srcColorImage.size().height));
    srcColorImage.copyTo(roi1);
    // Draw original image
    cv::Mat roi2 = cv::Mat(matchingImage, cv::Rect(srcColorImage.size().width, srcColorImage.size().height, dstColorImage.size().width, dstColorImage.size().height));
    //cv::Mat roi2 = cv::Mat(matchingImage, cv::Rect(srcColorImage.size().width, 0, dstColorImage.size().width, dstColorImage.size().height));
    dstColorImage.copyTo(roi2);
    
    cv::RNG rng(0xFFFFFFFF);
    // Draw line between nearest neighbor pairs
    for (int i = 0; i < (int)srcPoints.size(); ++i) {
        cv::Point2f pt1 = srcPoints[i];
        cv::Point2f pt2 = dstPoints[i];
        cv::Point2f from = pt1;
        cv::Point2f to   = cv::Point(srcColorImage.size().width + pt2.x, srcColorImage.size().height + pt2.y);
        //cv::Point2f to   = cv::Point(srcColorImage.size().width + pt2.x, pt2.y);
        cv::line(matchingImage, from, to, randomColor(rng), 2);
    }
    
    // show text in image
    /*cv::Point org;
     org.x = rng.uniform(matchingImage.cols/10, matchingImage.rows/10);
     org.y = rng.uniform(matchingImage.rows/10, matchingImage.rows/10);
     putText(matchingImage, "Testing text rendering", org, rng.uniform(0,8), rng.uniform(0,10)*0.05+0.1, randomColor(rng), rng.uniform(1, 10), 8);*/
    
    // Display mathing image
    cv::resize(matchingImage, matchingImage, cv::Size(matchingImage.cols/2, matchingImage.rows/2));
    //cv::resize(matchingImage, matchingImage, cv::Size(matchingImage.cols, matchingImage.rows));
    cv::imshow("Geometric Verification", matchingImage);
    cvWaitKey(0);
}

// do RASANC with OpenCV
void findInliers(std::vector<cv::KeyPoint> &qKeypoints,
                 std::vector<cv::KeyPoint> &objKeypoints,
                 std::vector<cv::DMatch> &matches,
                 const cv::Mat &srcColorImage,
                 const cv::Mat &dstColorImage)
{
    std::vector<cv::Point2f> queryCoord;
    std::vector<cv::Point2f> objectCoord;
    for( unsigned int i = 0; i < matches.size(); i++)
    {
        queryCoord.push_back((qKeypoints[matches[i].queryIdx]).pt);
        objectCoord.push_back((objKeypoints[matches[i].trainIdx]).pt);
    }
    plotMatches(srcColorImage, dstColorImage, queryCoord, objectCoord);
    
    cv::Mat mask;
    std::vector<cv::Point2f> queryInliers;
    std::vector<cv::Point2f> sceneInliers;
    cv::Mat H = findFundamentalMat(queryCoord, objectCoord, mask, CV_FM_RANSAC);
    //Mat H = findHomography( queryCoord, objectCoord, CV_RANSAC, 10, mask);
    int inliers_cnt = 0, outliers_cnt = 0;
    for (int j = 0; j < mask.rows; j++){
        if (mask.at<uchar>(j) == 1)
        {
            queryInliers.push_back(queryCoord[j]);
            sceneInliers.push_back(objectCoord[j]);
            inliers_cnt++;
            
        } else {
            outliers_cnt++;
        }
    }
    plotMatches(srcColorImage, dstColorImage, queryInliers, sceneInliers);
}


// vector of vector to cv mat
template <typename T>
cv::Mat_<T> vec2cvMat_2D(std::vector< std::vector<T> > &inVec)
{
    int rows = static_cast<int>(inVec.size());
    int cols = static_cast<int>(inVec[0].size());
    cv::Mat_<T> resmat(rows, cols);
    for (int i = 0; i < rows; i++){
        resmat.row(i) = cv::Mat(inVec[i]).t();
    }
    return resmat;
}


// vector of vector to arma mat
template <typename T>
arma::mat vec2mat(std::vector<std::vector<T>> &vec)
{
    int col = (int)vec.size();
    int row = (int)vec[0].size();
    arma::mat A(row, col, arma::fill::zeros);
    for(int i = 0; i < col; i++){
        for(int j=0; j < row; j++){
            A(j, i) = vec[i][j];
        }
    }
    return A;
}


// vector of vector to one array
template <typename T>
T * vectors2OneArray(std::vector<std::vector<T>> &descs)
{
    T * descsToOneArray = (T *)malloc(sizeof(T)*descs.size()*128);
    for(int i = 0; i < descs.size(); i++){
        for(int j = 0; j < 128; j++){
            descsToOneArray[i*128+j] = descs[i].at(j);
        }
    }
    return descsToOneArray;
}

#endif /* general_hpp */
