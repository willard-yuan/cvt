
#include "../include/common.hpp"
#include "../include/GHT.hpp"

GHT::GHT() {}

int GHT::spaceValidate(const cv::KeyPoint &pa0, const cv::KeyPoint &pa1, const cv::KeyPoint &pb0, const cv::KeyPoint &pb1) {
    // A图像两个关键点的角度差
    float angleA = pa0.angle - pa1.angle;
    // B图像
    float angleB = pb0.angle - pb1.angle;
    
    // 两角度差值的绝对值
    float diff1 = std::abs(angleA - angleB);
    
    float thetaA;
    float deltaX_A = pa1.pt.x - pa0.pt.x;
    float deltaY_A = pa1.pt.y - pa0.pt.y;
    if (deltaX_A == 0) {
        if (deltaY_A >= 0) {
            thetaA = 90;
        } else {
            thetaA = 270;
        }
    } else if (deltaX_A > 0) {
        float tanv = deltaY_A/deltaX_A;
        thetaA = atan(tanv)*180/PI;
    } else {
        float tanv = deltaY_A/deltaX_A;
        if (deltaY_A >= 0) {
            thetaA = atan(tanv)*180/PI + 180;
        } else {
            thetaA = atan(tanv)*180/PI - 180;
        }
    }
    thetaA -= pa0.angle;
    
    float thetaB;
    float deltaX_B = pb1.pt.x - pb0.pt.x;
    float deltaY_B = pb1.pt.y - pb0.pt.y;
    if (deltaX_B == 0)
    {
        if (deltaY_B >= 0)
        {
            thetaB = 90;
        } else {
            thetaB = 270;
        }
    } else if (deltaX_B > 0) {
        float tanv = deltaY_B/deltaX_B;
        thetaB = atan(tanv)*180/PI;
    } else {
        float tanv = deltaY_B/deltaX_B;
        if (deltaY_B >= 0) {
            thetaB = atan(tanv)*180/PI + 180;
        } else {
            thetaB = atan(tanv)*180/PI - 180;
        }
    }
    
    thetaB -= pb0.angle;
    float diff2 = std::abs(thetaA - thetaB);
    if (diff1 < 10 && diff2 < 10)
        return 1;
    return 0;
}

int GHT::getInliers(std::vector<cv::KeyPoint> &qKpts1, std::vector<cv::KeyPoint> &qKpts2, std::vector<cv::DMatch> &rawMatches)
{
    SVFMatches.clear();
    std::vector<cv::DMatch> goodMatches;
    int numRawMatch = (int)rawMatches.size();
    int **brother_matrix = new int*[numRawMatch];
    for (int i = 0; i < numRawMatch; i++) {
        brother_matrix[i] = new int[numRawMatch];
        brother_matrix[i][i] = 0;
    }
    
    for (int i = 0; i < numRawMatch; i++) {
        int queryIdx0 = rawMatches[i].queryIdx;
        int trainIdx0 = rawMatches[i].trainIdx;
        for (int j = i+1; j < numRawMatch; j++) {
            int queryIdx1 = rawMatches[j].queryIdx;
            int trainIdx1 = rawMatches[j].trainIdx;
            auto lp0 = qKpts1[queryIdx0];
            auto rp0 = qKpts1[queryIdx1];
            auto lp1 = qKpts2[trainIdx0];
            auto rp1 = qKpts2[trainIdx1];
            brother_matrix[i][j] = spaceValidate(lp0, rp0, lp1, rp1);
            brother_matrix[j][i] = brother_matrix[i][j];
        }
    }
    
    int map_size = numRawMatch;
    int *mapId = new int[numRawMatch];
    for (int i = 0; i < numRawMatch; i++) mapId[i] = i;
    while (true)
    {
        int maxv = -1;
        int maxid = 0;
        for (int i = 0; i < map_size; i++)
        {
            int sum = 0;
            for (int j = 0; j < map_size; j++) sum += brother_matrix[mapId[i]][mapId[j]];
            if (sum > maxv)
            {
                maxv = sum;
                maxid = mapId[i];
            }
        }
        if (maxv == 0) break;
        goodMatches.push_back(rawMatches[maxid]);
        int id = 0;
        for (int i = 0; i < map_size; i++)
        {
            if (brother_matrix[maxid][mapId[i]] > 0) mapId[id++] = mapId[i];
        }
        map_size = maxv;
    }
    
    // free memory
    delete [] mapId;
    for(int i=0; i < numRawMatch; i++)
    {
        delete [] brother_matrix[i];
    }
    delete [] brother_matrix;
    
    if(goodMatches.size() <= 0) return 0;
    std::sort(goodMatches.begin(), goodMatches.end(), idxdist);
    SVFMatches.push_back(goodMatches[0]);
    int base = goodMatches[0].trainIdx;
    for(int i = 1; i < goodMatches.size(); ++i)
    {
        if(base != goodMatches[i].trainIdx)
        {
            base = goodMatches[i].trainIdx;
            SVFMatches.push_back(goodMatches[i]);
        }
    }

    
    return (int)SVFMatches.size();
}

int GHT::OctaveDiff(float octave1, float octave2, int* idx)
{
    float octave_diff = octave1 - octave2;
    while(octave_diff < 0.0) octave_diff += 8;
    while(octave_diff >= 8) octave_diff -= 8;
    *idx = int(octave_diff + 0.5) / kMaxOctaveSlices;
    *idx = (*idx < 0) ? 0 : *idx;
    *idx = (*idx >= kMaxOctaveSlices) ? (kMaxOctaveSlices-1) : *idx;
    return 0;
}

int GHT::OctaveValidate(std::vector<cv::KeyPoint>& kpts1, std::vector<cv::KeyPoint>& kpts2, std::vector<cv::DMatch>* good_matches)
{
    std::vector<int32_t> octave_diff;
    octave_diff.resize(kMaxOctaveSlices);
    std::fill(octave_diff.begin(), octave_diff.end(), 0);
    int octave_diff_idx = -1;
    for(auto i = good_matches->begin(); i != good_matches->end(); i++) {
        OctaveDiff(kpts1[i->queryIdx].octave, kpts2[i->trainIdx].octave, &octave_diff_idx);
        octave_diff[octave_diff_idx] += 1;
    }
    int max_idx = (int)std::distance(octave_diff.begin(), std::max_element(octave_diff.begin(), octave_diff.end()));
    std::cout << "CheckOctaveDiff max_idx: " << max_idx << std::endl;
    auto i = good_matches->begin();
    do {
        OctaveDiff(kpts1[i->queryIdx].octave, kpts2[i->trainIdx].octave, &octave_diff_idx);
        if(abs(octave_diff_idx - max_idx) > 1) {
            i = good_matches->erase(i);
        }else {
            i++;
        }
    }while(i != good_matches->end());
    return 0;
}

// this function should be called after getInliers
std::vector<cv::DMatch> GHT::getSVFMatches()
{
    return this->SVFMatches;
}


cv::Scalar GHT::randomColor(cv::RNG& rng)
{
    int icolor = (unsigned)rng;
    
    return cv::Scalar(icolor&0xFF, (icolor>>8)&0xFF, (icolor>>16)&0xFF);
}

void GHT::plotMatches(const cv::Mat &srcColorImage, const cv::Mat &dstColorImage,
                 std::vector<cv::Point2f> &srcPoints, std::vector<cv::Point2f> &dstPoints)
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
    
    // 在图像中显示匹配点数文本
    /*Point org;
     org.x = rng.uniform(matchingImage.cols/10, matchingImage.rows/10);
     org.y = rng.uniform(matchingImage.rows/10, matchingImage.rows/10);
     putText(matchingImage, "Testing text rendering", org, rng.uniform(0,8), rng.uniform(0,10)*0.05+0.1, randomColor(rng), rng.uniform(1, 10), 8);*/
    
    // Display mathing image
    cv::resize(matchingImage, matchingImage, cv::Size(matchingImage.cols/4, matchingImage.rows/4));
    //cv::resize(matchingImage, matchingImage, cv::Size(matchingImage.cols, matchingImage.rows));
    cv::imshow("Geometric Verification", matchingImage);
    cv::waitKey();
}

/******************************************************************
 * 函数功能：使用OpenCV自带的RANSAC寻找内点
 */
void GHT::getRansacInliers(std::vector<cv::KeyPoint> &qKeypoints, std::vector<cv::KeyPoint> &objKeypoints, std::vector<cv::DMatch> &matches, const cv::Mat &srcColorImage, const cv::Mat &dstColorImage)
{
    // 获取关键点坐标
    std::vector<cv::Point2f> queryCoord;
    std::vector<cv::Point2f> objectCoord;
    for( unsigned int i = 0; i < matches.size(); i++){
        queryCoord.push_back((qKeypoints[matches[i].queryIdx]).pt);
        objectCoord.push_back((objKeypoints[matches[i].trainIdx]).pt);
    }
    // 使用自定义的函数显示匹配点对
    //plotMatches(srcColorImage, dstColorImage, queryCoord, objectCoord);
    
    // 计算homography矩阵
    cv::Mat mask;
    std::vector<cv::Point2f> queryInliers;
    std::vector<cv::Point2f> sceneInliers;
    //cv::Mat H = findFundamentalMat(queryCoord, objectCoord, mask, cv::FM_RANSAC);
    cv::Mat H = findHomography( queryCoord, objectCoord, cv::FM_RANSAC, 10, mask);
    int inliers_cnt = 0, outliers_cnt = 0;
    for (int j = 0; j < mask.rows; j++){
        if (mask.at<uchar>(j) == 1){
            queryInliers.push_back(queryCoord[j]);
            sceneInliers.push_back(objectCoord[j]);
            inliers_cnt++;
        }else {
            outliers_cnt++;
        }
    }
    //显示剔除误配点对后的匹配点对
    plotMatches(srcColorImage, dstColorImage, queryInliers, sceneInliers);
}


