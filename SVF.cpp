
#include "SVF.hpp"

SVF::SVF(){}

int SVF::spaceValidate(const cv::KeyPoint &pa0, const cv::KeyPoint &pa1, const cv::KeyPoint &pb0, const cv::KeyPoint &pb1) {
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

int SVF::getInliers(std::vector<cv::KeyPoint> &qKpts1, std::vector<cv::KeyPoint> &qKpts2, std::vector<cv::DMatch> &rawMatches)
{
    SVFMatches.clear();
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
        SVFMatches.push_back(rawMatches[maxid]);
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
    
    return (int)SVFMatches.size();
}


int SVF::OctaveDiff(float octave1, float octave2, int* idx)
{
    float octave_diff = octave1 - octave2;
    while(octave_diff < 0.0) octave_diff += 8;
    while(octave_diff >= 8) octave_diff -= 8;
    *idx = int(octave_diff + 0.5) / kMaxOctaveSlices;
    *idx = (*idx < 0) ? 0 : *idx;
    *idx = (*idx >= kMaxOctaveSlices) ? (kMaxOctaveSlices-1) : *idx;
    return 0;
}

int SVF::OctaveValidate(std::vector<cv::KeyPoint>& kpts1, std::vector<cv::KeyPoint>& kpts2, std::vector<cv::DMatch>* good_matches)
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
std::vector<cv::DMatch> SVF::getSVFMatches()
{
    return this->SVFMatches;
}


