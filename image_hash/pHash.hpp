
#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

// Function: compute DCT of an image
static inline void imgDct(const cv::Mat& image, cv::Mat& dct) {
    cv::Mat imgGray;
    cv::cvtColor(image, imgGray, CV_BGR2GRAY);
    imgGray.convertTo(imgGray, CV_64F);
    cv::resize(imgGray, imgGray, cv::Size(32,32));
    cv::dct(imgGray, dct);
}

// Function: compute image PHash
static inline int pHash(const cv::Mat& im, uint64 &hash, int cons) {
    if (im.empty()) {
        return 0;
    }
    
    cv::Mat dct;
    imgDct(im, dct);
    
    double dIdex[64];
    double mean = 0.0;
    uint64_t hashValue = 0;
    uint32_t flag = 0;
    
    int k = 0;
    for (int i = 1+cons; i < 9+cons; ++i) {
        for (int j = 1+cons; j < 9+cons; ++j) {
            dIdex[k++] = dct.at<double>(i, j);
            mean += dct.at<double>(i, j);
        }
    }
    mean /= 64;
    
    for (int i = 0; i < 32; ++i) {
        if (dIdex[i] > mean) {
            flag = 1;
            flag <<= 31-i;
            hashValue |= flag;
        }
    }
    
    hashValue <<= 32;
    for (int i = 32; i < 64; ++i) {
        if (dIdex[i] > mean) {
            flag = 1;
            flag <<= 63 - i;
            hashValue |= flag;
        }
    }
    
    hash = hashValue;
    return 1;
}

// Function: decimal number to binary
void decimal2Binary(uint64_t number, int len)
{
    char bitset[len];
    for(uint64_t i=0; i<len; ++i)
    {
        if((number & (static_cast<uint64_t>(1) << i)) != 0)
        {
            bitset[len-i-1] = '1';
        }else{
            bitset[len-i-1] = '0';
        }
    }
    for(uint64_t i = 0; i < len; ++i)
    {
        cout << bitset[i];
    }
}

// Function: fast compute hamming distance
static inline int distanceHamm(uint64 h1, uint64 h2) {
    uint64 h = h1^h2;
    int hammdis = 0;
    while(h) {
        h &= h-1;
        hammdis++;
    }
    return hammdis;
}

// Function: detect pure image, used to solve badcase of pure image using PHash
static int pureDet(const cv::Mat &imgBGR) {
    if (imgBGR.empty()) {
        return 0;
    }

    cv::Mat imGray;
    cv::cvtColor(imgBGR, imGray, CV_BGR2GRAY);
    int histSize = 256;
    float range[] = {0, 255} ;
    const float* histRange = {range};
    cv::Mat hist;
    cv::calcHist(&imGray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);

    cv::Mat idx;
    cv::sortIdx(hist, idx, CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING);
    cv::sort(hist, hist, CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING);

    float maxFre = hist.at<float>(0,0);
    float secFre = hist.at<float>(0,1);
    float allFre = cv::sum(hist)[0];
    float ratio1 = (maxFre + secFre)/allFre;

    if (ratio1 >= 0.51) return 1;
    return 0;
}
