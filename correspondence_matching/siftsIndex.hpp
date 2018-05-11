
#ifndef siftsIndex_hpp
#define siftsIndex_hpp

#include <map>
#include <iostream>
#include <fstream>
#include <queue>
#include <chrono>
#include <vector>
#include <utility>
#include <string>
#include <unordered_set>

#include "hnswlib.h"
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

typedef unsigned char uint8;

struct geoInfo{
    std::string fileBaseName;
    int ID;
    cv::KeyPoint Pt;
    int relateID;
};


struct rankInfo{
    std::map<std::string, int> numMatched;
    std::map<std::string, std::vector<std::pair<int, int>>> matchedPair;
};


class siftsIDX {
    
public:
    siftsIDX(const std::string& geoInfoPath, const std::string& indexInfoPath) {
        this->InitModel(geoInfoPath, indexInfoPath);
    }
    
    ~siftsIDX() {
        if (appr_alg) delete appr_alg;
    }
    
    void rootSift(cv::Mat &descriptors, const float eps = 1e-7);
    
    hnswlib::HierarchicalNSW<float> *appr_alg;
    
    std::vector<geoInfo> geosInfo;
    
private:
    int vecdim = 128;
    hnswlib::InnerProductSpace *l2space;
    void InitModel(const std::string& geoInfoPath, const std::string& indexInfoPath);
    void readMatBinary(std::ifstream& ifs, cv::Mat& in_mat);
    void readKeyPointBinary(std::ifstream& ifs, cv::KeyPoint& key_point);
    void readKeyPointsBinary(std::ifstream& ifs, std::vector<cv::KeyPoint>& key_point);
    bool existsFile(const std::string &name);
    
    cv::Ptr<cv::Feature2D> detector;
    
    
};


#endif /* siftsIndex_hpp */
