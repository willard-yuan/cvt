#ifndef pupil_tracker_h
#define pupil_tracker_h

#include <vector>
#include <string>

#include <opencv2/core/core.hpp>

#include "conic_section.h"

namespace pupiltracker {
    
    struct TrackerParams
    {
        int Radius_Min;
        int Radius_Max;
        
        double CannyBlur;
        double CannyThreshold1;
        double CannyThreshold2;
        int StarburstPoints;
        
        int PercentageInliers;
        int InlierIterations;
        bool ImageAwareSupport;
        int EarlyTerminationPercentage;
        bool EarlyRejection;
        int Seed;
    };
    
    const cv::Point2f UNKNOWN_POSITION = cv::Point2f(-1,-1);
    
    struct EdgePoint
    {
        cv::Point2f point;
        double edgeStrength;
        
        EdgePoint(const cv::Point2f& p, double s) : point(p), edgeStrength(s) {}
        EdgePoint(float x, float y, double s) : point(x,y), edgeStrength(s) {}
        
        bool operator== (const EdgePoint& other)
        {
            return point == other.point;
        }
    };
    
    struct findPupilEllipse_out {
        cv::Rect roiHaarPupil;
        cv::Mat_<uchar> mHaarPupil;
        
        cv::Mat_<float> histPupil;
        double threshold;
        cv::Mat_<uchar> mPupilThresh;
        
        cv::Rect bbPupilThresh;
        cv::RotatedRect elPupilThresh;
        
        cv::Rect roiPupil;
        cv::Mat_<uchar> mPupil;
        cv::Mat_<uchar> mPupilOpened;
        cv::Mat_<uchar> mPupilBlurred;
        cv::Mat_<uchar> mPupilEdges;
        cv::Mat_<float> mPupilSobelX;
        cv::Mat_<float> mPupilSobelY;
        
        std::vector<EdgePoint> edgePoints;
        std::vector<cv::Point2f> inliers;
        int ransacIterations;
        int earlyRejections;
        bool earlyTermination;
        
        cv::Point2f pPupil;
        cv::RotatedRect elPupil;
        
        findPupilEllipse_out() : pPupil(UNKNOWN_POSITION), threshold(-1) {}
    };

    bool findPupilEllipse(
                          const TrackerParams& params,
                          const cv::Mat& m,
                          findPupilEllipse_out& out,
                          bool eraseSpecular
                          );
    
} //namespace pupil_tracker


#endif /* pupil_tracker_hpp */
