
#ifndef covdetExtractor_hpp
#define covdetExtractor_hpp

#include "utils.h"

class siftDesctor{
public:
    siftDesctor(){};
    void covdet_keypoints_and_descriptors(std::string &imageName, std::vector<std::vector<float>> &frames, std::vector<std::vector<float>> &desctor);
    
    std::string imageName;
    std::vector<std::vector<float>> frame;
    std::vector<std::vector<float>> desc;
    
private:
    void flip_descriptor(std::vector<float> &dst, float *src);
};

#endif /* covdetExtractor_hpp */
