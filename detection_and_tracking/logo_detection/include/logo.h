#pragma once

#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <map>
#include <vector>
#include <utility>
#include <string>

#define MAX_TEMP_SIZE 1024*12*100

typedef unsigned char uint8;

namespace cvt {
    class Logo {
    public:
        Logo();
        virtual ~Logo();
        virtual int Init(const std::string logo_dat_address) = 0;
        virtual int LogoDetect(const std::string &image_address, \
                               const std::string &frame_id, std::string* logo) = 0;
    };
    class LogoFactory {
    public:
        static Logo* Create();
    };
    
}
