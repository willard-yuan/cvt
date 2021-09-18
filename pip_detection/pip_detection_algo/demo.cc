#include <iostream>
#include "include/motion_area_detection.h"
#include "protobuf_v3/src/google/protobuf/util/json_util.h"

#include <iostream>
#include <vector>
#include <fstream>
#include <string>

// convert file to memory
void load_file(const std::string &filename, std::vector<unsigned char> &buf)
{
    std::ifstream ifs(filename, std::ifstream::in);
    if(ifs.is_open())
    {
        ifs.seekg(0, std::ios::end);
        int length = ifs.tellg();
        ifs.seekg(0, std::ios::beg);
        buf.resize(length);
        ifs.read((char*)buf.data(), length);
    }
}

// read image from buffer
std::vector<char> static read_to_buffer(std::string img_path) {
    std::ifstream file(img_path);
    std::vector<char> data;
    file >> std::noskipws;
    std::copy(std::istream_iterator<char>(file), std::istream_iterator<char>(), std::back_inserter(data));
    return data;
}

int main(int argc, const char *argv[]) {
  mmu::img::MotionArea* motionLocInfo = new mmu::img::MotionArea();

  mmu::detect::Video video;
  for(int i = 0; i < 12; i++) {
    int idx = i*25;
    std::string tmpName = "./data/26368489118/MMU_26368489118_" +
        std::to_string(idx) + ".jpg";
    mmu::detect::Frame tmpFrame;
    tmpFrame.id = "test";
    std::vector<char> vec = read_to_buffer(tmpName);
    tmpFrame.image = vec;
    video.frames.push_back(tmpFrame);
  }

  mmu::detect::MotionAreaDetecion MotionAreaDetector;

  MotionAreaDetector.PipDetect(video, motionLocInfo);

  std::string result;
  ::google::protobuf::util::MessageToJsonString(*motionLocInfo, &result);
  std::cout << result << std::endl;
  motionLocInfo = nullptr;

  return 0;
}
