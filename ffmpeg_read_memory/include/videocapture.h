
#ifndef _DECODER_H_
#define _DECODER_H_

#include <string>
#include <opencv2/core/core.hpp>
#include <memory>

using namespace std;

class FFDecoder;

class VideoCapture
{
public:
    enum VideoCaptureProperties
    {
        CAP_PROP_FRAME_POSITION =0, //!< 0-based index of the frame to be decoded/captured next.
        CAP_PROP_FRAME_WIDTH    =1, //!< Width of the frames in the video stream.
        CAP_PROP_FRAME_HEIGHT   =2, //!< Height of the frames in the video stream.
        CAP_PROP_FPS            =3, //!< Frame rate.
        CAP_PROP_FRAME_COUNT    =4, //!< Number of frames in the video file.
    };

    VideoCapture();
    VideoCapture(const string &filename);
    VideoCapture(unsigned char* video_buf, int len);

    bool open(const string &filename);
    bool open(unsigned char* video_buf, int len);
    void close();
    bool isOpened();
    bool grab();
    bool retrieve(cv::Mat &image);
    bool read(cv::Mat &image);
    double get(int propId) const;

private:
    shared_ptr<FFDecoder> decoder;
};

#endif //_DECODER_H_

