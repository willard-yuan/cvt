
#include "ffdecoder.h"
#include "videocapture.h"

VideoCapture::VideoCapture()
{
    decoder = shared_ptr<FFDecoder>(new FFDecoder());
}

VideoCapture::VideoCapture(const string &filename)
{
    decoder = shared_ptr<FFDecoder>(new FFDecoder());
    open(filename);
}

VideoCapture::VideoCapture(unsigned char *video_buf, int len)
{
    decoder = shared_ptr<FFDecoder>(new FFDecoder());
    open(video_buf, len);
}

bool VideoCapture::open(const string &filename)
{
    return decoder->open(filename);
}

bool VideoCapture::open(unsigned char *video_buf, int len)
{
    return decoder->open(video_buf, len);
}

void VideoCapture::close()
{
    decoder->close();
}

bool VideoCapture::isOpened()
{
    return decoder->isOpened();
}

bool VideoCapture::grab()
{
    return decoder->grab();
}

bool VideoCapture::retrieve(cv::Mat &image)
{
    return decoder->retrieve(image);
}

bool VideoCapture::read(cv::Mat &image)
{
    return decoder->read(image);
}

double VideoCapture::get(int propId) const
{
    switch(propId)
    {
    case CAP_PROP_FRAME_HEIGHT:
        return (double)decoder->height();
    case CAP_PROP_FRAME_WIDTH:
        return (double)decoder->width();
    case CAP_PROP_FRAME_COUNT:
        return (double)decoder->count();
    case CAP_PROP_FRAME_POSITION:
        return (double)decoder->position();
    case CAP_PROP_FPS:
        return (double)decoder->fps();
    default:
        printf("invalid propId");
        return 0.0;
    }
}
