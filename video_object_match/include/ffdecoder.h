
#ifndef _FFDECODER_H_
#define _FFDECODER_H_

#include <string>
#include <opencv2/core/core.hpp>

extern "C"
{
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libavformat/avio.h"
#include "libswscale/swscale.h"
#include "libavutil/imgutils.h"
};

using namespace std;
class FFDecoder
{
public:
    FFDecoder();
    ~FFDecoder();
    bool open(const string &filename);
    bool open(unsigned char* video_buf, int len);
    void close();
    bool isOpened();
    bool grab();
    bool retrieve(cv::Mat &image);
    bool read(cv::Mat &image);
    int height() const;
    int width() const;
    int count() const;
    float fps() const;
    int  position() const;

private:
    AVIOContext *io_ctx;
    AVFrame *frame;
    AVFrame *rgbframe;
    AVFormatContext *format_ctx;
    AVCodecContext  *codec_ctx;
    struct SwsContext *sws_ctx;
    uint8_t *buffer;
    AVPacket pkt;

    string file_name;
    int video_stream;
    int frame_height;
    int frame_width;
    int frame_count;
    float frame_fps;
    int frame_pos;
    bool open_flag;

    bool init_ffmpeg();
    static int read_buffer(void *opaque, uint8_t *buf, int buf_size);
};

#endif //_FFDECODER_H_
