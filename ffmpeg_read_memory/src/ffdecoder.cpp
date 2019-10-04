/*
 代码参考：1. https://www.jianshu.com/p/3c95b0471d3a
         2. https://github.com/wangxuewen99/VideoCapture
 */

#include "ffdecoder.h"
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;

struct buffer_data
{
    uint8_t *ptr; /* 文件中对应位置指针 */
    size_t size;  // < size left in the buffer /* 文件当前指针到末尾 */
};

// 重点，自定的buffer数据要在外面这里定义
struct buffer_data bd = {0};

FFDecoder::~FFDecoder()
{
    if (open_flag == true)
        close();
}

FFDecoder::FFDecoder()
{
#ifdef NO_LOG
    av_log_set_level(AV_LOG_QUIET);
#endif
    av_register_all();
    av_init_packet(&pkt);

    frame = NULL;
    rgbframe = NULL;
    codec_ctx = NULL;
    format_ctx = NULL;
    io_ctx = NULL;
    sws_ctx = NULL;

    open_flag = false;
}

int FFDecoder::read_buffer(void *opaque, uint8_t *buf, int buf_size)
{
    buf_size = FFMIN(buf_size, bd.size);
    //printf("ptr:%p size:%zu\n", bd->ptr, bd->size);
    /* copy internal buffer data to buf */
    
    if (!buf_size)
        return AVERROR_EOF;
    
    memcpy(buf, bd.ptr, buf_size);
    bd.ptr  += buf_size;
    bd.size -= buf_size;
    return buf_size;
}


bool FFDecoder::open(unsigned char *video_buf, int len)
{
    if( open_flag == true)
        close();

    size_t io_ctx_buffer_size = 32768;
    uint8_t *io_ctx_buffer = NULL;
    //struct buffer_data bd = { 0 };
    /* fill opaque structure used by the AVIOContext read callback */
    bd.ptr  = video_buf;
    bd.size = len;
    if (!(format_ctx = avformat_alloc_context()))
    {
        return false;
    }

    io_ctx_buffer = static_cast<uint8_t *>(av_malloc(io_ctx_buffer_size));
    if (!io_ctx_buffer)
    {
        return false;
    }
    
    io_ctx = avio_alloc_context(io_ctx_buffer, io_ctx_buffer_size,
                                0, NULL, read_buffer, NULL, NULL);
    if (!io_ctx)
    {
        return false;
    }
    format_ctx->pb = io_ctx;
    
    if (avformat_open_input(&format_ctx, NULL, NULL, NULL) < 0) {
        av_log(format_ctx, AV_LOG_ERROR, "open stream failed.\n");
        return false;
    }
    file_name = "memory";
    return init_ffmpeg();
}

bool FFDecoder::open(const string &filename)
{
    //Open video file and allocate format context
    if (avformat_open_input(&format_ctx, filename.c_str(), NULL, NULL) < 0)
    {
        av_log(format_ctx, AV_LOG_ERROR, "Could not open file.\n");
        return false;
    }
    file_name = filename;
    return init_ffmpeg();
}

bool FFDecoder::isOpened()
{
    return open_flag;
}

void FFDecoder::close()
{
    if (open_flag == false)
        return; 

    sws_freeContext(sws_ctx);

    //free the rgb image
    av_freep(&rgbframe->data[0]);
    av_frame_free(&rgbframe);

    //free the yuv frame
    av_frame_free(&frame);

    //close the codec
    avcodec_close(codec_ctx);

    //av_free(pCodecCtx);
    avcodec_free_context(&codec_ctx);

    //close the video file
    avformat_close_input(&format_ctx);

    if (io_ctx)
    {
        av_freep(&io_ctx->buffer);
        av_freep(&io_ctx);
    }

    open_flag = false;
}

bool FFDecoder::init_ffmpeg()
{
    if (avformat_find_stream_info(format_ctx, NULL) < 0)
    {
        av_log(format_ctx, AV_LOG_ERROR, "Could not find stream info.\n");
        return false;
    }

#ifndef NO_LOG
    av_dump_format(format_ctx, 0, file_name.c_str(), 0);
#endif

    video_stream = 0;
    video_stream = av_find_best_stream(format_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    if (video_stream == -1)
    {
        av_log(format_ctx, AV_LOG_ERROR, "Could not find stream index.\n");
        return false;
    }

    AVStream* stream = format_ctx->streams[video_stream];
    AVCodecParameters *codecpar = stream->codecpar;
    AVCodec* codec = avcodec_find_decoder(codecpar->codec_id);
    codec_ctx = avcodec_alloc_context3(codec);

    if (codec == NULL)
    {
        av_log(codec, AV_LOG_ERROR, "Could not find decoder.\n");
        return false;
    }
    avcodec_parameters_to_context(codec_ctx, codecpar);
    //open codec
    if (avcodec_open2(codec_ctx, codec, NULL) < 0)
    {
        av_log(codec, AV_LOG_ERROR, "Could open decoder.\n");
        return false;
    }

    //get video information
    frame_height = codecpar->height;
    frame_width = codecpar->width;
    frame_count = stream->nb_frames;
    int coded_width = stream->codec->coded_width;
    int coded_height = stream->codec->coded_height;
    

    AVPixelFormat fmt = (AVPixelFormat)codecpar->format;
    
    if (fmt < 0) {
        fmt = (AVPixelFormat)(0);
    }
    
    AVRational avg_fps = stream->avg_frame_rate;
    frame_fps = (float)avg_fps.num / avg_fps.den;
    frame_pos = 0;

    rgbframe = av_frame_alloc();
    if (!rgbframe)
    {
        av_log(rgbframe , AV_LOG_ERROR, "Could allocate frame.\n");
        return false;
    }

    av_image_alloc(rgbframe->data, rgbframe->linesize, frame_width, frame_height, AV_PIX_FMT_RGB24, 4);
    frame = av_frame_alloc();

    sws_ctx = sws_getContext(coded_width, coded_height, fmt, frame_width, frame_height,
                            AV_PIX_FMT_BGR24, 0, NULL, NULL, NULL);
    open_flag = true;
    return true;
}

bool FFDecoder::grab()
{
    int finished = 0;
    while (av_read_frame(format_ctx, &pkt) >= 0)
    {
        if(pkt.stream_index == video_stream)
        {
            
            int ret = avcodec_decode_video2(codec_ctx, frame, &finished, &pkt);
            if (ret) {
                av_packet_unref(&pkt);
                frame_pos += 1;
                break;
            }
        }
        av_packet_unref(&pkt);
    }

    return finished;
}

bool FFDecoder::retrieve(cv::Mat &image)
{
    sws_scale(sws_ctx, (const uint8_t* const*)frame->data, frame->linesize, 0, codec_ctx->coded_height,
              (uint8_t *const *)rgbframe->data, rgbframe->linesize);
    int len = frame_height * rgbframe->linesize[0];
    if(len != 3*image.total())
    {
        av_log(rgbframe, AV_LOG_ERROR, "Could not copy rgb frame to mat.\n");
        return false;
    }
    memcpy(image.data, rgbframe->data[0], len);
    return true;
}

bool FFDecoder::read(cv::Mat &image)
{
    if( !grab())
    {
        //printf("grab frame failed!");
        return false;
    }
    if( !retrieve(image))
    {
        //printf("retrieve image failed!");
        return false;
    }
    return true;
}

int FFDecoder::count() const
{
    return frame_count;
}

int FFDecoder::height() const
{
    return frame_height;
}

int FFDecoder::width() const
{
    return frame_width;
}

float FFDecoder::fps() const
{
    return frame_fps;
}

int FFDecoder::position() const
{
    return frame_pos;
}

