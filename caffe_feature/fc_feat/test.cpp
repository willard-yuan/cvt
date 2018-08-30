
#include <sys/time.h>
#include <string>
#include <vector>
#include <stdio.h>
#include <math.h>

#include "boost/make_shared.hpp"
#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"

#include "caffe/caffe.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using std::string;
namespace db = caffe::db;

// Function: Convert a raw image Mat to float Mat
// Note: img is a raw CV:Mat
void raw_mat_to_float_mat(cv::Mat &img, int H_resized, int W_resized, int num_channels){
    // Convert raw Mat to BGR format
    if (img.channels() == 3 && num_channels == 1)
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && num_channels == 1)
        cv::cvtColor(img, img, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels == 3)
        cv::cvtColor(img, img, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && num_channels == 3)
        cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
    else
        img = img;

    // Resizing
    //cv::resize(img, img, cv::Size(H_resized, W_resized));
    // Float conversion
    if (num_channels == 3)
        img.convertTo(img, CV_32FC3);
    else
        img.convertTo(img, CV_32FC1);
}

// Function: Assign single img (BGR format) data to mutable cup data
// Note: i bigin from 0 
void assign_img_to_mutable_cpu_data(float* input_data, const cv::Mat &sample_float, const cv::Mat &mean_pixes_mat, int ith){
    cv::Mat img_substract_mean = sample_float - mean_pixes_mat;
    // Splitting each channel is essential, since we must assign all the B channel data to
    // mutable cpu data firstly. We can't use:
    // float* BGR_array = (float*)img_substract_mean.data
    // https://tolleybot.wordpress.com/2011/05/14/accessing-an-opencv-mat-data-using-c/
    std::vector<cv::Mat> input_channels(3);
    split(img_substract_mean, input_channels);
    float* B_array = (float*)input_channels[0].data;
    float* G_array = (float*)input_channels[1].data;
    float* R_array = (float*)input_channels[2].data;

    int Length = img_substract_mean.rows*img_substract_mean.cols;

    // Assign image data to the mutable cpu data
    memcpy( input_data + (3*ith+0)*Length, B_array, sizeof(B_array[0])*Length );
    memcpy( input_data + (3*ith+1)*Length, G_array, sizeof(B_array[0])*Length );
    memcpy( input_data + (3*ith+2)*Length, R_array, sizeof(B_array[0])*Length );
    printf("input number channels: %d\n", (int)input_channels.size());
}

// Class: Constract a Caffe feature extractor
class CaffeFeatExtractor{
    string caffemodel_file;
    string prototxt_file;
    string blob_name;
    int batch_size;
    int H_resized;
    int W_resized;
    cv::Mat imagenet_mean_mat;
    caffe::shared_ptr<Net<float> > feature_extraction_net;
    caffe::Blob<float>* input_layer;
    bool gpu_mode;
    int device_id;
    public:
        CaffeFeatExtractor(string _caffemodel_file, string _prototxt_file, string _blob_name, int _H_resized, int _W_resized, int _batch_size, string _compute_mode, int _device_id);
        bool extract_batch_single_feat(std::vector<cv::Mat> &imgs, std::vector<std::vector<float> > &feats);
        cv::Mat imagenet_mean_generation(int H_resized, int W_resized);
};

CaffeFeatExtractor::CaffeFeatExtractor(string _caffemodel_file, string _prototxt_file, string _blob_name, int _H_resized, int _W_resized, int _batch_size, string _compute_mode, int _device_id){
    // Setup the GPU or the CPU mode for Caffe
    if (strcmp(_compute_mode.c_str(), "GPU") == 0 || strcmp(_compute_mode.c_str(), "gpu") == 0) {
        printf("CaffeFeatExtractor::CaffeFeatExtractor(): using GPU......\n");
        gpu_mode = true;
        device_id = _device_id;
        Caffe::CheckDevice(device_id);
        printf("CaffeFeatExtractor::CaffeFeatExtractor(): using device_id = %d\n", device_id);
        Caffe::set_mode(Caffe::GPU);
        Caffe::SetDevice(device_id);
        // Optional: to check that the GPU is working properly...
        Caffe::DeviceQuery();
    }else{
        printf("CaffeFeatExtractor::CaffeFeatExtractor(): using CPU......\n");
        gpu_mode = false;
        device_id = -1;
        Caffe::set_mode(Caffe::CPU);
    }
    // Assign specified .caffemodel and .prototxt files
    caffemodel_file = _caffemodel_file;
    prototxt_file = _prototxt_file;
    blob_name = _blob_name;
    H_resized = _H_resized;
    W_resized = _W_resized;
    batch_size = _batch_size;
    // Generation imageNet mean mat
    imagenet_mean_mat = imagenet_mean_generation(H_resized, W_resized);
    // Network creation using the specified .prototxt
    feature_extraction_net = boost::make_shared<Net<float> >(prototxt_file, caffe::TEST);
    feature_extraction_net->CopyTrainedLayersFrom(caffemodel_file);
    // Input layer
    input_layer = feature_extraction_net->input_blobs()[0];
    input_layer->Reshape(batch_size, input_layer->channels(), H_resized, W_resized);
    // Forward dimension change to all layers
    feature_extraction_net->Reshape();
}

// Function: Generate ImageNet 2012 mean CV:Mat
// Note: The cv:Mat format is ordered in BGR
// B: 104.00698793 G: 116.66876762 R: 122.67891434
cv::Mat CaffeFeatExtractor::imagenet_mean_generation(int H_resized, int W_resized){
     std::vector<cv::Mat> pixes_mat;
     Mat B_Mat(H_resized, W_resized, CV_32FC1, Scalar(103.72267235));
     Mat G_Mat(H_resized, W_resized, CV_32FC1, Scalar(116.14597303));
     Mat R_Mat(H_resized, W_resized, CV_32FC1, Scalar(122.05719166));
     pixes_mat.push_back(B_Mat);
     pixes_mat.push_back(G_Mat);
     pixes_mat.push_back(R_Mat);
     cv::Mat mean_pixes_mat;
     cv::merge(pixes_mat, mean_pixes_mat);
     return mean_pixes_mat;
}

// Function: Extract features of one batch size 
bool CaffeFeatExtractor::extract_batch_single_feat(std::vector<cv::Mat> &imgs, std::vector<std::vector<float> > &feats){

    // Set the GPU/CPU mode for Caffe (here in order to be thread-safe)
    if (gpu_mode){
        Caffe::set_mode(Caffe::GPU);
        Caffe::SetDevice(device_id);
    }else{
        Caffe::set_mode(Caffe::CPU);
    }
    
    // Input data, used to store images data
    float * input_data = input_layer->mutable_cpu_data();

    // Reprocessing image
    for(int i = 0; i < imgs.size(); i++){
        // Convert the input image to the input image format of the network
        // BGR channel transform, resizing and float convertion
        raw_mat_to_float_mat(imgs[i], 0, 0, input_layer->channels());  
        assign_img_to_mutable_cpu_data(input_data, imgs[i], imagenet_mean_mat, i);
    }

    // Actual feature computation
    feature_extraction_net->Forward();

    // Select pool5 as the feature
    const boost::shared_ptr<Blob<float> > feature_blob = feature_extraction_net->blob_by_name(blob_name);

    // Obtain the feature
    const float* feature_blob_data  = feature_blob->cpu_data();

    int dim_feats = feature_blob->count();
    int map_height = feature_blob->height();
    int map_width = feature_blob->width();
    int blob_num_channels = feature_blob->channels();
    printf("channels: %d, width: %d, hight: %d, batch_size: %d, dim_features: %d\n", feature_blob->channels(), feature_blob->width(), feature_blob->height(), feature_blob->num(), dim_feats);
    int dim_per_feat = dim_feats / feature_blob->num();

    printf("dim_per_feat: %d\n", dim_per_feat);

    float * tmp_blob_data = new float[dim_per_feat];

    for(int i = 0; i < imgs.size(); i++){
        memcpy(tmp_blob_data, feature_blob_data + i*dim_per_feat, sizeof(feature_blob_data[0])*dim_per_feat);
        std::vector<float> feat{tmp_blob_data, tmp_blob_data + dim_per_feat};

        feats.push_back(feat);
    }
    
    delete [] tmp_blob_data;
    return true;
}

int main(int argc, char** argv) { 

    std::string image_path("/home/yuanyong/codes/cpt_feat/build/7833393995_2.jpg");
    cv::Mat img = cv::imread(image_path, -1);
    cv::Mat img_resized;
    cv::resize(img, img_resized, cv::Size(299, 299));
    img = img_resized;

    // Construct  multi-images
    std::vector<cv::Mat> imgs;
    imgs.push_back(img);
    imgs.push_back(img); // two images

    int num_imgs = imgs.size();
    int img_width = img.size().width;
    int img_height = img.size().height;

    // Net creation
    string layer = "fc1";
    int batch_size = 2;
    string compute_mode = "GPU";
    int device_id = 4;
    std::string pretrained_binary_proto("../model/iter_60000.caffemodel");
    std::string feature_extraction_proto("../model/deploy.prototxt");
    CaffeFeatExtractor CFEtor(pretrained_binary_proto, feature_extraction_proto, layer, img_height, img_width, batch_size, compute_mode, device_id);

    struct timeval t1, t2;
    gettimeofday(&t1, NULL);
    std::vector<std::vector<float> > feats;
    CFEtor.extract_batch_single_feat(imgs, feats); 
    gettimeofday(&t2, NULL);
    std::cout << "time: " << (t2.tv_sec - t1.tv_sec)*1000 + (t2.tv_usec - t1.tv_usec) / 1000 << std::endl;

    printf("image size: H: %d, W: %d, %d\n", img_height, img_width, num_imgs);

    for(vector<float>::iterator it = feats[1].begin(); it != feats[1].end(); ++it){
        printf("%f ", *it);
    }
   
    return 1; 
}
