
#include "../include/opencvUtils.hpp"
#include "../include/tfExtractFeature.hpp"

int CnnFeature::nomalizeVector(std::vector<float> &v, const int feature_dim)
{
    if (v.size() !=feature_dim) return 0;
    std::vector<float> v_norm;
    float norm_v = sqrt(std::inner_product( v.begin(), v.end(), v.begin(), 0.0 ));
    float denorm_v = std::max(1e-12, (double)norm_v);
    for (auto it = v.begin(); it != v.end(); it++)
    {
        *it = (*it)/denorm_v;
    }
    return 1;
}

int CnnFeature::initModel(const std::string& modelBin)
{
    // 读取模型文件
    if (!ReadBinaryProto(tensorflow::Env::Default(), modelBin, &graph_def).ok())
    {
        std::cout << "Read model .pb failed" << std::endl;
        return -1;
    }

    //sess_opt.config.mutable_gpu_options()->set_allow_growth(true);
    (&session)->reset(NewSession(sess_opt));
    if (!session->Create(graph_def).ok())
    {
        cout << "Create graph failed" << endl;
        return -1;
    }
    
    return 1;
}

int CnnFeature::computeFeat(const cv::Mat& img, std::vector<float> &ft)
{
    int tmpBatchSize = 1;
    
    if (img.empty()) return 0;
    
    cv::Mat imgResized(netInputSize, netInputSize, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::resize(img, imgResized, cv::Size(netInputSize, netInputSize));
    
    inputTensor = tensorflow::Tensor(tensorflow::DT_FLOAT,
                                     tensorflow::TensorShape({tmpBatchSize, netInputSize, netInputSize, 3}));
    auto inputTensorMapped = inputTensor.tensor<float, 4>();
    
    for (int y = 0; y < imgResized.rows; ++y)
    {
        for (int x = 0; x < imgResized.cols; ++x)
        {
            cv::Vec3b color = imgResized.at<cv::Vec3b>(cv::Point(x, y));
            inputTensorMapped(0, y, x, 0) = (float)color[2];
            inputTensorMapped(0, y, x, 1) = (float)color[1];
            inputTensorMapped(0, y, x, 2) = (float)color[0];
        }
    }
    
    std::vector<tensorflow::Tensor> outputs;
    std::pair<std::string, tensorflow::Tensor> imgPair(inPutName, inputTensor);
    
    tensorflow::Status status = session->Run({imgPair}, {outPutName}, {}, &outputs); //Run, 得到运行结果，存到outputs中
    if (!status.ok())
    {
        cout << "Running model failed"<<endl;
        cout << status.ToString() << endl;
        return -1;
    }
    
    // 得到模型运行结果
    tensorflow::Tensor t = outputs[0];
    auto tmap = t.tensor<float, 2>();
    int output_dim = (int)t.shape().dim_size(1);
    
    // 保存特征
    ft.clear();
    for (int n = 0; n < output_dim; n++)
    {
        ft.push_back(tmap(0, n));
    }
    
    // 特征归一化
    nomalizeVector(ft, (int)ft.size());
    
    return 1;
}

int CnnFeature::computeFeatsBatch(const std::vector<cv::Mat> &imgs, std::vector<std::vector<float>> &fts)
{
    if (imgs.empty()) return 0;
    
    fts.clear();
    
    int nBatchs = (int)imgs.size() / batchSize;
    int residNum = (int)imgs.size() % batchSize;
    
    inputTensor = tensorflow::Tensor(tensorflow::DT_FLOAT,
                                     tensorflow::TensorShape({batchSize, netInputSize, netInputSize, 3}));
    auto inputTensorMapped = inputTensor.tensor<float, 4>();
    std::vector<float> ft;

    
    // n 个batch
    for (int i = 0; i < nBatchs; i++)
    {
        cv::Mat imgResized;
        for (int j = 0; j < batchSize; j++)
        {
            if (imgs.at(i*batchSize+j).empty())
            {
                std::cout << "image is empty" << std::endl;
                imgResized = cv::Mat::zeros(netInputSize, netInputSize, CV_8UC3);
            } else {
                cv::resize(imgs.at(i*batchSize+j), imgResized, cv::Size(netInputSize, netInputSize));
            }
            
            // 填充数据到tensor里面
            for (int y = 0; y < imgResized.rows; ++y)
            {
                for (int x = 0; x < imgResized.cols; ++x)
                {
                    cv::Vec3b color = imgResized.at<cv::Vec3b>(cv::Point(x, y));
                    inputTensorMapped(j, y, x, 0) = (float)color[2];
                    inputTensorMapped(j, y, x, 1) = (float)color[1];
                    inputTensorMapped(j, y, x, 2) = (float)color[0];
                }
            }
        }

        
        std::vector<tensorflow::Tensor> outputs;
        std::pair<std::string, tensorflow::Tensor> imgPair(inPutName, inputTensor);
        
        tensorflow::Status status = session->Run({imgPair}, {outPutName}, {}, &outputs); //Run, 得到运行结果，存到outputs中
        if (!status.ok())
        {
            cout << "Running model failed: " << status.ToString() <<endl;
            return -1;
        }
        
        // 得到模型运行结果
        tensorflow::Tensor t = outputs[0];
        auto tmap = t.tensor<float, 2>();
        int output_dim = (int)t.shape().dim_size(1);
        
        // 保存特征
        std::vector<float> feat;
        for (int k = 0; k < batchSize; k++)
        {
            ft.clear();
            for (int n = 0; n < output_dim; n++)
            {
                ft.push_back(tmap(k, n));
            }
            
            // 特征归一化
            nomalizeVector(ft, (int)ft.size());
            
            fts.push_back(ft);
        }
    }
    
    // 剩余的图片
    for (int i = 0; i < residNum; i++)
    {
        ft.clear();
        cv::Mat tmpImg = imgs.at(nBatchs*batchSize + i);
        if (tmpImg.empty())
        {
            std::cout << "image is empty" << std::endl;
            tmpImg = cv::Mat::zeros(netInputSize, netInputSize, CV_8UC3);
        }
        if (computeFeat(tmpImg, ft))
        {
            fts.push_back(ft);
        }
    }
    
    return 1;
}

