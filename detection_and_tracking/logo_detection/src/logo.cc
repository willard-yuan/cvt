#include <iostream>
#include <algorithm>
#include <map>
#include <vector>
#include <utility>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

#include "../include/logo.h"

namespace cvt {

    const double PI = 3.1415926;

    cv::Point2f RotatePoint(const cv::Point2f& p, float rad)
    {
        const float x = std::cos(rad) * p.x - std::sin(rad) * p.y;
        const float y = std::sin(rad) * p.x + std::cos(rad) * p.y;

        const cv::Point2f rot_p(x, y);
        return rot_p;
    }

    cv::Point2f RotatePoint(const cv::Point2f& cen_pt, const cv::Point2f& p, float rad)
    {
        const cv::Point2f trans_pt = p - cen_pt;
        const cv::Point2f rot_pt   = RotatePoint(trans_pt, rad);
        const cv::Point2f fin_pt   = rot_pt;

        return fin_pt;
    }

    void rotate2D(const cv::Mat & src, cv::Mat & dst, const double degrees)
    {
        cv::Point2f center(src.cols/2.0, src.rows/2.0);
        cv::Mat rot = cv::getRotationMatrix2D(center, degrees, 1.0);
        cv::Rect bbox = cv::RotatedRect(center,src.size(), degrees).boundingRect();

        rot.at<double>(0,2) += bbox.width/2.0 - center.x;
        rot.at<double>(1,2) += bbox.height/2.0 - center.y;

        cv::warpAffine(src, dst, rot, bbox.size());
    }

    Logo::Logo() {}

    Logo::~Logo() {}

    class LogoWorker : public Logo {
    public:
        LogoWorker();
        virtual ~LogoWorker();
        virtual int Init(const std::string logo_dat_address);
        virtual int LogoDetect(const std::string &image_address, const std::string &frame_id, std::string* logo);
    private:
        std::string detect(cv::Mat &frame, const std::string frame_id);
        std::vector<cv::DMatch> spatialVerify(std::vector<cv::KeyPoint> &qKpts1, std::vector<cv::KeyPoint> &qKpts2, std::vector<cv::DMatch> &rawMatches);
        bool isContinue(const std::string &logo_temp_name, int vali_num, float hog_dis);
        int loadLogoDat(const std::string logo_dat_address);
        int templateVerify(cv::Mat &img, cv::Mat &img_descs, std::vector<cv::KeyPoint> &img_kpts, cv::flann::Index &img_sift_index, const std::string frame_id, int ith, bool &rotate_flag);

        void matchInfoShow(cv::Mat img, std::vector<cv::DMatch> vali_matches, std::vector<cv::KeyPoint> img_kpts,
                           int cx, int W, int cy, int H, cv::Mat template_img, std::string logoName);
        int getGlobalFeature(cv::Mat img, std::vector<float> &fea);

        std::vector<std::string> logo_temp_name;
        std::vector<uint8> logo_site;
        std::vector< std::vector<cv::KeyPoint> > templates_kpts;
        std::vector< std::vector<float> > templates_hog_feat;
        std::vector<cv::Mat> template_img;
        std::vector<cv::Mat> templates_descs;
        std::map<std::string, std::vector<std::pair<int, float> > > cons;
        cv::Ptr<cv::Feature2D> detector;
        int logo_temp_num;
        int min_point_num;
        int min_match_num;
        float sift_near_thresh;

        // HOG特征
        int HOGH;
        int HOGW;
        cv::HOGDescriptor hog;
    };

    Logo* LogoFactory::Create() {
        return reinterpret_cast<Logo*>(new LogoWorker());
    }

    LogoWorker::LogoWorker() {
        min_point_num = 0;
        min_match_num = 0;
        sift_near_thresh = 0;
    }

    LogoWorker::~LogoWorker() {}

    int LogoWorker::Init(const std::string logo_dat_address) {
        this->detector = cv::SIFT::create(0, 3, 0.04, 10, 1.6);
        templates_hog_feat.clear();
        min_point_num = 5;
        min_match_num = 3;
        sift_near_thresh = 0.65;
        sift_near_thresh *= 512*512*sift_near_thresh;
        HOGH = 64;
        HOGW = 128;
        hog = cv::HOGDescriptor(cv::Size(HOGW, HOGH), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9);
        if (loadLogoDat(logo_dat_address) != 0) {
            std::cout << "can not read logo dat" << std::endl;
            return -1;
        }
        return 0;
    }

    // 功能说明：提取图像的HOG特征
    int LogoWorker::getGlobalFeature(cv::Mat img, std::vector<float> &fea) {
        if (img.empty()) return -1;
        cv::resize(img, img, cv::Size(HOGW, HOGH));
        hog.compute(img, fea, cv::Size(1, 1), cv::Size(0, 0));
        return 0;
    }

    int splitStr(const std::string str, std::vector<std::string> *pRet, std::string sep = "/") {
        if (str.empty()) return 0;
        pRet->clear();
        std::string tmp;
        std::string::size_type pos_begin = str.find_first_not_of(sep);
        std::string::size_type comma_pos = 0;
        while (pos_begin != std::string::npos) {
            comma_pos = str.find(sep, pos_begin);
            if (comma_pos != std::string::npos) {
                tmp = str.substr(pos_begin, comma_pos - pos_begin);
                pos_begin = comma_pos + sep.length();
            } else {
                tmp = str.substr(pos_begin);
                pos_begin = comma_pos;
            }

            if (!tmp.empty()) {
                (*pRet).push_back(tmp);
                tmp.clear();
            }
        }
        return (int)(*pRet).size();
    }

    // 载入模型文件
    int LogoWorker::loadLogoDat(const std::string logo_dat_address) {
        FILE* fp = fopen(logo_dat_address.c_str(), "rb");
        if (fp == 0) { return -1; }
        fread(&logo_temp_num, sizeof(int), 1, fp);
        int buf_size;
        char* buf = new char[MAX_TEMP_SIZE];
        uint8 site;
        int vali_num;
        float hog_dis;
        int cons_num;
        cons.clear();
        for (int i = 0; i < logo_temp_num; i++) {
            fread(&buf_size, sizeof(int), 1, fp);
            fread(buf, sizeof(char), buf_size, fp);
            buf[buf_size] = 0;
            std::string logo_image_name = std::string(buf);
            logo_temp_name.push_back(logo_image_name);
            fread(&site, sizeof(uint8), 1, fp);
            logo_site.push_back(site);

            fread(&cons_num, sizeof(int), 1, fp);
            if (cons_num > 0) {
                std::vector<std::pair<int, float> > cons_temp;
                for (int k = 0; k < cons_num; k++) {
                    fread(&vali_num, sizeof(int), 1, fp);
                    fread(&hog_dis, sizeof(float), 1, fp);
                    cons_temp.push_back(std::make_pair(vali_num, hog_dis));
                }
                cons[logo_image_name] = cons_temp;
            }

            fread(&buf_size, sizeof(int), 1, fp);
            fread(buf, sizeof(char), buf_size, fp);
            std::vector<char> img_vec;
            for (int i = 0; i < buf_size; i++) img_vec.push_back(buf[i]);

            // 对图片解码
            cv::Mat img = cv::imdecode(img_vec, 1);
            template_img.push_back(img);
        }
        delete [] buf;
        fclose(fp);
        std::vector<cv::KeyPoint> template_kpts;
        templates_hog_feat.resize(logo_temp_num);
        for (int n = 0; n < logo_temp_num; n++) {
            if(getGlobalFeature(template_img[n], templates_hog_feat[n]) < 0) continue;
            detector->detect(template_img[n], template_kpts);
            templates_kpts.push_back(template_kpts);
            if (template_kpts.size() < min_point_num) return -1;
            cv::Mat template_desc;
            detector->compute(template_img[n], template_kpts, template_desc);
            templates_descs.push_back(template_desc);
        }

        return 0;
    }

    uint8 getSiteCode(int height, int width, cv::Point2f pt) {
        float w = width*0.382;
        float h = w / 3;
        if (h >= height || w >= width) return 0;
        uint8 sitecode = 0;
        cv::Point2f ptup, ptdown;
        ptup.y = (height-h)/2;
        ptup.x = (width-w)/2;
        ptdown.y = ptup.y+h-1;
        ptdown.x = ptup.x+w-1;
        if (pt.x > ptup.x && pt.y > ptup.y && pt.x < ptdown.x && pt.y < ptdown.y){
            sitecode += 16;
        }
        w = width / 3.0;
        h = std::min(height/3.0, 120.0);
        ptup.y = (height-h)/2;
        ptup.x = (width-w)/2;
        ptdown.y = ptup.y+h-1;
        ptdown.x = ptup.x+w-1;
        if (pt.x < ptup.x && pt.y < ptup.y) {
            sitecode += 8;
        } else if (pt.x > ptdown.x && pt.y < ptup.y) {
            sitecode += 4;
        } else if (pt.x < ptup.x && pt.y > ptdown.y) {
            sitecode += 2;
        } else if (pt.x > ptdown.x && pt.y > ptdown.y) {
            sitecode += 1;
        }
        return sitecode;
    }

    // 功能说明：对SIFT进行空间校验
    // 输入：图a的两个关键点，图b的两个关键点
    static inline int spaceValidate(const cv::KeyPoint &pa0, const cv::KeyPoint &pa1, \
                                    const cv::KeyPoint &pb0, const cv::KeyPoint &pb1) {
        // A图像两个关键点的角度差
        float angleA = pa0.angle - pa1.angle;
        // B图像
        float angleB = pb0.angle - pb1.angle;

        // 两角度差值的绝对值
        float diff1 = std::abs(angleA - angleB);

        float thetaA;
        float deltaX_A = pa1.pt.x - pa0.pt.x;
        float deltaY_A = pa1.pt.y - pa0.pt.y;
        if (deltaX_A == 0) {
            if (deltaY_A >= 0) {
                thetaA = 90;
            } else {
                thetaA = 270;
            }
        } else if (deltaX_A > 0) {
            float tanv = deltaY_A/deltaX_A;
            thetaA = atan(tanv)*180/PI;
        } else {
            float tanv = deltaY_A/deltaX_A;
            if (deltaY_A >= 0) {
                thetaA = atan(tanv)*180/PI + 180;
            } else {
                thetaA = atan(tanv)*180/PI - 180;
            }
        }
        thetaA -= pa0.angle;

        float thetaB;
        float deltaX_B = pb1.pt.x - pb0.pt.x;
        float deltaY_B = pb1.pt.y - pb0.pt.y;
        if (deltaX_B == 0) {
            if (deltaY_B >= 0) {
                thetaB = 90;
            } else {
                thetaB = 270;
            }
        } else if (deltaX_B > 0) {
            float tanv = deltaY_B/deltaX_B;
            thetaB = atan(tanv)*180/PI;
        } else {
            float tanv = deltaY_B/deltaX_B;
            if (deltaY_B >= 0) {
                thetaB = atan(tanv)*180/PI + 180;
            } else {
                thetaB = atan(tanv)*180/PI - 180;
            }
        }
        thetaB -= pb0.angle;

        float diff2 = std::abs(thetaA - thetaB);

        if (diff1 < 10 && diff2 < 10) return 1;
        return 0;
    }

    // 删除误匹配特征点
    std::vector<cv::DMatch> LogoWorker::spatialVerify(std::vector<cv::KeyPoint> &qKpts1, std::vector<cv::KeyPoint> &qKpts2, std::vector<cv::DMatch> &rawMatches)
    {
        std::vector<cv::DMatch> goodMatches;
        goodMatches.clear();
        int numRawMatch = (int)rawMatches.size();
        int **brother_matrix = new int*[numRawMatch];
        for (int i = 0; i < numRawMatch; i++) {
            brother_matrix[i] = new int[numRawMatch];
            brother_matrix[i][i] = 0;
        }

        for (int i = 0; i < numRawMatch; i++)
        {
            int queryIdx0 = rawMatches[i].queryIdx;
            int trainIdx0 = rawMatches[i].trainIdx;
            for (int j = i+1; j < numRawMatch; j++)
            {
                int queryIdx1 = rawMatches[j].queryIdx;
                int trainIdx1 = rawMatches[j].trainIdx;
                auto lp0 = qKpts1[queryIdx0];
                auto rp0 = qKpts1[queryIdx1];
                auto lp1 = qKpts2[trainIdx0];
                auto rp1 = qKpts2[trainIdx1];
                brother_matrix[i][j] = spaceValidate(lp0, rp0, lp1, rp1);
                brother_matrix[j][i] = brother_matrix[i][j];
            }
        }

        int map_size = numRawMatch;
        int *mapId = new int[numRawMatch];
        for (int i = 0; i < numRawMatch; i++) mapId[i] = i;
        while (true)
        {
            int maxv = -1;
            int maxid = 0;
            for (int i = 0; i < map_size; i++)
            {
                int sum = 0;
                for (int j = 0; j < map_size; j++) sum += brother_matrix[mapId[i]][mapId[j]];
                if (sum > maxv)
                {
                    maxv = sum;
                    maxid = mapId[i];
                }
            }
            if (maxv == 0) break;
            goodMatches.push_back(rawMatches[maxid]);
            int id = 0;
            for (int i = 0; i < map_size; i++)
            {
                if (brother_matrix[maxid][mapId[i]] > 0) mapId[id++] = mapId[i];
            }
            map_size = maxv;
        }

        delete [] mapId;
        for(int i=0; i < numRawMatch; i++)
        {
            delete [] brother_matrix[i];
        }
        delete [] brother_matrix;

        return goodMatches;
    }

    static inline float distanceL2(const std::vector<float> &x, const std::vector<float> &y) {
        if (x.size() != y.size()) return -1;
        float dis = 0;
        for (int i = 0; i < x.size(); i++) {
            float dif = x[i] - y[i];
            dis += dif*dif;
        }
        return sqrt(dis);
    }

    static inline float distanceL2(const cv::KeyPoint &x, const cv::KeyPoint &y) {
        float dis = 0;
        float dif = x.pt.x - y.pt.x;
        dis += dif*dif;
        dif = x.pt.y - y.pt.y;
        dis += dif*dif;
        return sqrt(dis);
    }

    /*
     返回：True，当前模板没有匹配上
     False，当前模板匹配上
     1. SIFT匹配点数设置为-1.0， 算出来的HOG距离大于设定的距离，当前模板匹配失败
     2. HOG距离设置为-1.0， 算出来SIFT的匹配点小于设定的匹配点数，当前模板匹配失败
     3. HOG和SIFT都设置了阈值，算出来的HOG距离大于设定的距离且算出来SIFT的匹配点小于设定的匹配点数，当前模板匹配失败

     匹配成功的条件
     1. SIFT匹配点数设置为-1.0， 算出来的HOG距离小于等于设定的距离，当前模板匹配成功
     2. HOG距离设置为-1.0， 算出来SIFT的匹配点大于等于设定的匹配点数，当前模板匹配成功
     3. HOG和SIFT都设置了阈值，算出来的HOG距离小于等于设定的距离，或者，算出来SIFT的匹配点大于等于设定的匹配点数，当前模板匹配成功
     */
    bool LogoWorker::isContinue(const std::string &logo_temp_name, int vali_num, float hog_dis) {
        std::map<std::string, std::vector<std::pair<int, float> > >::iterator it;
        it = cons.find(logo_temp_name);
        if (it == cons.end()) return false;
        for (int i = 0; i < it->second.size(); i++) {
            if (it->second[i].first < 0) {
                std::cout << "hog_dis: " << hog_dis << ", hog threhold: " << it->second[i].second << std::endl;
                if (hog_dis > it->second[i].second) return true;
            } else if (it->second[i].second < 0) {
                if (vali_num < it->second[i].first) return true;
            } else {
                if (vali_num < it->second[i].first && hog_dis > it->second[i].second) return true;
            }
        }
        return false;
    }

    void LogoWorker::matchInfoShow(cv::Mat img, std::vector<cv::DMatch> vali_matches, std::vector<cv::KeyPoint> img_kpts,
                                   int cx, int W, int cy, int H, cv::Mat template_img, std::string logoName) {

        // 区域可视化
        cv::Point2f pt;
        int height = img.rows;
        int width = img.cols;
        float w = width*0.382;
        float h = w / 3;
        if (h >= height || w >= width) return;
        cv::Point2f ptup1, ptdown1;
        ptup1.y = (height-h)/2;
        ptup1.x = (width-w)/2;
        ptdown1.y = ptup1.y+h-1;
        ptdown1.x = ptup1.x+w-1;
        cv::Point2f ptup2, ptdown2;
        w = width / 3.0;
        h = std::min(height/3.0, 120.0);
        ptup2.y = (height-h)/2;
        ptup2.x = (width-w)/2;
        ptdown2.y = ptup2.y+h-1;
        ptdown2.x = ptup2.x+w-1;
        cv::rectangle(img, ptup1, ptdown1, cv::Scalar(0, 0, 255)); // 中心区域判断
        putText(img, "16", cv::Point((ptup1.x + ptdown1.x)/2, (ptup1.y + ptdown1.y)/2), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255));
        cv::rectangle(img, cv::Point2f(0, 0), ptup2, cv::Scalar(255, 255, 255)); // 区域8
        putText(img, "8", cv::Point((ptup2.x)/2, (ptup2.y)/2), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255));
        cv::rectangle(img, cv::Point2f(ptdown2.x, 0), cv::Point2f(img.cols-1, ptup2.y), cv::Scalar(255, 255, 255)); // 区域4
        putText(img, "4", cv::Point((ptdown2.x+img.cols-1)/2, (ptup2.y)/2), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255));
        cv::rectangle(img, cv::Point2f(0, ptdown2.y), cv::Point2f(ptup2.x, img.rows-1), cv::Scalar(255, 255, 255)); // 区域2
        putText(img, "2", cv::Point((ptup2.x)/2, (ptdown2.y+img.rows-1)/2), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255));
        cv::rectangle(img, ptdown2, cv::Point2f(img.cols-1, img.rows-1), cv::Scalar(255, 255, 255)); // 区域1
        putText(img, "1", cv::Point((ptdown2.x+img.cols-1)/2, (ptdown2.y+img.rows-1)/2), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255));
        for (int i = 0; i < vali_matches.size(); i++) {
            int tId = vali_matches[i].trainIdx;
            if (tId < 0) continue;
            pt.x = img_kpts[tId].pt.x;
            pt.y = img_kpts[tId].pt.y;
            //cv::circle(imgClone, pt, 4, cv::Scalar(0, 0, 255), 4);
        }
        cv::rectangle(img, cv::Point(cx, cy), cv::Point(cx+W, cy+H), cv::Scalar(255, 255, 255));

        const int ht = std::max(img.rows, template_img.rows);
        const int wt = img.cols + template_img.cols;
        cv::Mat output(ht, wt, CV_8UC3, cv::Scalar(0, 0, 0));
        img.copyTo(output(cv::Rect(0, 0, img.cols, img.rows)));
        template_img.copyTo(output(cv::Rect(img.cols, 0, template_img.cols, template_img.rows)));
        cv::imshow("img", output);
        cv::waitKey();
    }

    int LogoWorker::templateVerify(cv::Mat &img, cv::Mat &img_descs, std::vector<cv::KeyPoint> &img_kpts,
                                   cv::flann::Index &img_sift_index, const std::string frame_id,
                                   int ith, bool &rotate_flag) {
        cv::Mat indices;
        cv::Mat dis;
        // 计算模板与图片间SIFT特征的最近邻
        img_sift_index.knnSearch(templates_descs[ith], indices, dis, 1, cv::flann::SearchParams(64));
        //img_sift_index.knnSearch(templates_descs[ith], indices, dis, 2);
        std::vector<cv::DMatch> matches;
        for (int i = 0; i < templates_kpts[ith].size(); i++) {
            cv::DMatch dmatch;
            dmatch.imgIdx = 0;
            if (dis.at<float>(i, 0) < sift_near_thresh) {
            //if (dis.at<float>(i, 0) < 0.9*dis.at<float>(i, 1)){
                dmatch.distance = dis.at<float>(i, 0);
                // queryIdx为模板
                dmatch.queryIdx = i;
                // trainIdx为待检测图像
                dmatch.trainIdx = indices.at<int>(i, 0);
                matches.push_back(dmatch);
            }
        }


        // 匹配点数目小于3，跳过该模板
        int match_num = (int)matches.size();
        if (match_num < 3) return 0;
        std::vector<cv::DMatch> vali_matches = spatialVerify(templates_kpts[ith], img_kpts, matches);
        int vali_match_num = (int)vali_matches.size();
        if (vali_match_num < min_match_num) return 0;
        /*std::vector<cv::DMatch> vali_matches = matches;
        int vali_match_num = (int)vali_matches.size();*/
#if 0
        // 显示SIFT匹配点对
        cv::Mat img_matches0;
        cv::drawMatches(template_img[ith], templates_kpts[ith], img, img_kpts, vali_matches, img_matches0);
        cv::imshow("match_point", img_matches0);
        cv::waitKey();
#endif

        // 判断图片是否旋转
        // note: 直接对img做变换，可能潜在的风险，应该对img做一份复制
        /*int count0 = 0;
        int count90 = 0;
        int countAngle = 0;
        std::vector<cv::DMatch> vali_matches90;
        for (int i = 0; i < vali_matches.size(); i++) {
            int qId0 = vali_matches[i].queryIdx;
            int tId0 = vali_matches[i].trainIdx;
            float diffAngle = templates_kpts[ith][qId0].angle - img_kpts[tId0].angle;
            if(std::abs(diffAngle) < 10) {
                count0 += 1;
            }
            if (std::abs(diffAngle) < 100 && std::abs(diffAngle) > 90) {
                count90 += 1;
                if(diffAngle <= 0) countAngle += 1;
                vali_matches90.push_back(vali_matches.at(i));
            }
        }
        cv::Point center, center_new;
        if(count90 > count0 && count90>=3) {
            rotate_flag = true;
            center.y = img.rows/2.0;
            center.x = img.cols/2.0;
            if (countAngle > 0){
                rotate2D(img, img, 90);
            } else {
                rotate2D(img, img, -90);
            }
            center_new.y = img.rows/2.0;
            center_new.x = img.cols/2.0;
            for(int i = 0; i < img_kpts.size(); i++) {
                if (countAngle > 0){
                    img_kpts[i].pt = RotatePoint(center, img_kpts[i].pt, -PI/2);
                } else {
                    img_kpts[i].pt = RotatePoint(center, img_kpts[i].pt, PI/2);
                }
                img_kpts[i].pt.x = img_kpts[i].pt.x + center_new.x;
                img_kpts[i].pt.y = img_kpts[i].pt.y + center_new.y;
            }
            std::cout << "frame_id: " << frame_id << ", image is rotated" << std::endl;
            vali_matches = vali_matches90;
        }*/
#if 1
        std::cout << "img keypoints num: " << img_kpts.size() << std::endl;
        std::cout << "match num: " << vali_match_num << std::endl;
#endif

        // 计算角度差
        float aver_angle = 0;
        for (int i = 0; i < vali_matches.size(); i++) {
            int qId = vali_matches[i].queryIdx;
            int tId = vali_matches[i].trainIdx;
            float tmp_angle = img_kpts[tId].angle - templates_kpts[ith][qId].angle;
            aver_angle += tmp_angle;
            std::cout << "角度差: " << tmp_angle << std::endl;
        }
        aver_angle = aver_angle/vali_matches.size();

        // 通过距离求累积缩放因子
        float scale = 0;
        int num = 0;
        for (int i = 0; i < vali_matches.size(); i++) {
            int qId0 = vali_matches[i].queryIdx;
            int tId0 = vali_matches[i].trainIdx;
            for (int j = i+1; j < vali_matches.size(); j++) {
                int qId1 = vali_matches[j].queryIdx;
                int tId1 = vali_matches[j].trainIdx;
                float dist_logo = distanceL2(templates_kpts[ith][qId0], templates_kpts[ith][qId1]);
                float dist = distanceL2(img_kpts[tId0], img_kpts[tId1]);
                if (dist_logo != 0) {
                    num++;
                    scale += dist / dist_logo;
                }
            }
            // 上面过程，跟用scale来计算差不大
            //float tmp_scale = 1.0*img_kpts[tId0].size/templates_kpts[ith][qId0].size;
        }

        // 累积缩放因子求平均值
        if (num > 0) scale /= num;

        if (scale >= 1) {
            std::cout << "scale: " << scale << ", " << "图像比模板尺寸大" << std::endl;
        } else {
           std::cout << "scale: " << scale << ", " << "图像比模板尺寸小" << std::endl;
        }

        //std::cout << "scale: " << scale << std::endl;
        //if (scale < 0.36 || scale > 2.5) return 0;
        //if (scale < 0.3 || scale > 2.5) return 0;
        if (scale < 0.25 || scale > 2.5) return 0;

        int h = template_img[ith].rows;
        int w = template_img[ith].cols;
        int H = std::min((double)h*scale, (double)img.rows);
        int W = std::min((double)w*scale, (double)img.cols);

        // 求待检测图像logo所在区域的原点
        float fcx = 0;
        float fcy = 0;
        for (int i = 0; i < vali_matches.size(); i++) {
            int qId = vali_matches[i].queryIdx;
            int tId = vali_matches[i].trainIdx;
            int tmp_x = img_kpts[tId].pt.x - templates_kpts[ith][qId].pt.x * scale;
            int tmp_y = img_kpts[tId].pt.y - templates_kpts[ith][qId].pt.y * scale;
            std::cout << tmp_x << ": " << tmp_y << std::endl;
            fcx += img_kpts[tId].pt.x - templates_kpts[ith][qId].pt.x * scale;
            fcy += img_kpts[tId].pt.y - templates_kpts[ith][qId].pt.y * scale;
        }
        // 区域原点求平均值
        fcx /= vali_matches.size();
        fcy /= vali_matches.size();

        // 获得待检测图像上logo位置区域
        int cx = fcx;
        int cy = fcy;
        cv::Mat crop(H, W, CV_8UC3);
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                for (int k = 0; k < img.channels(); k++) {
                    int ii = i + cy;
                    int jj = j + cx;
                    if (ii < 0) {
                        ii = -ii;
                    } else if (ii >= img.rows) {
                        ii = 2*img.rows-ii-1;
                    }
                    if (jj < 0) {
                        jj = -jj;
                    } else if (jj >= img.cols) {
                        jj = 2*img.cols-jj-1;
                    }
                    crop.at<cv::Vec3b>(i, j)[k] = img.at<cv::Vec3b>(ii, jj)[k];
                }
            }
        }
#if 1
        // 显示SIFT匹配点对
        cv::Mat img_matches;
        cv::Mat img_clone = img.clone();
        cv::drawMatches(template_img[ith], templates_kpts[ith], img_clone, img_kpts, vali_matches, img_matches);
        cv::imshow("match", img_matches);

        cv::imshow("crop_img", crop);

        // 显示SIFT找到的区域
        img_clone = img.clone();
        cv::circle(img_clone, cv::Point(cx, cy), 4, cv::Scalar(255, 0, 0), 4);
        int xcrop0 = std::min(std::max(0.0, (double)fcx), (double)img.cols);
        int ycrop0 = std::min(std::max(0.0, (double)fcy), (double)img.rows);
        int xcrop1 = std::min(std::max(0.0, (double)(fcx+W)), (double)img.cols);
        int ycrop1 = std::min(std::max(0.0, (double)(fcy+H)), (double)img.rows);

        if(std::abs(aver_angle - 0) <= 5) {
            cv::rectangle(img_clone, cv::Point(xcrop0, ycrop0), cv::Point(xcrop1, ycrop1), cv::Scalar(0, 0, 255), 4);
        }

        if(std::abs(aver_angle - 90) <= 5 && aver_angle > 0) {
            cv::rectangle(img_clone, cv::Point(xcrop0, ycrop0), cv::Point(xcrop1, ycrop1), cv::Scalar(255, 0, 0), 4);
        }

        if(std::abs(aver_angle + 90) <= 5 && aver_angle < 0) {
            cv::rectangle(img_clone, cv::Point(xcrop0, ycrop0), cv::Point(xcrop1, ycrop1), cv::Scalar(0, 255, 0), 4);
        }

        cv::imshow("img", img_clone);
        cv::waitKey();
#endif

        // 计算待检测图像上logo位置区域的HOG特征
        std::vector<float> fea;
        if(getGlobalFeature(crop, fea) < 0) return 0;
        float dist_feat = distanceL2(templates_hog_feat[ith], fea);
        //printf("\nHoG distance: %f, vali_match_num: %d\n", dist_feat, vali_match_num);
        //检查SIFT匹配点数和HOG距离是否满足要求
        if (isContinue(logo_temp_name[ith], vali_match_num, dist_feat)) return 0;
        if (dist_feat < 5.20) {
#if 1
            img_clone = img.clone();
            matchInfoShow(img_clone, vali_matches, img_kpts, cx,  W, cy, H, template_img[ith], logo_temp_name[ith]);
#endif
            std::cout << "hit template: " << logo_temp_name[ith] << ", dis: " << dist_feat << std::endl;
            return 1;
        } else if (dist_feat < 6.5 || (dist_feat < 7.1 && vali_matches.size() >= 5) ) {
            if (logo_site[ith] == 0) {
#if 1
                img_clone = img.clone();
                matchInfoShow(img_clone, vali_matches, img_kpts, cx,  W, cy, H, template_img[ith], logo_temp_name[ith]);
#endif
                return 1;
            } else if (logo_site[ith] < 0) {
                if (frame_id.substr(frame_id.size()-1, frame_id.size()) != "t") return 0;
            }
            int valid_count = 0;
            cv::Point2f pt;
            for (int i = 0; i < vali_matches.size(); i++) {
                int tId = vali_matches[i].trainIdx;
                if (tId < 0) continue;
                pt.x = img_kpts[tId].pt.x;
                pt.y = img_kpts[tId].pt.y;
                uint8 site_code = getSiteCode(img.rows, img.cols, pt);
                uint8 valid_kp;
                uint8 icenterv = 255 - 16;
                if (frame_id.substr(frame_id.size()-1, frame_id.size()) == "t") {
                    valid_kp = site_code & logo_site[ith];
                } else {
                    valid_kp = site_code & logo_site[ith] & icenterv;
                }
                if (valid_kp) valid_count++;
            }
            if (valid_count >= min_match_num) {
#if 1
                img_clone = img.clone();
                matchInfoShow(img_clone, vali_matches, img_kpts, cx,  W, cy, H, template_img[ith], logo_temp_name[ith]);
#endif
                std::cout << "hit template: " << logo_temp_name[ith] << ", points: " << valid_count << std::endl;
                return 1;
            } else {
                return 0;
            }
        }
        return 0;
    }

    // logo检测主要函数
    std::string LogoWorker::detect(cv::Mat &img, const std::string frame_id) {
        // 异常处理
        if (img.empty()){
            std::cout << "image is empty" << std::endl;
            return "";
        }

        // 提取待检测图像的SIFT
        cv::Mat img_descs;
        std::vector<cv::KeyPoint> img_kpts;
        detector->detect(img, img_kpts);

        // 关键点数目小于5(模板生成中也是设置的5)则不检测
        if (img_kpts.size() < min_point_num){
            std::cout << "number of image keypoints: " << img_kpts.size() << std::endl;
            return "";
        }
        detector->compute(img, img_kpts, img_descs);

        // 对待检测图片的SIFT描述子构建KD树
        cv::flann::Index img_sift_index(img_descs, cv::flann::KDTreeIndexParams(8));
        //cv::flann::Index img_sift_index(img_descs, cv::flann::LinearIndexParams());

        // 遍历模板
        int n = 0;
        cv::Mat imgClone = img.clone();
        std::vector<cv::KeyPoint> img_kptsClone = img_kpts;
        bool rotate_flag = false;
        for (n = 0; n < logo_temp_num; n++) {
            if (rotate_flag) {
                img = imgClone;
                img_kpts = img_kptsClone;
            }
            int ret = templateVerify(img, img_descs, img_kpts, img_sift_index, frame_id, n, rotate_flag);
            if (ret) break;
        }

        if (n < logo_temp_name.size()) {
            return logo_temp_name[n];
        } else {
            return "";
        }
    }

    int LogoWorker::LogoDetect(const std::string &image_address, const std::string &frame_id, std::string *logo) {
        *logo = "";
        cv::Mat img;
        if (image_address.size() < 512) {
            img = cv::imread(image_address.c_str(), 1);
        } else {
            std::vector<char> img_vec(image_address.c_str(), image_address.c_str()+image_address.size());
            img = cv::imdecode(img_vec, 1);
        }
        if (img.empty()) return -1;
        int maxlen = std::max(img.rows, img.cols);
        if (maxlen > 640) {
            float scale = 640.0 / maxlen;
            int std_width = img.cols*scale;
            int std_height = img.rows*scale;
            cv::resize(img, img, cv::Size(std_width, std_height));
        }
        std::string logoname = detect(img, frame_id);
        if (logoname != "") {
            std::vector<std::string> tmpStr;
            splitStr(logoname, &tmpStr, "_");
            *logo = tmpStr[0];
            //*logo = logoname;
        }
        return 0;
    }

}
