#include "covdetExtractor.hpp"
#include "vl_kdtree.hpp"
#include "general.h"

#include <opencv2/flann/flann.hpp>

extern "C" {
#include "vl/kdtree.h"
}

int main(int argc, char** argv){
    
	superluOpts opts; //几何校正参数
    int numNeighbors = 2;
    std::string annSearch = "OpenCV_KD";
    
    // 提取covdet特征
    siftDesctor covdetQ;
    covdetQ.imageName = "/Users/willard/Pictures/test/test_1.jpg";
    covdetQ.covdet_keypoints_and_descriptors(covdetQ.imageName, covdetQ.frame, covdetQ.desc);
    
    siftDesctor covdetD;
    covdetD.imageName = "/Users/willard/Pictures/test/test_2.jpg";
    covdetD.covdet_keypoints_and_descriptors(covdetD.imageName, covdetD.frame, covdetD.desc);
    
    std::vector<std::vector<int>> idx;
    std::vector<std::vector<float>> dis;
    
    // 使用OpenCV的kd树
    if(annSearch == "OpenCV_KD"){
        cv::Mat covdetDMat = vec2cvMat_2D(covdetD.desc);
        cv::flann::Index flann_index(covdetDMat, cv::flann::KDTreeIndexParams(64), cvflann::FLANN_DIST_EUCLIDEAN); //kd树
        //cv::flann::Index flann_index(covdetDMat,cv::flann::LinearIndexParams::LinearIndexParams(), cvflann::FLANN_DIST_EUCLIDEAN); //brute search
        
        flann_index.save("myFirstIndex"); //保存索引结构
        cv::flann::Index flann;
        flann.load(covdetDMat,"myFirstIndex");
        flann.save("mySecondIndex");
        
        std::vector<int> tmpIdx(numNeighbors);
        std::vector<float> tmpDis(numNeighbors);
        for(int i = 0; i < covdetQ.desc.size(); i++){
            flann_index.knnSearch(covdetQ.desc[i], tmpIdx, tmpDis, numNeighbors, cv::flann::SearchParams(128));
            idx.push_back(tmpIdx);
            dis.push_back(tmpDis);
        }
    }
    
    // 使用vlfeat中的kd树
    if(annSearch == "VLfeat_KD"){
        float * descsQ = vectors2OneArray(covdetQ.desc); // 将特征拼成一位数组
        float * descsD = vectors2OneArray(covdetD.desc);
        int numTrees = 8;
        int maxNumComparisions = 800;
        VlKDForest * forest = kdtreebuild(descsD, covdetD.desc.size(), numTrees, maxNumComparisions, descsQ, covdetQ.desc.size(), numNeighbors, idx, dis);
        vl_kdforestquery(forest, numNeighbors, (int)covdetQ.desc.size(), descsQ, idx, dis);
        delete descsQ;
        delete descsD;
    }
    
    std::vector<std::vector<int>> matchedIdx; // <查询图像, 目标图像>
    std::vector<cv::Point2f> Qcoords, Dcoords;
    for(int i=0; i < idx.size(); i++){
        if(dis[i].at(0) < 0.64*dis[i].at(1)){
            //matchedIdx.push_back(std::pair<int, int>(ind[i].at(0), i));
            std::vector<int> tmpIdx(2);
            tmpIdx[0] = i + 1; //查询图像查询点ID
            tmpIdx[1] = idx[i].at(0) + 1;
            matchedIdx.push_back(tmpIdx);
            cv::Point2f tmp1;
            tmp1.x = covdetD.frame[idx[i].at(0)].at(0);
            tmp1.y = covdetD.frame[idx[i].at(0)].at(1);
            Qcoords.push_back(tmp1);
            cv::Point2f tmp2;
            tmp2.x = covdetQ.frame[i].at(0);
            tmp2.y = covdetQ.frame[i].at(1);
            Dcoords.push_back(tmp2);
        }
    }
    
    cv::Mat QImage = cv::imread(covdetQ.imageName);
    cv::Mat DImage = cv::imread(covdetD.imageName);
    //plotMatches(QImage, DImage, Qcoords, Dcoords); // 画1nn<0.64*2nn的匹配结果
    
    arma::mat frames1 = vec2mat(covdetQ.frame);
    arma::mat frames2 = vec2mat(covdetD.frame);
    arma::mat matches = vec2mat(matchedIdx);
    
	arma::uvec inliers_final;
	//arma::mat H_final;

	inliers_final = geometricVerification(frames1, frames2, matches, opts);

    std::vector<cv::Point2f> srcPoints, dstPoints;
    arma::mat matches_geo = matches.cols(inliers_final);
	//cout << matches_geo.n_rows << "+++++" <<matches_geo.n_cols << endl;
	for (unsigned int i = 0; i < matches_geo.n_cols; ++i){
		cv::Point2f pt1, pt2;
		//cout << matches_geo.at(0, i) << " " << matches_geo.at(1, i) << endl;
		pt1.x = frames1.at(0, matches_geo.at(0, i) - 1);
		pt1.y = frames1.at(1, matches_geo.at(0, i) - 1);
		pt2.x = frames2.at(0, matches_geo.at(1, i) - 1);
		pt2.y = frames2.at(1, matches_geo.at(1, i) - 1);
		srcPoints.push_back(pt1);
		dstPoints.push_back(pt2);
	}

	plotMatches(QImage, DImage, srcPoints, dstPoints);

	system("pause");
	return 0;
}
