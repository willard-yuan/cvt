
#include "general.hpp"
#include "vl_covdet.hpp"
#include "vl_kdtree.hpp"
#include "geo_verification.hpp"
#include <opencv2/flann/flann.hpp>


int main(int argc, char** argv){
    
	superluOpts opts; //几何校正参数
    int numNeighbors = 2;
    std::string annSearch = "OpenCV_KD";
    
    // 提取covdet特征
    std::string qimageName = "/Users/willard/Pictures/0919/wine_2.jpg";
    std::vector<std::vector<float>> qframes;
    std::vector<std::vector<float>> qdesc;
    cv::Mat qMat = cv::imread(qimageName);
    vl_covdet(qMat, qframes, qdesc);
    
    std::string dimageName = "/Users/willard/Pictures/0919/wine_1.jpg";
    std::vector<std::vector<float>> dframes;
    std::vector<std::vector<float>> ddesc;
    cv::Mat dMat = cv::imread(dimageName);
    //cv::resize(dMat, dMat, cv::Size(int(0.6*dMat.cols), int(0.6*dMat.rows)));
    vl_covdet(dMat, dframes, ddesc);
    
    std::vector<std::vector<int>> idx;
    std::vector<std::vector<float>> dis;
    
    // 使用OpenCV的kd树
    if(annSearch == "OpenCV_KD"){
        cv::Mat covdetDMat = vec2cvMat_2D(ddesc);
        cv::flann::Index flann_index(covdetDMat, cv::flann::KDTreeIndexParams(64), cvflann::FLANN_DIST_EUCLIDEAN); //kd树
        //cv::flann::Index flann_index(covdetDMat,cv::flann::LinearIndexParams::LinearIndexParams(), cvflann::FLANN_DIST_EUCLIDEAN); //brute search
        
        flann_index.save("myFirstIndex"); //保存索引结构
        cv::flann::Index flann;
        flann.load(covdetDMat,"myFirstIndex");
        flann.save("mySecondIndex");
        
        std::vector<int> tmpIdx(numNeighbors);
        std::vector<float> tmpDis(numNeighbors);
        for(int i = 0; i < qdesc.size(); i++){
            flann_index.knnSearch(qdesc[i], tmpIdx, tmpDis, numNeighbors, cv::flann::SearchParams(128));
            idx.push_back(tmpIdx);
            dis.push_back(tmpDis);
        }
    }
    
    // 使用vlfeat中的kd树
    if(annSearch == "VLfeat_KD"){
        float * descsQ = vectors2OneArray(qdesc); // 将特征拼成一位数组
        float * descsD = vectors2OneArray(ddesc);
        int numTrees = 8;
        int maxNumComparisions = 800;
        VlKDForest * forest = kdtreebuild(descsD, ddesc.size(), numTrees, maxNumComparisions, descsQ, qdesc.size(), numNeighbors, idx, dis);
        vl_kdforestquery(forest, numNeighbors, (int)qdesc.size(), descsQ, idx, dis);
        delete descsQ;
        delete descsD;
    }
    
    std::vector<std::vector<int>> matchedIdx; // <查询图像, 目标图像>
    std::vector<cv::Point2f> Qcoords, Dcoords;
    for(int i=0; i < idx.size(); i++){
        if(dis[i].at(0) < 0.84*dis[i].at(1)){
            //matchedIdx.push_back(std::pair<int, int>(ind[i].at(0), i));
            std::vector<int> tmpIdx(2);
            tmpIdx[0] = i + 1; //查询图像查询点ID
            tmpIdx[1] = idx[i].at(0) + 1;
            matchedIdx.push_back(tmpIdx);
            cv::Point2f tmp1;
            tmp1.x = dframes[idx[i].at(0)].at(0);
            tmp1.y = dframes[idx[i].at(0)].at(1);
            Dcoords.push_back(tmp1);
            cv::Point2f tmp2;
            tmp2.x = qframes[i].at(0);
            tmp2.y = qframes[i].at(1);
            Qcoords.push_back(tmp2);
        }
    }
    
    // draw 2nn result
    plotMatches(qMat, dMat, Qcoords, Dcoords);
    
    arma::mat frames1 = vec2mat(qframes);
    arma::mat frames2 = vec2mat(dframes);
    arma::mat matches = vec2mat(matchedIdx);
    
	arma::uvec inliers_final;
	//arma::mat H_final;

	inliers_final = geometricVerification(frames1, frames2, matches, opts);

    std::vector<cv::Point2f> srcPoints, dstPoints;
    arma::mat matches_geo = matches.cols(inliers_final);
	for (unsigned int i = 0; i < matches_geo.n_cols; ++i){
		cv::Point2f pt1, pt2;
		pt1.x = frames1.at(0, matches_geo.at(0, i) - 1);
		pt1.y = frames1.at(1, matches_geo.at(0, i) - 1);
		pt2.x = frames2.at(0, matches_geo.at(1, i) - 1);
		pt2.y = frames2.at(1, matches_geo.at(1, i) - 1);
		srcPoints.push_back(pt1);
		dstPoints.push_back(pt2);
	}

    // draw geo verification result
	plotMatches(qMat, dMat, srcPoints, dstPoints);
	system("pause");
	return 0;
}
