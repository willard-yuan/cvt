#include "utils.h"

/******************************************************************
 * 函数功能：生成随机颜色
 */
static cv::Scalar randomColor(cv::RNG& rng)
{
    int icolor = (unsigned)rng;

    return cv::Scalar(icolor&0xFF, (icolor>>8)&0xFF, (icolor>>16)&0xFF);
}

/******************************************************************
 * 函数功能：画匹配的特征点对
 * 参考：https://gist.github.com/thorikawa/3398619
 */
void plotMatches(const cv::Mat &srcColorImage, const cv::Mat &dstColorImage, std::vector<cv::Point2f> &srcPoints, std::vector<cv::Point2f> &dstPoints){
    
	// Create a image for displaying mathing keypoints
    cv::Size sz = cv::Size(srcColorImage.size().width + dstColorImage.size().width, srcColorImage.size().height + dstColorImage.size().height);
    cv::Mat matchingImage = cv::Mat::zeros(sz, CV_8UC3);
    
    // Draw camera frame
    cv::Mat roi1 = cv::Mat(matchingImage, cv::Rect(0, 0, srcColorImage.size().width, srcColorImage.size().height));
    srcColorImage.copyTo(roi1);
    // Draw original image
    cv::Mat roi2 = cv::Mat(matchingImage, cv::Rect(srcColorImage.size().width, srcColorImage.size().height, dstColorImage.size().width, dstColorImage.size().height));
    //cv::Mat roi2 = cv::Mat(matchingImage, cv::Rect(srcColorImage.size().width, 0, dstColorImage.size().width, dstColorImage.size().height));
    dstColorImage.copyTo(roi2);

    cv::RNG rng(0xFFFFFFFF);
    // Draw line between nearest neighbor pairs
    for (int i = 0; i < (int)srcPoints.size(); ++i) {
        cv::Point2f pt1 = srcPoints[i];
        cv::Point2f pt2 = dstPoints[i];
        cv::Point2f from = pt1;
        cv::Point2f to   = cv::Point(srcColorImage.size().width + pt2.x, srcColorImage.size().height + pt2.y);
        //cv::Point2f to   = cv::Point(srcColorImage.size().width + pt2.x, pt2.y);
        cv::line(matchingImage, from, to, randomColor(rng), 2);
    }

	// 在图像中显示匹配点数文本
	/*Point org;
    org.x = rng.uniform(matchingImage.cols/10, matchingImage.rows/10);
    org.y = rng.uniform(matchingImage.rows/10, matchingImage.rows/10);
	putText(matchingImage, "Testing text rendering", org, rng.uniform(0,8), rng.uniform(0,10)*0.05+0.1, randomColor(rng), rng.uniform(1, 10), 8);*/

    // Display mathing image
    cv::resize(matchingImage, matchingImage, cv::Size(matchingImage.cols/2, matchingImage.rows/2));
    //cv::resize(matchingImage, matchingImage, cv::Size(matchingImage.cols, matchingImage.rows));
    cv::imshow("Geometric Verification", matchingImage);
    cvWaitKey(0);
}

/******************************************************************
 * 函数功能：使用OpenCV自带的RANSAC寻找内点
 * 
 */
void findInliers(std::vector<cv::KeyPoint> &qKeypoints, std::vector<cv::KeyPoint> &objKeypoints, std::vector<cv::DMatch> &matches, const cv::Mat &srcColorImage, const cv::Mat &dstColorImage){
    // 获取关键点坐标
    std::vector<cv::Point2f> queryCoord;
    std::vector<cv::Point2f> objectCoord;
    for( unsigned int i = 0; i < matches.size(); i++){
        queryCoord.push_back((qKeypoints[matches[i].queryIdx]).pt);
        objectCoord.push_back((objKeypoints[matches[i].trainIdx]).pt);
    }
    // 使用自定义的函数显示匹配点对
    plotMatches(srcColorImage, dstColorImage, queryCoord, objectCoord);
    
    // 计算homography矩阵
    cv::Mat mask;
    std::vector<cv::Point2f> queryInliers;
    std::vector<cv::Point2f> sceneInliers;
    cv::Mat H = findFundamentalMat(queryCoord, objectCoord, mask, CV_FM_RANSAC);
    //Mat H = findHomography( queryCoord, objectCoord, CV_RANSAC, 10, mask);
    int inliers_cnt = 0, outliers_cnt = 0;
    for (int j = 0; j < mask.rows; j++){
        if (mask.at<uchar>(j) == 1){
            queryInliers.push_back(queryCoord[j]);
            sceneInliers.push_back(objectCoord[j]);
            inliers_cnt++;
        }else {
            outliers_cnt++;
        }
    }
    //显示剔除误配点对后的匹配点对
    plotMatches(srcColorImage, dstColorImage, queryInliers, sceneInliers);
}

/****************************************************************/

/*
 * Returns the (stacking of the) 2x2 matrix A that maps the unit circle
 * into the ellipses satisfying the equation x' inv(S) x = 1. Here S
 * is a stacked covariance matrix, with elements S11, S12 and S22.
 *
mat mapFromS(mat &S){
	tmp = sqrt(S(3,:)) + eps ;
	A(1,1) = sqrt(S(1,:).*S(3,:) - S(2,:).^2) ./ tmp ;
	A(2,1) = zeros(1,length(tmp));
	A(1,2) = S(2,:) ./ tmp ;
	A(2,2) = tmp ;
	return A;
}*/

/******************************************************************
 * 函数功能：几何校正需要调用的函数
 * 
 *
 */
arma::mat centering(arma::mat &x){
	arma::mat tmp = -mean(x.rows(0, 1), 1);
	arma::mat tmp2(2,2);
	tmp2.eye();
	arma::mat tmp1 = join_horiz(tmp2, tmp);
	arma::mat v;
	v << 0 << 0 << 1 << arma::endr;
	arma::mat T = join_vert(tmp1, v);
	//T.print("T =");
	arma::mat xm = T * x;
	//xm.print("xm =");
	
	//at least one pixel apart to avoid numerical problems
	//xm.print("xm =");
	double std11 = arma::stddev(xm.row(0));
	//cout << "std11:" << std11 << endl;
	double std22 = stddev(xm.row(1));
	//cout << "std22:" << std22 << endl;

    double std1 = std::max(std11, 1.0);
    double std2 = std::max(std22, 1.0);
	
	arma::mat S;
	S << 1/std1 << 0 << 0 << arma::endr
	  << 0 << 1/std2 << 0 << arma::endr
	  << 0 << 0 << 1 << arma::endr;
	arma::mat C = S * T ;
	//C.print("C =");
	return C;
}

/*******************************************************************
 * 函数功能：几何校正需要调用的函数
 * 
 *
 */
arma::mat toAffinity(arma::mat &f){
	arma::mat A;
	arma::mat v;
	v << 0 << 0 << 1 << arma::endr;
	int flag = f.n_rows;
	switch(flag){
		case 6:{ // oriented ellipses
			arma::mat T = f.rows(0, 1);
			arma::mat tmp = join_horiz(f.rows(2, 3), f.rows(4, 5));
			arma::mat tmp1 = join_horiz(tmp, T);
			A = join_vert(tmp1, v);
			break;}
		case 4:{   // oriented discs
			arma::mat T = f.rows(0, 1);
			double s = f.at(2,0);
			double th = f.at(3,0);
			arma::mat S = arma::randu<arma::mat>(2,2);
			/*S.at(0, 0) = s*cos(th);
			S.at(0, 1) = -s*sin(th);
			S.at(1, 0) = s*sin(th);
			S.at(1, 1) = s*cos(th);*/
			S << s*cos(th) << -s*sin(th) << arma::endr
			  << s*sin(th) << s*cos(th)  << arma::endr;
			arma::mat tmp1 = join_horiz(S, T);
			A = join_vert(tmp1, v);
			//A.print("A =");
			break;}
		/*case 3:{    // discs
			mat T = f.rows(0, 1);
			mat s = f.row(2);
			int th = 0 ;
			A = [s*[cos(th) -sin(th) ; sin(th) cos(th)], T ; 0 0 1] ;
			   }
		case 5:{ // ellipses
			mat T = f.rows(0, 1);
			A = [mapFromS(f(3:5)), T ; 0 0 1] ;
			   }*/
		default:
            std::cout << "出错啦！" << std::endl;
			break;
	}
	return A;
}

/******************************************************************
 * 函数功能：几何校正
 * 
 * 待写：H_final的值也应该返回去
 */
arma::uvec geometricVerification(const arma::mat &frames1, const arma::mat &frames2, 
	const arma::mat &matches, const superluOpts &opts){
	// 测试载入是否准确
    std::cout<< "element测试: " << " x: " << frames1(0,1) << " y: " << frames1(1,1) << std::endl;
    std::cout << " 行数： " << frames1.n_rows << " 列数：" << frames1.n_cols << std::endl;
    std::cout << "==========================================================" << std::endl;

	int numMatches = matches.n_cols;
	// 测试匹配数目是否准确
    std::cout << "没有RANSAC前匹配数目： " << numMatches << std::endl;
    std::cout << "==========================================================" << std::endl;

	arma::field<arma::uvec> inliers(1, numMatches);
	arma::field<arma::mat> H(1, numMatches);

	arma::uvec v = arma::linspace<arma::uvec>(0,1,2);
	arma::uvec matchedIndex_Query = arma::conv_to<arma::uvec>::from(matches.row(0)-1);
	arma::uvec matchedIndex_Object = arma::conv_to<arma::uvec>::from(matches.row(1)-1);

	arma::mat x1 = frames1(v, matchedIndex_Query) ;
	arma::mat x2 = frames2(v, matchedIndex_Object);
    std::cout << " x1查询图像匹配行数： " << x1.n_rows << " 查询图像匹配列数：" << x1.n_cols << std::endl;
    std::cout << " x2目标图像匹配行数： " << x2.n_rows << " 目标图像匹配列数：" << x2.n_cols << std::endl;
    std::cout<< "x1 element测试: " << " x: " << x1(0,1) << " y: " << x1(1,1) << std::endl;
    std::cout<< "x2 element测试: " << " x: " << x2(0,1) << " y: " << x2(1,1) << std::endl;
    std::cout << "==========================================================" << std::endl;

	arma::mat x1hom = arma::join_cols(x1, arma::ones<arma::mat>(1, numMatches));  //在下面添加一行，注意和join_rows的区别
	arma::mat x2hom = arma::join_cols(x2, arma::ones<arma::mat>(1, numMatches));
    std::cout << " x1hom查询图像匹配行数： " << x1hom.n_rows << " 查询图像匹配列数：" << x1hom.n_cols << std::endl;
    std::cout<< "x1hom element测试: " << " x: " << x1hom(0,1) << " y: " << x1hom(1,1) << " z: " << x1hom(2,1) << std::endl;
    std::cout << "==========================================================" << std::endl;

	arma::mat x1p, H21;  //作用域
	double tol;
	for(int m = 0; m < numMatches; ++m){
		//cout << "m: " << m << endl;
		for(unsigned int t = 0; t < opts.numRefinementIterations; ++t){
			//cout << "t: " << t << endl;
			if (t == 0){
				arma::mat tmp1 = frames1.col(matches(0, m)-1);
				arma::mat A1 = toAffinity(tmp1);
				//A1.print("A1 =");
				arma::mat tmp2 = frames2.col(matches(1, m)-1);
				arma::mat A2 = toAffinity(tmp2);
				//A2.print("A2 =");
				H21 = A2 * inv(A1);
				//H21.print("H21 =");
				x1p = H21.rows(0, 1) * x1hom ;
				//x1p.print("x1p =");
				tol = opts.tolerance1;
			}else if(t !=0 && t <= 3){
				arma::mat A1 = x1hom.cols(inliers(0, m));
				arma::mat A2 = x2.cols(inliers(0, m));
				//A1.print("A1 =");
				//A2.print("A2 =");
		        H21 = A2*pinv(A1);
				//H21.print("H21 =");
				x1p = H21.rows(0, 1) * x1hom ;
				//x1p.print("x1p =");
				arma::mat v;
				v << 0 << 0 << 1 << arma::endr;
				H21 = join_vert(H21, v);
				//H21.print("H21 =");
				//x1p.print("x1p =");
				tol = opts.tolerance2;
			}else{
				arma::mat x1in = x1hom.cols(inliers(0, m));
				arma::mat x2in = x2hom.cols(inliers(0, m));
				arma::mat S1 = centering(x1in);
				arma::mat S2 = centering(x2in);
				arma::mat x1c = S1 * x1in;
				//x1c.print("x1c =");
				arma::mat x2c = S2 * x2in;
				//x2c.print("x2c =");
				arma::mat A1 = arma::randu<arma::mat>(x1c.n_rows ,x1c.n_cols);
				A1.zeros();
				arma::mat A2 = arma::randu<arma::mat>(x1c.n_rows ,x1c.n_cols);
				A2.zeros();
				arma::mat A3 = arma::randu<arma::mat>(x1c.n_rows ,x1c.n_cols);
				A3.zeros();
				for(unsigned int i = 0; i < x1c.n_cols; ++i){
					A2.col(i) = x1c.col(i)*(-x2c.row(0).col(i));
					A3.col(i) = x1c.col(i)*(-x2c.row(1).col(i));
				}
				arma::mat T1 = join_cols(join_horiz(x1c, A1), join_horiz(A1, x1c));
				arma::mat T2 = join_cols(T1, join_horiz(A2, A3));
				//T2.print("T2 =");
				arma::mat U;
				arma::vec s;
				arma::mat V;
				svd_econ(U, s, V, T2);
				//U.print("U =");
				//V.print("V =");
				arma::vec tmm = U.col(U.n_cols-1);
				H21 = reshape(tmm, 3, 3).t();
				H21 = inv(S2) * H21 * S1;
				H21 = H21 / H21(H21.n_rows-1, H21.n_cols-1) ;
				//H21.print("H21 =");
				arma::mat x1phom = H21 * x1hom ;
				arma::mat cc1 = x1phom.row(0) / x1phom.row(2);
				arma::mat cc2 = x1phom.row(1) / x1phom.row(2);
				arma::mat x1p = join_cols(cc1, cc2);
				//x1p.print("x1p =");
				tol = opts.tolerance3;
			}
			arma::mat tmp = arma::square(x2 - x1p); //精度跟matlab相比更高？
			//tmp.print("tmp =");
			arma::mat dist2 = tmp.row(0) + tmp.row(1);
			//dist2.print("dist2 =");
			inliers(0, m) = arma::find(dist2 < pow(tol, 2));
			H(0, m) = H21;
			//H(0, m).print("H(0, m) =");
			//inliers(0, m).print("inliers(0, m) =");
			//cout << inliers(0, m).size() << endl;
			//cout << "==========================================================" << endl;
			if (inliers(0, m).size() < opts.minInliers) break;
			if (inliers(0, m).size() > 0.7 * numMatches) break;
		}
	}
	arma::uvec scores(numMatches);
	for(int i = 0; i < numMatches; ++i){
		scores.at(i) = inliers(0, i).n_rows;
	}
	//scores.print("scores = ");
	arma::uword index;
	scores.max(index);
	//cout << index << endl;
	arma::mat H_final = inv(H(0, index));
	H_final.print("H_final = ");
	arma::uvec inliers_final = inliers(0, index);
	//inliers_final.print("inliers_final = ");
	return inliers_final;
}
