#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/photo/photo.hpp>

#include <iostream>     // cout
#include <stdio.h>
#include <time.h>       // clock_t, clock, CLOCKS_PER_SEC

using namespace std;

const cv::Mat ERASE_SPEC_KERNEL = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));

//void erase_specular(Mat& eye_bgr) {
void erase_specular(cv::Mat& eye_grey) {
    
    // Rather arbitrary decision on how large a specularity may be
    //int max_spec_contour_area = (eye_bgr.size().width + eye_bgr.size().height)/2;
    int max_spec_contour_area = (eye_grey.size().width + eye_grey.size().height)/2;
    
    // Convert BGR coarse ROI to gray, blur it slightly to reduce noise
    //Mat eye_grey;
    //cvtColor(eye_bgr, eye_grey, CV_BGR2GRAY);
    GaussianBlur(eye_grey, eye_grey, cv::Size(5, 5), 0);
    
    // Close to suppress eyelashes
    morphologyEx(eye_grey, eye_grey, cv::MORPH_CLOSE, ERASE_SPEC_KERNEL);
    
    // Compute thresh value (using of highest and lowest pixel values)
    double m, M; // m(in) and (M)ax values in image
    minMaxLoc(eye_grey, &m, &M, NULL, NULL);
    double thresh = (m + M) * 3/4;
    
    // Threshold the image
    cv::Mat eye_thresh;
    threshold(eye_grey, eye_thresh, thresh, 255, cv::THRESH_BINARY);
    
    // Find all contours in threshed image (possible specularities)
    vector< vector<cv::Point> > all_contours, contours;
    findContours(eye_thresh, all_contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    
    // Only save small ones (assumed to be spec.s)
    for (int i=0; i<all_contours.size(); i++){
        if( contourArea(all_contours[i]) < max_spec_contour_area )
            contours.push_back(all_contours[i]);
    }
    
    // Draw the contours into an inpaint mask
    cv::Mat small_contours_mask = cv::Mat::zeros(eye_grey.size(), eye_grey.type());
    drawContours(small_contours_mask, contours, -1, 255, -1);
    dilate(small_contours_mask, small_contours_mask, ERASE_SPEC_KERNEL);
    
    // Inpaint within contour bounds
    inpaint(eye_grey, small_contours_mask, eye_grey, 2, cv::INPAINT_TELEA);
}

void test_erase_specular() {
    
    // Load test image and convert to gray
    cv::Mat eye_bgr, eye_grey, eye_grey_small;
    eye_bgr = cv::imread("/Users/willard/data/small/erroll1_l.png", cv::IMREAD_COLOR);
    
    cvtColor(eye_bgr, eye_grey, cv::COLOR_BGRA2GRAY);
    
    cv::namedWindow("origion gray image");
    cv::imshow("origion gray image", eye_grey);
    
    erase_specular(eye_grey);
    
    cv::namedWindow("erase specular");
    cv::imshow("erase specular", eye_grey);
    
    cv::waitKey();
}
