#include "../include/eye_center.h"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <stdio.h>
#include <time.h>       // clock_t, clock, CLOCKS_PER_SEC

using namespace std;
//using namespace cv;

// GLOBAL VARIABLES

const int fastSize_width = 40;
const int DARKNESS_WEIGHT_SCALE = 100;

cv::Mat get_centermap(cv::Mat& eye_grey) {

    // Calculate image gradients
    cv::Mat grad_x, grad_y;
    Sobel(eye_grey, grad_x, CV_32F, 1, 0, 5);
    Sobel(eye_grey, grad_y, CV_32F, 0, 1, 5);

    // Get magnitudes of gradients, and calculate thresh
    cv::Mat mags;
    cv::Scalar mean, stddev;
    magnitude(grad_x, grad_y, mags);
    meanStdDev(mags, mean, stddev);
    int mag_thresh = stddev.val[0] / 2 + mean.val[0];

    // Threshold out gradients with mags which are too low
    grad_x.setTo(0, mags < mag_thresh);
    grad_y.setTo(0, mags < mag_thresh);

    // Normalize gradients
    grad_x = grad_x / (mags+1); // (+1 is hack to guard against div by 0)
    grad_y = grad_y / (mags+1);

    // Initialize 1d vectors of x and y indicies of Mat
    vector<int> x_inds_vec, y_inds_vec;
    for(int i = 0; i < eye_grey.size().width; i++)
        x_inds_vec.push_back(i);
    for(int i = 0; i < eye_grey.size().height; i++)
        y_inds_vec.push_back(i);

    // Repeat vectors to form indices Mats
    cv::Mat x_inds(x_inds_vec), y_inds(y_inds_vec);
    x_inds = repeat(x_inds.t(), eye_grey.size().height, 1);
    y_inds = repeat(y_inds, 1, eye_grey.size().width);
    x_inds.convertTo(x_inds, CV_32F);	// Has to be float for arith. with dx, dy
    y_inds.convertTo(y_inds, CV_32F);

    // Set-up Mats for main loop
    cv::Mat ones = cv::Mat::ones(x_inds.rows, x_inds.cols, CV_32F);	// for re-use with creating normalized disp. vecs
    cv::Mat darkness_weights = (255 - eye_grey) / DARKNESS_WEIGHT_SCALE;
    cv::Mat accumulator = cv::Mat::zeros(eye_grey.size(), CV_32F);
    cv::Mat diffs, dx, dy;

    // Loop over all pixels, testing each as a possible center
    for(int y = 0; y < eye_grey.rows; ++y) {

        // Get pointers for each row
        float* grd_x_p = grad_x.ptr<float>(y);
        float* grd_y_p = grad_y.ptr<float>(y);
        uchar* d_w_p = darkness_weights.ptr<uchar>(y);

        for(int x = 0; x < eye_grey.cols; ++x) {

            // Deref and increment pointers
            float grad_x_val = *grd_x_p++;
            float grad_y_val = *grd_y_p++;

            // Skip if no gradient
            if(grad_x_val == 0 && grad_y_val == 0)
                continue;

            dx = ones * x - x_inds;
            dy = ones * y - y_inds;

            magnitude(dx, dy, mags);
            dx = dx / mags;
            dy = dy / mags;

            diffs = (dx * grad_x_val + dy * grad_y_val) * *d_w_p++;
            diffs.setTo(0, diffs < 0);

            accumulator = accumulator + diffs;
        }
    }

    // Normalize and convert accumulator
    accumulator = accumulator / eye_grey.total();
    normalize(accumulator, accumulator, 0, 255, cv::NORM_MINMAX);
    accumulator.convertTo(accumulator, CV_8U);

    return accumulator;
}

cv::Point find_eye_center(cv::Mat& eye_grey){

    // Convert BGR coarse ROI to gray
    cv::Mat eye_grey_small;
    //cv::Mat eye_grey, eye_grey_small;
    //cvtColor(eye_bgr, eye_grey, CV_BGR2GRAY);

    // Resize the image to a constant fast size
    // TODO: prevent upscaling
    float scale = fastSize_width / (float)eye_grey.size().width;
    resize(eye_grey, eye_grey_small, cv::Size(0,0), scale, scale);
    cv::GaussianBlur(eye_grey_small, eye_grey_small, cv::Size(5,5),0);

    // Create centermap
    cv::Mat centermap = get_centermap(eye_grey_small);

    // Find position of max value in small-size centermap
    cv::Point maxLoc;
    minMaxLoc(centermap, NULL, NULL, NULL, &maxLoc);

    // Return re-scaled center to full size
    return maxLoc * (1/scale);
}

void test_centermap() {

    // Load test image and convert to gray
    cv::Mat eye_bgr, eye_grey, eye_grey_small;
    eye_bgr = cv::imread("/Users/willard/data/small/erroll1_l.png", cv::IMREAD_COLOR);

    cvtColor(eye_bgr, eye_grey, cv::COLOR_BGR2GRAY);

    // Find eye-center 10 times, and find equivalent FPS
    int reps = 10;
    clock_t start = clock();
    cv::Point maxLoc;
    for (int i = 0; i < 10; i++)
        maxLoc = find_eye_center(eye_grey);

    float t = ( ((float)clock()-start) / CLOCKS_PER_SEC );
    printf ("Find eye-center: %f FPS\n", 1/(t/reps));

    // Show output
    cv::circle(eye_bgr, maxLoc , 4, cv::Scalar(0, 0, 255), -1);
    cv::namedWindow("centermap");
    cv::imshow("centermap", eye_bgr);

    cv::waitKey();
}
