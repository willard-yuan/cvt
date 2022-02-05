#include "../include/motion_area_detection.h"
#include <glog/logging.h>

namespace cvtk {
namespace detect {
MotionAreaDetecion::MotionAreaDetecion() {
  pBackSub = cv::createBackgroundSubtractorMOG2();
  kernel = cv::Mat(3, 3, CV_8U, cv::Scalar(1));
  imgScale = 0.4;
  thresh = 70;
  maxValue = 255;
  motionAreaRatio = 0.25;
  motionAreaRatioLow = 0.14;
}
        
bool MotionAreaDetecion::invalidWidthAndHeight(const cv::Mat& mat) {
  int max = std::max(mat.cols, mat.rows);
  int min = std::min(mat.cols, mat.rows);
  return min == 0 ? true: (max / min) >= 20;
}

int MotionAreaDetecion::findMaxBBox(cv::Mat& outMask, cv::Rect& maxBBox) {
  int largest_area = 0;
  std::vector<std::vector<cv::Point> > contours;
  cv::findContours(outMask, contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
  for(int i = 0; i< contours.size(); i++) {
    double area = cv::contourArea(contours[i]);
    if(area > largest_area) {
      largest_area = area;
      maxBBox = cv::boundingRect(contours[i]);
    }
  }
  return 1;
}

int MotionAreaDetecion::findTopKBBox(cv::Mat& outMask, std::vector<cv::Rect>& BBoxs) {
    std::vector<std::vector<cv::Point> > contours;
    std::vector<float> areas;
    std::vector<cv::Rect> tmpBBoxs;
    cv::findContours(outMask, contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
    for(int i = 0; i< contours.size(); i++)
    {
        float area = cv::contourArea(contours[i]);
        cv::Rect BBox = cv::boundingRect(contours[i]);
        tmpBBoxs.push_back(BBox);
        areas.push_back(area);
    }

    std::vector<size_t> idx(areas.size());
    std::iota(idx.begin(), idx.end(), 0);
    stable_sort(idx.begin(), idx.end(),
                [&areas](size_t i1, size_t i2) {return areas[i1] >= areas[i2];});

    for (int i = 0; i < areas.size(); ++i) {
        BBoxs.push_back(tmpBBoxs[idx[i]]);
    }

    return 1;
}

int MotionAreaDetecion::globalThreshold(cv::Mat &image, cv::Mat &imageThresd, int setThresh) {
  if(image.empty()) {
    return 0;
  }
  cv::threshold(image, imageThresd, setThresh, maxValue, cv::THRESH_BINARY);
  return 1;
}
int MotionAreaDetecion::PipDetect(const Video& video,
                                         ::cvtk::img::MotionArea* motionLocInfo) {
  float prob_detect = 0.0;
  int numFrames = int(video.frames.size());
  int globalUpLines = 0;
  int globalDownLines = 0;

  cv::Mat outMaskGuaissX, outMaskGuaissY, outMaskGuaiss, outMaskGuaissLow;
  cv::Mat gradX, gradY;
  cv::Mat absGradX, absGradY;
  cv::Mat frameGray;

  cv::Mat frameCrop, frameMask, openedMask;
  cv::Mat frame = cv::imdecode(video.frames[0].image, 1);

  motionLocInfo->set_video_width(frame.cols);
  motionLocInfo->set_video_height(frame.rows);

  motionLocInfo->set_left(0);
  motionLocInfo->set_right((frame.cols-1));
  motionLocInfo->set_top(0);
  motionLocInfo->set_bottom((frame.rows-1));
  
  bool status = invalidWidthAndHeight(frame);
  if (status) {
      return 0;
  }
  
  int frameWidth = frame.cols;
  int frameHeight = frame.rows;
  cv::Mat frameMaskSum = cv::Mat::zeros(int(imgScale*frameHeight), int(imgScale*frameWidth),
                                        CV_32F);
  cv::resize(frame, frameCrop, cv::Size(int(imgScale*frameWidth), int(imgScale*frameHeight)));
  cv::Mat frameMaskGuaissSumX = cv::Mat::zeros(int(imgScale*frameHeight), int(imgScale*frameWidth), CV_32F);
  cv::Mat frameMaskGuaissSumY = cv::Mat::zeros(int(imgScale*frameHeight), int(imgScale*frameWidth), CV_32F);
  
  pBackSub->apply(frameCrop, frameMask);
  cv::morphologyEx(frameMask, openedMask, cv::MORPH_OPEN, kernel);
  frameMask.convertTo(frameMask, CV_32F);
  frameMaskSum = frameMaskSum + frameMask;
  
  std::vector<float> xlines;
  std::vector<float> ylines;

  for(int i = 1; i < numFrames; i++) {
      frame = cv::imdecode(video.frames[i].image, 1);
      bool status = invalidWidthAndHeight(frame);
      if (status) {
          continue;
      }
      cv::resize(frame, frameCrop, cv::Size(int(imgScale*frameWidth), int(imgScale*frameHeight)));

      cv::Mat cannyEdgeDetecionFrame = cv::Mat::zeros(frame.rows, frame.cols, CV_8U);
      cv::Mat image;
      frame.copyTo(image);
      cv::Mat edgeDetectorOutputFrame = cannyEdgeDetector.detectEdges(image, cannyEdgeDetecionFrame);
      std::string frameId = "123_1";
      LineDetectInfo tmpInfo = houghTransform.detectLines(edgeDetectorOutputFrame, frameId);

      for (int j = 0; j < tmpInfo.xSelectcoords.size(); j++) {
          xlines.push_back(tmpInfo.xSelectcoords[j]);
      }

      for (int j = 0; j < tmpInfo.ySelectcoords.size(); j++) {
          ylines.push_back(tmpInfo.ySelectcoords[j]);
      }

      pBackSub->apply(frameCrop, frameMask);
      cv::morphologyEx(frameMask, openedMask, cv::MORPH_OPEN, kernel);
      frameMask.convertTo(frameMask, CV_32F);
      frameMaskSum = frameMaskSum + frameMask;

      // 高斯模式视频
      cv::cvtColor(frameCrop, frameGray, cv::COLOR_RGB2GRAY);
      cv::GaussianBlur(frameGray, frameGray, cv::Size(3,3), 0, 0);
      cv::Sobel(frameGray, gradX, CV_16S, 1, 0, 3, 1, 1, cv::BORDER_DEFAULT);
      cv::convertScaleAbs(gradX, absGradX);
      cv::Sobel(frameGray, gradY,CV_16S, 0, 1, 3, 1, 1, cv::BORDER_DEFAULT);
      cv::convertScaleAbs(gradY, absGradY);
      status = globalThreshold(absGradX, outMaskGuaissX);
      outMaskGuaissX.convertTo(outMaskGuaissX, CV_32F);
      frameMaskGuaissSumX = frameMaskGuaissSumX + outMaskGuaissX;
      status = globalThreshold(absGradY, outMaskGuaissY);
      outMaskGuaissY.convertTo(outMaskGuaissY, CV_32F);
      frameMaskGuaissSumY = frameMaskGuaissSumY + outMaskGuaissY;
  }

  frameMaskGuaissSumX = frameMaskGuaissSumX/(numFrames-1);
  frameMaskGuaissSumX.convertTo(frameMaskGuaissSumX, CV_8U);
  cv::morphologyEx(frameMaskGuaissSumX, frameMaskGuaissSumX, cv::MORPH_CLOSE, kernel);
  frameMaskGuaissSumY = frameMaskGuaissSumY/(numFrames-1);
  frameMaskGuaissSumY.convertTo(frameMaskGuaissSumY, CV_8U);
  cv::morphologyEx(frameMaskGuaissSumY, frameMaskGuaissSumY, cv::MORPH_CLOSE, kernel);
  cv::Mat frameMaskGuaissSum = frameMaskGuaissSumX + frameMaskGuaissSumY;
  cv::morphologyEx(frameMaskGuaissSum, frameMaskGuaissSum, cv::MORPH_CLOSE, kernel);

  int upLine = 0;
  int downLine = frameMaskSum.rows;
  float upLineProb = 0;
  float downLineProb = 0;
  if (ylines.size() >= 2) {
      cv::Mat labels,centers;
      int K=2, attempts=10, flags=cv::KMEANS_RANDOM_CENTERS;
      cv::TermCriteria tc;
      cv::kmeans(ylines, K, labels, tc, attempts, flags, centers);
      int numUpCount = 0;
      int numDownCount = 0;
      upLine = std::min(centers.at<float>( 0, 0  ), centers.at<float>( 1, 0  ));
      downLine = std::max(centers.at<float>( 0, 0  ), centers.at<float>( 1, 0  ));
      for (int i = 0; i < ylines.size(); i++) {
          if (std::abs(ylines[i] - upLine) <= 2){
              numUpCount = numUpCount + 1;
          }
          if (std::abs(ylines[i] - downLine) <= 2){
              numDownCount = numDownCount + 1;
          }
      }
      upLineProb = numUpCount/float(numFrames-1);
      downLineProb = numDownCount/float(numFrames-1);
  }

  frameMaskSum = frameMaskSum/numFrames;
  frameMaskSum.convertTo(frameMaskSum, CV_8U);
  cv::morphologyEx(frameMaskSum, frameMaskSum, cv::MORPH_CLOSE, kernel);

  cv::Rect maxBBox, maxBBoxLow;
  cv::Mat outMask;
  cv::Mat outMaskLow;

  status = globalThreshold(frameMaskSum, outMask, 70);
  status = findMaxBBox(outMask, maxBBox);

  int y1 = int(maxBBox.y/(1.0*imgScale));
  int y2 = int((maxBBox.y + maxBBox.height)/(1.0*imgScale));
  int numTopLines = 0;
  int numDownLines = 0;
  for (int i = 0; i < ylines.size(); i++) {
    if(std::abs(ylines[i] - y1) <= 3) {
      ++numTopLines;
    }
    if(std::abs(ylines[i] - y2) <= 3) {
      ++numDownLines;
    }
  }
  status = globalThreshold(frameMaskSum, outMaskLow, 30);
  status = findMaxBBox(outMaskLow, maxBBoxLow);
  y1 = int(maxBBoxLow.y/(1.0*imgScale));
  y2 = int((maxBBoxLow.y + maxBBoxLow.height)/(1.0*imgScale));
  int numTopLinesLow = 0;
  int numDownLinesLow = 0;
  for(int i = 0; i < ylines.size(); i++) {
    if(std::abs(ylines[i] - y1) <= 3) {
      ++numTopLinesLow;
    }
    if(std::abs(ylines[i] - y2) <= 3) {
      ++numDownLinesLow;
    }
  }

  // 高斯模糊视频
  cv::Rect maxBBoxGuaiss, maxBBoxGuaissLow;
  status = globalThreshold(frameMaskGuaissSum, outMaskGuaiss, 70);
  status = findMaxBBox(outMaskGuaiss, maxBBoxGuaiss);
  y1 = int(maxBBoxGuaiss.y/(1.0*imgScale));
  y2 = int((maxBBoxGuaiss.y + maxBBoxGuaiss.height)/(1.0*imgScale));
  int numTopLinesGuaiss = 0;
  int numDownLinesGuaiss = 0;
  for(int i = 0; i < ylines.size(); i++) {
      if(std::abs(ylines[i] - y1) <= 10) {
          ++numTopLinesGuaiss;
      }
      if(std::abs(ylines[i] - y2) <= 10) {
          ++numDownLinesGuaiss;
      }
  }

  status = globalThreshold(frameMaskGuaissSum, outMaskGuaissLow, 30);
  status = findMaxBBox(outMaskGuaissLow, maxBBoxGuaissLow);
  y1 = int(maxBBoxGuaissLow.y/(1.0*imgScale));
  y2 = int((maxBBoxGuaissLow.y + maxBBoxGuaissLow.height)/(1.0*imgScale));
  int numTopLinesGuaissLow = 0;
  int numDownLinesGuaissLow = 0;
  for(int i = 0; i < ylines.size(); i++) {
      if(std::abs(ylines[i] - y1) <= 15) {
          ++numTopLinesGuaissLow;
      }
      if(std::abs(ylines[i] - y2) <= 15) {
          ++numDownLinesGuaissLow;
      }
  }

  if(numDownLinesLow>=1 && numTopLinesLow>=1 && (numTopLines<=0 || numDownLines<=0)) {
    maxBBox = maxBBoxLow;
    outMask = outMaskLow;
  }
  int selectGuaissFlag = 0;
  if((numTopLinesGuaiss>=1 && numDownLinesGuaiss>=1 && (numTopLines<=0 || numDownLines<=0))
          || (numTopLinesGuaiss>=1 && numDownLinesGuaiss>=1 && (numTopLinesLow<=0 || numDownLinesLow<=0))) {
      maxBBox = maxBBoxGuaiss;
      outMask = outMaskGuaiss;
      selectGuaissFlag = 1;
  }
  if((numTopLinesGuaissLow>=1 && numDownLinesGuaissLow>=1 && (numTopLines<=0 || numDownLines<=0))
          || (numTopLinesGuaissLow>=1 && numDownLinesGuaissLow>=1 && (numTopLinesLow<=0 || numDownLinesLow<=0))) {
      maxBBox = maxBBoxGuaissLow;
      outMask = outMaskGuaissLow;
      selectGuaissFlag = 1;
  }

  if(upLineProb>=0.6 && downLineProb>=0.6 && (numTopLines<=0 
          || numDownLines<=0) && int(downLine-upLine)>=int(outMask.rows/4.0)) {
    maxBBox = cv::Rect(0, int(upLine*imgScale), outMask.cols, int(int(downLine-upLine)*imgScale));
    outMask = cv::Mat::zeros(outMask.rows, outMask.cols, CV_8U);
    cv::Mat tmpOnes = 255*cv::Mat::ones(int((downLine-upLine)*imgScale), outMask.cols, CV_8U);
    cv::Mat destRoi = outMask(maxBBox);
    tmpOnes.copyTo(destRoi);
  }

  // 上下三格，保留中间
  int midConfidenceFlag = 0;
  std::vector<cv::Rect> lowBBoxs;
  status = findTopKBBox(outMaskLow, lowBBoxs);
  for(int i = 0; i < lowBBoxs.size() && i < 3; ++i) {
      if(lowBBoxs[i].width*lowBBoxs[i].height <= 100
              || lowBBoxs[i].width < 50 || lowBBoxs[i].height < 50) {
          continue;
      }
      y1 = int(lowBBoxs[i].y/(1.0*imgScale));
      y2 = int((lowBBoxs[i].y + lowBBoxs[i].height)/(1.0*imgScale));
      int numTopLinesLow = 0;
      int numDownLinesLow = 0;
      for(int i = 0; i < ylines.size(); i++) {
          if(std::abs(ylines[i] - y1) <= 10) {
              ++numTopLinesLow;
              ++globalUpLines;
          }
          if(std::abs(ylines[i] - y2) <= 10) {
              ++numDownLinesLow;
              ++globalDownLines;
          }
      }
      if(numTopLinesLow >= (0.4*numFrames) && numDownLinesLow >= (0.4*numFrames)
              && int(y2-y1)>=int(outMaskLow.rows/(imgScale*6.0))) {
          maxBBox = lowBBoxs[i];
          midConfidenceFlag = 1;
      }
  }

  std::vector<cv::Rect> BBoxs;
  status = findTopKBBox(outMask, BBoxs);
  for(int i = 0; i < BBoxs.size() && i < 3; ++i) {
      if(BBoxs[i].width*BBoxs[i].height <= 100
              || BBoxs[i].width < 50 || BBoxs[i].height < 50) {
          continue;
      }
      y1 = int(BBoxs[i].y/(1.0*imgScale));
      y2 = int((BBoxs[i].y + BBoxs[i].height)/(1.0*imgScale));
      int numTopLines = 0;
      int numDownLines = 0;
      for(int i = 0; i < ylines.size(); i++) {
          if(std::abs(ylines[i] - y1) <= 10) {
              ++numTopLines;
              ++globalUpLines;
          }
          if(std::abs(ylines[i] - y2) <= 10) {
              ++numDownLines;
              ++globalDownLines;
          }
      }
      if(numTopLines >= (0.4*numFrames) && numDownLines >= (0.4*numFrames)
              && int(y2-y1)>=int(outMask.rows/(imgScale*6.0))) {
          if(midConfidenceFlag == 1 &&
                  std::abs(maxBBox.x - BBoxs[i].x) <= 5 &&
                  std::abs(maxBBox.y - BBoxs[i].y) <= 5 &&
                  std::abs(maxBBox.width - BBoxs[i].width) <= 5 &&
                  std::abs(maxBBox.height - BBoxs[i].height) <= 5){
              midConfidenceFlag = 1;
              break;
          }
          maxBBox = BBoxs[i];
          midConfidenceFlag = 1;
      }
  }

  int upConfidenceFlag = 0;
  int downConfidenceFlag = 0;
  if (std::abs(maxBBox.y/imgScale - upLine) <= 3 && upLineProb >= 0.4) {
      upConfidenceFlag = 1;
  }
  if(std::abs(maxBBox.y/imgScale + maxBBox.height/imgScale - downLine) <= 3 && downLineProb >= 0.4) {
      downConfidenceFlag = 1;
  }

  int boxArea = (maxBBox.width*maxBBox.height);
  int tmp_left = int(std::max((double)maxBBox.x/imgScale, (double)(0.0)));
  int tmp_right = int(std::min((double)(maxBBox.x + maxBBox.width)/imgScale,
                               (double)(frame.cols-1)));
  int tmp_top = int(std::max((double)maxBBox.y/imgScale, (double)(0.0)));
  int tmp_bottom = int(std::min((double)(maxBBox.y + maxBBox.height)/imgScale,
                                (double)(frame.rows-1)));

  int count_num = 0;
  int tmp_left_outmask = int(std::max((double)maxBBox.x, (double)(0.0)));
  int tmp_right_outmask = int(std::min((double)(maxBBox.x + maxBBox.width),
                                       (double)(outMask.cols-1)));
  int tmp_top_outmask = int(std::max((double)maxBBox.y, (double)(0.0)));
  int tmp_bottom_outmask = int(std::min((double)(maxBBox.y + maxBBox.height),
                                        (double)(outMask.rows-1)));
  for(int i = tmp_top_outmask; i < tmp_bottom_outmask; i++) {
    for(int j = tmp_left_outmask; j < tmp_right_outmask; j++) {
      int tmp = (int)outMask.at<uchar>(i,j);
      if(tmp>128) {
        ++count_num;
      }
    }
  }

  // 修正高斯模糊视频偏移
  int upGuaissConfidenceFlag = 0;
  int downGuaissConfidenceFlag = 0;
  std::map<float, int> ylinesCountsMap;
  for (auto & elem : ylines)
  {
      auto result = ylinesCountsMap.insert(std::pair<float, int>(elem, 1));
      if (result.second == false)
          result.first->second++;
  }
  std::vector<std::pair<float, int>> ylinesCountsVec;
  for (auto itr = ylinesCountsMap.begin(); itr != ylinesCountsMap.end(); ++itr)
      ylinesCountsVec.push_back(*itr);
  std::sort(ylinesCountsVec.begin(), ylinesCountsVec.end(), [=](std::pair<float, int>& a, std::pair<float, int>& b) {
          return a.second > b.second;
          }
          );
  for (int i = 0; i < ylinesCountsVec.size();i++)
  {
      if (ylinesCountsVec[i].second >= 1 && selectGuaissFlag == 1)
      {
          if (std::abs(tmp_top-ylinesCountsVec[i].first) <= 10) {
              tmp_top = ylinesCountsVec[i].first;
              if(ylinesCountsVec[i].second >= (0.2*numFrames)) {
                  upGuaissConfidenceFlag = 1;
              }
              break;
          }
      }
  }
  for (int i = 0; i < ylinesCountsVec.size();i++)
  {
      if (ylinesCountsVec[i].second >= 1 && selectGuaissFlag == 1)
      {
          if (std::abs(tmp_bottom-ylinesCountsVec[i].first) <= 10) {
              tmp_bottom = ylinesCountsVec[i].first;
              if(ylinesCountsVec[i].second >= (0.2*numFrames)) {
                  downGuaissConfidenceFlag = 1;
              }
              break;
          }
      } 
  }

  // 修正没有命中水平线的情况
  int topConfidenceFlag = 0, bottomConfidenceFlag = 0;
  int bottomCount = 0, topCount = 0;
  for (int i = 0; i < ylinesCountsVec.size();i++)
  {
    if (ylinesCountsVec[i].second >= 1 && tmp_top > 10 && tmp_bottom < (frameHeight-10))
    {
        if (std::abs(tmp_top-ylinesCountsVec[i].first) <= 5) {
            ++topCount;
        }
        if (std::abs(tmp_bottom-ylinesCountsVec[i].first) <= 5) {
            ++bottomCount;
        }
    }
  }

  int topTmpCount = 0, bottomTmpCount = 0;
  for(int i = 0; i < ylines.size(); i++) {
      if (std::abs(tmp_top-ylines[i]) <= 15 && tmp_top > 10) {
          ++topTmpCount;
      }
      if (std::abs(tmp_bottom-ylines[i]) <= 15 && tmp_bottom < (frameHeight-10)) {
          ++bottomTmpCount;
      }
  }
  if(topTmpCount <= 0) {
      tmp_top = 0;
  }
  if(bottomTmpCount <= 0) {
      tmp_bottom = frame.rows-1;
  }

  if(bottomCount < 1 && topCount > 0) 
  {
    for (int i = 0; i < ylinesCountsVec.size();i++)
    {
      if (ylinesCountsVec[i].second >= 1 && ylinesCountsVec[i].first < tmp_bottom-100
              && (std::abs(ylinesCountsVec[i].first - tmp_top) < 15))
      {
        tmp_top = ylinesCountsVec[i].first;
        if(ylinesCountsVec[i].second >= (0.2*numFrames)) 
        {
          topConfidenceFlag = 1;
        }
        break;
      }
    }
  }
  if(bottomCount > 0 && topCount < 1) 
  {
    for (int i = 0; i < ylinesCountsVec.size();i++)
    {
      if (ylinesCountsVec[i].second >= 1 && ylinesCountsVec[i].first > tmp_top+100
              && (std::abs(ylinesCountsVec[i].first - tmp_bottom) < 15))
      {
        tmp_bottom = ylinesCountsVec[i].first;
        if(ylinesCountsVec[i].second >= (0.2*numFrames)) 
        {
          bottomConfidenceFlag = 1;
        }
        break;
      }
    }
  }

  // 竖直方向检测
  int leftCount = 0, rightCount = 0;
  for(int i = 0; i < xlines.size(); i++) {
      if (std::abs(tmp_left-xlines[i]) <= 15) {
          ++leftCount;
      }
      if (std::abs(tmp_right-xlines[i]) <= 15) {
          ++rightCount;
      }
  }

  prob_detect = 1.0*count_num/boxArea;
  if(upConfidenceFlag >=1 && downConfidenceFlag >= 1) {
      prob_detect = prob_detect + 0.4;
  }
  if(midConfidenceFlag >= 1) {
      prob_detect = prob_detect + 0.4;
  }
  if(upGuaissConfidenceFlag >= 1 && downGuaissConfidenceFlag >= 1) {
      prob_detect = prob_detect + 0.4;
  }
  if(topConfidenceFlag >= 1 && bottomConfidenceFlag >= 1) {
      prob_detect = prob_detect + 0.4;
  }
  if(leftCount <= 0) {
      tmp_left = 0;
  }
  if(rightCount <= 0) {
      tmp_right = frame.cols-1;
  }

  if(std::abs(tmp_bottom-tmp_top) < 0.14*frameHeight) {
      tmp_top = 0;
      tmp_bottom = frameHeight - 1;
      prob_detect = 0.0;
  }

  if(std::abs(tmp_right-tmp_left) < 0.1*frameWidth) {
      tmp_left = 0;
      tmp_right = frameWidth - 1;
  }

  prob_detect = std::min((double)prob_detect, 1.0);

  if (boxArea > (motionAreaRatio*outMask.rows*outMask.cols && (globalUpLines>=1 || globalDownLines>=1))) {
    motionLocInfo->set_left(tmp_left);
    motionLocInfo->set_right(tmp_right);
    motionLocInfo->set_top(tmp_top);
    motionLocInfo->set_bottom(tmp_bottom);
    motionLocInfo->set_prob_detect(prob_detect);
    return 1;
  } else if(boxArea > (motionAreaRatioLow*outMask.rows * outMask.cols) &&
            boxArea <= (motionAreaRatio*outMask.rows*outMask.cols) && (globalUpLines>=1 || globalDownLines>=1)) {
    motionLocInfo->set_left(tmp_left);
    motionLocInfo->set_right(tmp_right);
    motionLocInfo->set_top(tmp_top);
    motionLocInfo->set_bottom(tmp_bottom);
    motionLocInfo->set_prob_detect(prob_detect);
    return 1;
  }

  return 0;
}
        
int MotionAreaDetecion::MotionAreaDetect(const Video& video,
                                         ::cvtk::img::MotionArea* motionLocInfo) {
  float prob_detect = 0.0;
  int numFrames = int(video.frames.size());
  int globalUpLines = 0;
  int globalDownLines = 0;
            
  cv::Mat frameCrop, frameMask, openedMask;
  cv::Mat frame = cv::imdecode(video.frames[0].image, 1);

  motionLocInfo->set_video_width(frame.cols);
  motionLocInfo->set_video_height(frame.rows);

  motionLocInfo->set_left(0);
  motionLocInfo->set_right((frame.cols-1));
  motionLocInfo->set_top(0);
  motionLocInfo->set_bottom((frame.rows-1));
  
  bool status = invalidWidthAndHeight(frame);
  if (status) {
      return 0;
  }
  
  int frameWidth = frame.cols;
  int frameHeight = frame.rows;
  cv::Mat frameMaskSum = cv::Mat::zeros(int(imgScale*frameHeight), int(imgScale*frameWidth),
                                        CV_32F);
  cv::resize(frame, frameCrop, cv::Size(int(imgScale*frameWidth), int(imgScale*frameHeight)));
  
  pBackSub->apply(frameCrop, frameMask);
  cv::morphologyEx(frameMask, openedMask, cv::MORPH_OPEN, kernel);
  frameMask.convertTo(frameMask, CV_32F);
  frameMaskSum = frameMaskSum + frameMask;
  
  std::vector<float> xlines;
  std::vector<float> ylines;

  for(int i = 1; i < numFrames; i++) {
      frame = cv::imdecode(video.frames[i].image, 1);
      bool status = invalidWidthAndHeight(frame);
      if (status) {
          continue;
      }
      cv::resize(frame, frameCrop, cv::Size(int(imgScale*frameWidth), int(imgScale*frameHeight)));

      cv::Mat cannyEdgeDetecionFrame = cv::Mat::zeros(frame.rows, frame.cols, CV_8U);
      cv::Mat image;
      frame.copyTo(image);
      cv::Mat edgeDetectorOutputFrame = cannyEdgeDetector.detectEdges(image, cannyEdgeDetecionFrame);
      std::string frameId = "123_1";
      LineDetectInfo tmpInfo = houghTransform.detectLines(edgeDetectorOutputFrame, frameId);

      for (int j = 0; j < tmpInfo.xSelectcoords.size(); j++) {
          xlines.push_back(tmpInfo.xSelectcoords[j]);
      }

      for (int j = 0; j < tmpInfo.ySelectcoords.size(); j++) {
          ylines.push_back(tmpInfo.ySelectcoords[j]);
      }

      pBackSub->apply(frameCrop, frameMask);
      cv::morphologyEx(frameMask, openedMask, cv::MORPH_OPEN, kernel);
      frameMask.convertTo(frameMask, CV_32F);
      frameMaskSum = frameMaskSum + frameMask;
  }

  int upLine = 0;
  int downLine = frameMaskSum.rows;
  float upLineProb = 0;
  float downLineProb = 0;
  if (ylines.size() >= 2) {
      cv::Mat labels,centers;
      int K=2, attempts=10, flags=cv::KMEANS_RANDOM_CENTERS;
      cv::TermCriteria tc;
      cv::kmeans(ylines, K, labels, tc, attempts, flags, centers);
      int numUpCount = 0;
      int numDownCount = 0;
      upLine = std::min(centers.at<float>( 0, 0  ), centers.at<float>( 1, 0  ));
      downLine = std::max(centers.at<float>( 0, 0  ), centers.at<float>( 1, 0  ));
      for (int i = 0; i < ylines.size(); i++) {
          if (std::abs(ylines[i] - upLine) <= 2){
              numUpCount = numUpCount + 1;
          }
          if (std::abs(ylines[i] - downLine) <= 2){
              numDownCount = numDownCount + 1;
          }
      }
      upLineProb = numUpCount/float(numFrames-1);
      downLineProb = numDownCount/float(numFrames-1);
  }

  frameMaskSum = frameMaskSum/numFrames;
  frameMaskSum.convertTo(frameMaskSum, CV_8U);
  cv::morphologyEx(frameMaskSum, frameMaskSum, cv::MORPH_CLOSE, kernel);

  cv::Rect maxBBox, maxBBoxLow;
  cv::Mat outMask;
  cv::Mat outMaskLow;

  status = globalThreshold(frameMaskSum, outMask, 70);
  status = findMaxBBox(outMask, maxBBox);

  int y1 = int(maxBBox.y/(1.0*imgScale));
  int y2 = int((maxBBox.y + maxBBox.height)/(1.0*imgScale));
  int numTopLines = 0;
  int numDownLines = 0;
  for (int i = 0; i < ylines.size(); i++) {
    if(std::abs(ylines[i] - y1) <= 3) {
      ++numTopLines;
    }
    if(std::abs(ylines[i] - y2) <= 3) {
      ++numDownLines;
    }
  }
  status = globalThreshold(frameMaskSum, outMaskLow, 30);
  status = findMaxBBox(outMaskLow, maxBBoxLow);
  y1 = int(maxBBoxLow.y/(1.0*imgScale));
  y2 = int((maxBBoxLow.y + maxBBoxLow.height)/(1.0*imgScale));
  int numTopLinesLow = 0;
  int numDownLinesLow = 0;
  for(int i = 0; i < ylines.size(); i++) {
    if(std::abs(ylines[i] - y1) <= 3) {
      ++numTopLinesLow;
    }
    if(std::abs(ylines[i] - y2) <= 3) {
      ++numDownLinesLow;
    }
  }

  if(numDownLinesLow>=1 && numTopLinesLow>=1 && (numTopLines<=0 || numDownLines<=0)) {
    maxBBox = maxBBoxLow;
    outMask = outMaskLow;
  }

  if(upLineProb>=0.6 && downLineProb>=0.6 && numTopLines<=0 
          && numDownLines<=0 && int(downLine-upLine)>=int(outMask.rows/4.0)) {
    maxBBox = cv::Rect(0, int(upLine*imgScale), outMask.cols, int(int(downLine-upLine)*imgScale));
    outMask = cv::Mat::zeros(outMask.rows, outMask.cols, CV_8U);
    cv::Mat tmpOnes = 255*cv::Mat::ones(int((downLine-upLine)*imgScale), outMask.cols, CV_8U);
    cv::Mat destRoi = outMask(maxBBox);
    tmpOnes.copyTo(destRoi);
  }

  int midConfidenceFlag = 0;
  std::vector<cv::Rect> lowBBoxs;
  status = findTopKBBox(outMaskLow, lowBBoxs);
  for(int i = 0; i < lowBBoxs.size() && i < 3; ++i) {
      if(lowBBoxs[i].width*lowBBoxs[i].height <= 100) {
          continue;
      }
      y1 = int(lowBBoxs[i].y/(1.0*imgScale));
      y2 = int((lowBBoxs[i].y + lowBBoxs[i].height)/(1.0*imgScale));
      int numTopLinesLow = 0;
      int numDownLinesLow = 0;
      for(int i = 0; i < ylines.size(); i++) {
          if(std::abs(ylines[i] - y1) <= 4) {
              ++numTopLinesLow;
              ++globalUpLines;
          }
          if(std::abs(ylines[i] - y2) <= 4) {
              ++numDownLinesLow;
              ++globalDownLines;
          }
      }
      if(numTopLinesLow >= (0.4*numFrames) && numDownLinesLow >= (0.4*numFrames)
              && int(y2-y1)>=int(outMaskLow.rows/(imgScale*6.0))) {
          maxBBox = lowBBoxs[i];
          midConfidenceFlag = 1;
      }
  }

  std::vector<cv::Rect> BBoxs;
  status = findTopKBBox(outMask, BBoxs);
  for(int i = 0; i < BBoxs.size() && i < 3; ++i) {
      if(BBoxs[i].width*BBoxs[i].height <= 100) {
          continue;
      }
      y1 = int(BBoxs[i].y/(1.0*imgScale));
      y2 = int((BBoxs[i].y + BBoxs[i].height)/(1.0*imgScale));
      int numTopLines = 0;
      int numDownLines = 0;
      for(int i = 0; i < ylines.size(); i++) {
          if(std::abs(ylines[i] - y1) <= 4) {
              ++numTopLines;
              ++globalUpLines;
          }
          if(std::abs(ylines[i] - y2) <= 4) {
              ++numDownLines;
              ++globalDownLines;
          }
      }
      if(numTopLines >= (0.4*numFrames) && numDownLines >= (0.4*numFrames)
              && int(y2-y1)>=int(outMask.rows/(imgScale*6.0))) {
          maxBBox = BBoxs[i];
          midConfidenceFlag = 1;
      }
  }

  int upConfidenceFlag = 0;
  int downConfidenceFlag = 0;
  if (std::abs(maxBBox.y/imgScale - upLine) <= 3 && upLineProb >= 0.4) {
      upConfidenceFlag = 1;
  }
  if(std::abs(maxBBox.y/imgScale + maxBBox.height/imgScale - downLine) <= 3 && downLineProb >= 0.4) {
      downConfidenceFlag = 1;
  }

  int boxArea = (maxBBox.width*maxBBox.height);
  int tmp_left = int(std::max((double)maxBBox.x/imgScale, (double)(0.0)));
  int tmp_right = int(std::min((double)(maxBBox.x + maxBBox.width)/imgScale,
                               (double)(frame.cols-1)));
  int tmp_top = int(std::max((double)maxBBox.y/imgScale, (double)(0.0)));
  int tmp_bottom = int(std::min((double)(maxBBox.y + maxBBox.height)/imgScale,
                                (double)(frame.rows-1)));

  int count_num = 0;
  int tmp_left_outmask = int(std::max((double)maxBBox.x, (double)(0.0)));
  int tmp_right_outmask = int(std::min((double)(maxBBox.x + maxBBox.width),
                                       (double)(outMask.cols-1)));
  int tmp_top_outmask = int(std::max((double)maxBBox.y, (double)(0.0)));
  int tmp_bottom_outmask = int(std::min((double)(maxBBox.y + maxBBox.height),
                                        (double)(outMask.rows-1)));
  for(int i = tmp_top_outmask; i < tmp_bottom_outmask; i++) {
    for(int j = tmp_left_outmask; j < tmp_right_outmask; j++) {
      int tmp = (int)outMask.at<uchar>(i,j);
      if(tmp>128) {
        ++count_num;
      }
    }
  }

  prob_detect = 1.0*count_num/boxArea;
  if(upConfidenceFlag >=1 && downConfidenceFlag >= 1) {
      prob_detect = prob_detect + 0.4;
  }
  if(midConfidenceFlag >= 1) {
      prob_detect = prob_detect + 0.4;
  }
  prob_detect = std::min((double)prob_detect, 1.0);

  if (boxArea > (motionAreaRatio*outMask.rows*outMask.cols && (globalUpLines>=1 || globalDownLines>=1))) {
    motionLocInfo->set_left(tmp_left);
    motionLocInfo->set_right(tmp_right);
    motionLocInfo->set_top(tmp_top);
    motionLocInfo->set_bottom(tmp_bottom);
    motionLocInfo->set_prob_detect(prob_detect);
    return 1;
  } else if(boxArea > (motionAreaRatioLow*outMask.rows * outMask.cols) &&
            boxArea <= (motionAreaRatio*outMask.rows*outMask.cols) && (globalUpLines>=1 || globalDownLines>=1)) {
    motionLocInfo->set_left(tmp_left);
    motionLocInfo->set_right(tmp_right);
    motionLocInfo->set_top(tmp_top);
    motionLocInfo->set_bottom(tmp_bottom);
    motionLocInfo->set_prob_detect(prob_detect);
    return 1;
  }

  return 0;
}

}
}
