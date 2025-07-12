#ifndef EVALUATION_HPP
#define EVALUATION_HPP

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

struct Detection {
    std::string imageName;
    cv::Rect bbox;
};

std::vector<Detection> loadDetectionsFromCSV(const std::string& csvPath);
double computeIoU(const cv::Rect& pred, const cv::Rect& truth);
void evaluateFaceDetection(const std::string& predCsv, const std::string& gtCsv, double iouThreshold = 0.5);

#endif
