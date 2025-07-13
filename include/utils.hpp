// Mattia Cozza

#ifndef UTILS_HPP
#define UTILS_HPP

#include <string>
#include <opencv2/opencv.hpp>

void createOutputFolder(const std::string& path);
cv::CascadeClassifier loadCascade(const std::string& cascadePath);
std::vector<cv::String> getImagePaths(const std::string& folder);

#endif
