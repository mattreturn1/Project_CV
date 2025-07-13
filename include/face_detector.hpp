// Mattia Cozza

#ifndef FACE_DETECTOR_HPP
#define FACE_DETECTOR_HPP

#include <string>
#include <opencv2/opencv.hpp>
#include <fstream>

void processImage(const std::string& file,
                  const std::string& outputFolder,
                  cv::CascadeClassifier& faceCascade,
                  std::ofstream& csv);

#endif
