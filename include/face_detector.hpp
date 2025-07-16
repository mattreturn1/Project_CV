// Author: Mattia Cozza

#ifndef FACE_DETECTOR_HPP
#define FACE_DETECTOR_HPP

#include <string>
#include <opencv2/opencv.hpp>

cv::Mat preprocessImage(const cv::Mat &img);

std::vector<cv::Rect> detectFrontalFaces(const cv::Mat &gray,
                                         cv::CascadeClassifier &frontalCascade,
                                         double scaleFactor,
                                         int minNeighbors,
                                         const cv::Size &minSize);

std::vector<cv::Rect> detectProfileFaces(const cv::Mat &gray,
                                         cv::CascadeClassifier &profileCascade,
                                         double scaleFactor,
                                         int minNeighbors,
                                         const cv::Size &minSize);

std::vector<cv::Rect> detectRotatedFaces(const cv::Mat &gray,
                                         cv::CascadeClassifier &frontalCascade,
                                         double scaleFactor,
                                         int minNeighbors,
                                         const cv::Size &minSize,
                                         const std::vector<int> &rotationAngles);

std::vector<cv::Rect> mergeOverlappingBoxes(const std::vector<cv::Rect> &boxes, float iouThreshold);

bool isValidFace(const cv::Rect &face, const cv::Mat &grayImage);

void drawAndSaveDetections(const std::string &inputFile,
                           const std::string &outputFolder,
                           const std::vector<cv::Rect> &faces,
                           const cv::Mat &image,
                           std::ofstream &csv);

void processImage(const std::string &file,
                  const std::string &outputFolder,
                  cv::CascadeClassifier &frontalCascade,
                  cv::CascadeClassifier &profileCascade,
                  std::ofstream &csv);

#endif
