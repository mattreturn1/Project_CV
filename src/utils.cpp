#include "utils.hpp"
#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

void createOutputFolder(const std::string& path) {
    if (!fs::exists(path)) {
        fs::create_directories(path);
        std::cout << "Directory created: " << path << std::endl;
    }
}

cv::CascadeClassifier loadCascade(const std::string& cascadePath) {
    cv::CascadeClassifier cascade;
    if (!cascade.load(cascadePath)) {
        std::cerr << "Error loading cascade file" << std::endl;
        exit(-1);
    }
    return cascade;
}

std::vector<cv::String> getImagePaths(const std::string& folder) {
    std::vector<cv::String> files;
    cv::glob(folder + "*.jpg", files);
    return files;
}
