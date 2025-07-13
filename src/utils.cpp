// Mattia Cozza

#include "utils.hpp"
#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

// Creates the output directory if it doesn't exist
void createOutputFolder(const std::string& path) {
    if (!fs::exists(path)) {
        fs::create_directories(path);
        std::cout << "Directory created: " << path << std::endl;
    }
}

// Loads the Haar cascade classifier from the specified file
cv::CascadeClassifier loadCascade(const std::string& cascadePath) {
    cv::CascadeClassifier cascade;
    if (!cascade.load(cascadePath)) {
        std::cerr << "Error loading cascade file" << std::endl;
        exit(-1);  // Exit if loading fails
    }
    return cascade;
}

// Retrieves all .jpg image file paths from the given folder
std::vector<cv::String> getImagePaths(const std::string& folder) {
    std::vector<cv::String> files;
    cv::glob(folder + "*.jpg", files);
    return files;
}
