// Author: Mattia Cozza

#include <iostream>
#include <filesystem>
#include "utils.hpp"
#include "face_detector.hpp"
#include "evaluation.hpp"
#include "yolo_converter.hpp"

namespace fs = std::filesystem;

int main() {
    std::string cascadePathFrontal = "haar_cascade/haarcascade_frontalface_alt2.xml";
    std::string cascadePathProfile = "haar_cascade/haarcascade_profileface.xml";
    std::string inputFolder = "data/input/images/";
    std::string outputFolder = "data/output/images/";
    std::string outputCsv = "data/output/alldetections.csv";

    // Convert YOLO labels to CSV format if ground truth CSV does not exist
    if (!fs::exists("data/input/ground_truth.csv")) {
        convertYoloToCsv("data/input/labels/", "data/input/images/", "data/input/ground_truth.csv");
    }

    // Create the output folder if it doesn't exist
    createOutputFolder(outputFolder);

    // Load Haar cascade classifiers
    cv::CascadeClassifier frontalCascade = loadCascade(cascadePathFrontal);
    cv::CascadeClassifier profileCascade = loadCascade(cascadePathProfile);

    // Get list of all image file paths from input folder
    std::vector<cv::String> imageFiles = getImagePaths(inputFolder);

    // Open CSV file to save detection results
    std::ofstream csv(outputCsv);
    csv << "image,x,y,w,h\n";

    // Process each image: detect faces and save results
    for (const auto &file: imageFiles) {
        processImage(file, outputFolder, frontalCascade, profileCascade, csv);
    }

    csv.close();
    std::cout << "Face detection completed.\nResults in: " << outputFolder << " and " << outputCsv << "\n";

    // Evaluate detections against ground truth using IoU threshold
    std::string predCsv = "data/output/alldetections.csv";
    std::string gtCsv = "data/input/ground_truth.csv";
    std::string tpCsv = "data/detections.csv";
    evaluateFaceDetection(predCsv, gtCsv, tpCsv, 0.5);

    return 0;
}
