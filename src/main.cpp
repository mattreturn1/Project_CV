// Author: Mattia Cozza

#include <iostream>
#include <filesystem>
#include <fstream>
#include "utils.hpp"
#include "face_detector.hpp"
#include "evaluation.hpp"
#include "yolo_converter.hpp"

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    std::string cascadePathFrontal = "haar_cascade/haarcascade_frontalface_alt2.xml";
    std::string cascadePathProfile = "haar_cascade/haarcascade_profileface.xml";

    // Default input root folder
    std::string inputRoot = "data/input/";
    if (argc > 1) {
        inputRoot = argv[1];
        if (inputRoot.back() != '/' && inputRoot.back() != '\\')
            inputRoot += '/';
    }

    std::string inputImages = inputRoot + "images/";
    std::string inputLabels = inputRoot + "labels/";
    std::string groundTruthCsv = inputRoot + "ground_truth.csv";

    std::string outputFolder = "data/output/images/";
    std::string outputCsv = "data/output/alldetections.csv";

    // Convert YOLO labels to CSV format if ground truth CSV does not exist
    if (!fs::exists(groundTruthCsv)) {
        convertYoloToCsv(inputLabels, inputImages, groundTruthCsv);
    }

    // Create the output folder if it doesn't exist
    createOutputFolder(outputFolder);

    // Load Haar cascade classifiers
    cv::CascadeClassifier frontalCascade = loadCascade(cascadePathFrontal);
    cv::CascadeClassifier profileCascade = loadCascade(cascadePathProfile);

    // Get list of all image file paths from input images folder
    std::vector<cv::String> imageFiles = getImagePaths(inputImages);

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
    std::string tpCsv = "data/detections.csv";
    evaluateFaceDetection(outputCsv, groundTruthCsv, tpCsv, 0.5);

    return 0;
}
