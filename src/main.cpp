#include <iostream>ù
#include <filesystem>
#include "utils.hpp"
#include "face_detector.hpp"
#include "evaluation.hpp"
#include "yolo_converter.hpp"

namespace fs = std::filesystem;


int main() {
    std::string cascadePath = "haar/haarcascade_frontalface_alt2.xml";
    std::string inputFolder = "data/input/images/";
    std::string outputFolder = "data/output/images/";
    std::string outputCsv = "data/detections.csv";

    if (!fs::exists("data/input/ground_truth.csv")) {
        convertYoloToCsv("data/input/labels/", "data/input/images/", "data/input/ground_truth.csv");
    }
    createOutputFolder(outputFolder);
    cv::CascadeClassifier faceCascade = loadCascade(cascadePath);

    std::vector<cv::String> imageFiles = getImagePaths(inputFolder);

    std::ofstream csv(outputCsv);
    csv << "image,x,y,w,h\n";

    for (const auto& file : imageFiles) {
        processImage(file, outputFolder, faceCascade, csv);
    }

    csv.close();
    std::cout << "Face detection completed.\nResults in: " << outputFolder << " and " << outputCsv << "\n";

    std::string predCsv = "data/detections.csv";
    std::string gtCsv = "data/input/ground_truth.csv"; // da creare manualmente o già fornito
    evaluateFaceDetection(predCsv, gtCsv, 0.5);
    return 0;
}
