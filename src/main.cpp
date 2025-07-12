#include <iostream>
#include "utils.hpp"
#include "face_detector.hpp"

int main() {
    std::string cascadePath = "haar/haarcascade_frontalface_alt2.xml";
    std::string inputFolder = "data/input/images/";
    std::string outputFolder = "data/output/images/";
    std::string outputCsv = "data/detections.csv";

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

    return 0;
}
