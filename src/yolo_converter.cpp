#include <opencv2/opencv.hpp>
#include <fstream>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

void convertYoloToCsv(const std::string& labelFolder, const std::string& imageFolder, const std::string& outputCsv) {
    std::ofstream csv(outputCsv);
    csv << "image,label,x,y,w,h\n";  // CSV header

    for (const auto& entry : fs::directory_iterator(labelFolder)) {
        if (entry.path().extension() != ".txt") continue;

        std::string labelPath = entry.path().string();
        std::string fileName = entry.path().stem().string(); // es: img1
        std::string imagePath = imageFolder + "/" + fileName + ".jpg";

        cv::Mat image = cv::imread(imagePath);
        if (image.empty()) {
            std::cerr << "Image not found or unreadable: " << imagePath << "\n";
            continue;
        }

        int width = image.cols;
        int height = image.rows;

        std::ifstream labelFile(labelPath);
        std::string line;
        while (std::getline(labelFile, line)) {
            int label;
            float cx, cy, w, h;
            std::istringstream ss(line);
            ss >> label >> cx >> cy >> w >> h;

            int abs_x = static_cast<int>((cx - w / 2) * width);
            int abs_y = static_cast<int>((cy - h / 2) * height);
            int abs_w = static_cast<int>(w * width);
            int abs_h = static_cast<int>(h * height);

            csv << fileName + ".jpg" << "," << label << "," << abs_x << "," << abs_y << "," << abs_w << "," << abs_h << "\n";
        }
    }

    csv.close();
    std::cout << "Conversion completed: " << outputCsv << std::endl;
}
