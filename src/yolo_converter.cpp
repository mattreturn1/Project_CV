// Author: Mattia Cozza

#include <opencv2/opencv.hpp>
#include <fstream>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

void convertYoloToCsv(const std::string& labelFolder, const std::string& imageFolder, const std::string& outputCsv) {
    std::ofstream csv(outputCsv);
    csv << "image,label,x,y,w,h\n";  // CSV header

    for (const auto& entry : fs::directory_iterator(labelFolder)) {
        if (entry.path().extension() != ".txt") continue;  // Process only .txt label files

        std::string labelPath = entry.path().string();
        std::string fileName = entry.path().stem().string(); // Get filename without extension
        std::string imagePath = imageFolder + "/" + fileName + ".jpg";

        if (!fs::exists(imagePath)) {
            imagePath = imageFolder + "/" + fileName + ".png";  // Optional fallback to .png
        }

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
            if (!(ss >> label >> cx >> cy >> w >> h)) {
                std::cerr << "Malformed line in: " << labelPath << " → " << line << "\n";
                continue;  // Skip malformed lines
            }

            // Convert YOLO normalized format to absolute coordinates
            int abs_x = static_cast<int>((cx - w / 2.0f) * static_cast<float>(width));
            int abs_y = static_cast<int>((cy - h / 2.0f) * static_cast<float>(height));
            int abs_w = static_cast<int>(w * static_cast<float>(width));
            int abs_h = static_cast<int>(h * static_cast<float>(height));

            // Ensure bounding boxes stay within image boundaries
            abs_x = std::max(0, abs_x);
            abs_y = std::max(0, abs_y);
            abs_w = std::min(abs_w, width - abs_x);
            abs_h = std::min(abs_h, height - abs_y);

            csv << fileName + ".jpg" << "," << label << "," << abs_x << "," << abs_y << "," << abs_w << "," << abs_h << "\n";
        }
    }

    csv.close();
    std::cout << "Conversion completed: " << outputCsv << std::endl;
}
