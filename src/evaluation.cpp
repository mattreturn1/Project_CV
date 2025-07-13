// Author: Mattia Cozza

#include "evaluation.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map>
#include <opencv2/opencv.hpp>

// Load detections from a CSV file
std::vector<Detection> loadDetectionsFromCSV(const std::string& csvPath) {
    std::vector<Detection> detections;
    std::ifstream file(csvPath);
    if (!file.is_open()) {
        std::cerr << "Error opening: " << csvPath << std::endl;
        return detections;
    }

    std::string line;
    std::getline(file, line); // Skip header

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<std::string> tokens;
        std::string token;

        while (std::getline(ss, token, ',')) {
            tokens.push_back(token);
        }

        if (tokens.size() < 5) {
            std::cerr << "Invalid line: " << line << "\n";
            continue;
        }

        std::string name = tokens[0];
        int offset = (tokens.size() == 6) ? 1 : 0;  // If label is present, skip token[1]

        try {
            int x = std::stoi(tokens[1 + offset]);
            int y = std::stoi(tokens[2 + offset]);
            int w = std::stoi(tokens[3 + offset]);
            int h = std::stoi(tokens[4 + offset]);

            // Skip invalid bounding boxes
            if (w <= 0 || h <= 0 || x < 0 || y < 0 || w > 10000 || h > 10000) {
                std::cerr << "Invalid bbox skipped in " << name << ": x=" << x << " y=" << y << " w=" << w << " h=" << h << "\n";
                continue;
            }

            detections.push_back({name, cv::Rect(x, y, w, h)});
        } catch (...) {
            std::cerr << "Error parsing line: " << line << "\n";
            continue;
        }
    }

    return detections;
}

// Compute Intersection over Union (IoU) between two bounding boxes
double computeIoU(const cv::Rect& pred, const cv::Rect& truth) {
    int xA = std::max(pred.x, truth.x);
    int yA = std::max(pred.y, truth.y);
    int xB = std::min(pred.x + pred.width, truth.x + truth.width);
    int yB = std::min(pred.y + pred.height, truth.y + truth.height);

    int interArea = std::max(0, xB - xA) * std::max(0, yB - yA);
    int unionArea = pred.area() + truth.area() - interArea;

    return unionArea > 0 ? static_cast<double>(interArea) / unionArea : 0.0;
}

// Evaluate face detection by comparing predictions with ground truth
void evaluateFaceDetection(const std::string& predCsv, const std::string& gtCsv, double iouThreshold) {
    auto preds = loadDetectionsFromCSV(predCsv);
    auto gts = loadDetectionsFromCSV(gtCsv);

    // Group detections by image
    std::unordered_map<std::string, std::vector<cv::Rect>> gtMap, predMap;
    for (const auto& det : gts) gtMap[det.imageName].push_back(det.bbox);
    for (const auto& det : preds) predMap[det.imageName].push_back(det.bbox);

    int TP = 0, FP = 0, FN = 0;

    //Visual Debugging Output
    std::string debugFolder = "data/output/debug/";
    std::filesystem::create_directories(debugFolder);

    for (const auto& [imageName, gtRects] : gtMap) {
        cv::Mat img = cv::imread("data/input/images/" + imageName);
        if (img.empty()) continue;

        // Draw Ground Truth (red)
        for (const auto& gt : gtRects) {
            cv::rectangle(img, gt, cv::Scalar(0, 0, 255), 2);
        }

        // Draw Predictions (green)
        for (const auto& pred : predMap[imageName]) {
            cv::rectangle(img, pred, cv::Scalar(0, 255, 0), 2);
        }

        // Save image with annotations
        cv::imwrite(debugFolder + imageName, img);
    }

    std::cout << "\nDebug images saved in: " << debugFolder << "\n";

    for (const auto& [imageName, gtRects] : gtMap) {
        std::vector<cv::Rect> predRects = predMap[imageName];
        std::vector<bool> gtMatched(gtRects.size(), false);  // Track matched GTs

        for (const auto& pred : predRects) {
            double maxIoU = 0.0;
            int bestIdx = -1;
            for (size_t i = 0; i < gtRects.size(); ++i) {
                double iou = computeIoU(pred, gtRects[i]);
                std::cout << "[DEBUG] " << imageName << ":\n";
                std::cout << " - Pred: x=" << pred.x << " y=" << pred.y
                          << " w=" << pred.width << " h=" << pred.height << "\n";
                std::cout << " - GT[" << i << "]: x=" << gtRects[i].x << " y=" << gtRects[i].y
                          << " w=" << gtRects[i].width << " h=" << gtRects[i].height << "\n";
                std::cout << " - IoU = " << iou << "\n";

                // Keep track of the best IoU match
                if (iou > maxIoU) {
                    maxIoU = iou;
                    bestIdx = static_cast<int>(i);
                }
            }

            // If the best IoU is valid and the GT wasn't matched yet → True Positive
            if (maxIoU >= iouThreshold && bestIdx != -1 && !gtMatched[bestIdx]) {
                TP++;
                gtMatched[bestIdx] = true;
                std::cout << "TP match with GT[" << bestIdx << "] (IoU: " << maxIoU << ")\n";
            } else {
                FP++;
                std::cout << "FP (best IoU: " << maxIoU << ")\n";
            }
        }

        // Count unmatched ground truths as False Negatives
        for (bool matched : gtMatched) {
            if (!matched) FN++;
        }
    }

    // Calculate precision, recall, and F1-score
    double precision = TP + FP > 0 ? static_cast<double>(TP) / (TP + FP) : 0.0;
    double recall = TP + FN > 0 ? static_cast<double>(TP) / (TP + FN) : 0.0;
    double f1 = precision + recall > 0 ? 2 * (precision * recall) / (precision + recall) : 0.0;

    std::cout << "\nFace Detection Evaluation\n";
    std::cout << "True Positives: " << TP << "\n";
    std::cout << "False Positives: " << FP << "\n";
    std::cout << "False Negatives: " << FN << "\n";
    std::cout << "Precision: " << precision << "\n";
    std::cout << "Recall:    " << recall << "\n";
    std::cout << "F1-Score:  " << f1 << "\n";
}
