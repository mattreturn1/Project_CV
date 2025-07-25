// Author: Mattia Cozza

#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include "evaluation.hpp"

// Load detections from a CSV file
std::vector<Detection> loadDetectionsFromCSV(const std::string &csvPath) {
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
        int offset = (tokens.size() == 6) ? 1 : 0; // If label is present, skip token[1]

        try {
            int x = std::stoi(tokens[1 + offset]);
            int y = std::stoi(tokens[2 + offset]);
            int w = std::stoi(tokens[3 + offset]);
            int h = std::stoi(tokens[4 + offset]);

            // Skip invalid bounding boxes
            if (w <= 0 || h <= 0 || x < 0 || y < 0 || w > 10000 || h > 10000) {
                std::cerr << "Invalid bbox skipped in " << name << ": x=" << x << " y=" << y << " w=" << w << " h=" << h
                        << "\n";
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
double computeIoU(const cv::Rect &pred, const cv::Rect &truth) {
    int xA = std::max(pred.x, truth.x);
    int yA = std::max(pred.y, truth.y);
    int xB = std::min(pred.x + pred.width, truth.x + truth.width);
    int yB = std::min(pred.y + pred.height, truth.y + truth.height);

    int interArea = std::max(0, xB - xA) * std::max(0, yB - yA);
    int unionArea = pred.area() + truth.area() - interArea;

    return unionArea > 0 ? static_cast<double>(interArea) / unionArea : 0.0;
}

// Evaluate face detection by comparing predictions with ground truth
void evaluateFaceDetection(const std::string &predCsv, const std::string &gtCsv, const std::string &tpCsvOutput,
                           double iouThreshold) {
    auto predictions = loadDetectionsFromCSV(predCsv);
    auto gts = loadDetectionsFromCSV(gtCsv);

    std::unordered_map<std::string, std::vector<std::pair<int, cv::Rect> > > gtMap;
    std::unordered_map<std::string, std::vector<cv::Rect> > predMap;

    std::ifstream gtFile(gtCsv);
    std::string line;
    std::getline(gtFile, line); // skip header
    while (std::getline(gtFile, line)) {
        std::stringstream ss(line);
        std::vector<std::string> tokens;
        std::string token;
        while (std::getline(ss, token, ',')) tokens.push_back(token);
        if (tokens.size() != 6) continue;

        std::string name = tokens[0];
        int label = std::stoi(tokens[1]);
        int x = std::stoi(tokens[2]);
        int y = std::stoi(tokens[3]);
        int w = std::stoi(tokens[4]);
        int h = std::stoi(tokens[5]);

        gtMap[name].emplace_back(label, cv::Rect(x, y, w, h));
    }

    for (const auto &det: predictions)
        predMap[det.imageName].push_back(det.bbox);

    int TP = 0, FP = 0, FN = 0;

    std::ofstream tpCsv;
    if (!tpCsvOutput.empty()) {
        tpCsv.open(tpCsvOutput);
        tpCsv << "image,label,x,y,w,h\n";
    }

    for (const auto &[imageName, gtPairs]: gtMap) {
        std::vector<cv::Rect> predRects = predMap[imageName];
        std::vector<bool> gtMatched(gtPairs.size(), false);
        std::vector<bool> predMatched(predRects.size(), false);

        std::vector<std::pair<cv::Rect, cv::Scalar> > rectsToDraw;

        for (size_t p = 0; p < predRects.size(); ++p) {
            const auto &pred = predRects[p];
            double maxIoU = 0.0;
            int bestIdx = -1;

            for (size_t g = 0; g < gtPairs.size(); ++g) {
                double iou = computeIoU(pred, gtPairs[g].second);
                if (iou > maxIoU) {
                    maxIoU = iou;
                    bestIdx = static_cast<int>(g);
                }
            }

            if (maxIoU >= iouThreshold && bestIdx != -1 && !gtMatched[bestIdx]) {
                TP++;
                gtMatched[bestIdx] = true;
                predMatched[p] = true;

                rectsToDraw.emplace_back(pred, cv::Scalar(0, 255, 0)); // Green

                if (tpCsv.is_open()) {
                    int label = gtPairs[bestIdx].first;
                    tpCsv << imageName << "," << label << "," << pred.x << "," << pred.y << "," << pred.width << "," <<
                            pred.height << "\n";
                }
            }
        }

        // FP: pred not matched
        for (size_t p = 0; p < predRects.size(); ++p) {
            if (!predMatched[p]) {
                FP++;
                rectsToDraw.emplace_back(predRects[p], cv::Scalar(0, 0, 255)); // Red
            }
        }

        // FN: gt not matched
        for (size_t g = 0; g < gtPairs.size(); ++g) {
            if (!gtMatched[g]) {
                FN++;
                rectsToDraw.emplace_back(gtPairs[g].second, cv::Scalar(255, 0, 0)); // Blue
            }
        }
    }

    if (tpCsv.is_open()) tpCsv.close();

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
