#include "evaluation.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map>

std::vector<Detection> loadDetectionsFromCSV(const std::string& csvPath) {
    std::vector<Detection> detections;
    std::ifstream file(csvPath);
    if (!file.is_open()) {
        std::cerr << "❌ Error opening: " << csvPath << std::endl;
        return detections;
    }

    std::string line;
    std::getline(file, line); // Skip header

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string name, field;
        int x, y, w, h;
        int label = -1;  // Optional: if label is included

        std::getline(ss, name, ',');

        // Check how many fields per line
        std::getline(ss, field, ',');
        bool hasLabel = field.find_first_not_of("0123456789") == std::string::npos && line.find(',') != std::string::npos;

        try {
            if (hasLabel) {
                label = std::stoi(field);
                std::getline(ss, field, ','); x = std::stoi(field);
                std::getline(ss, field, ','); y = std::stoi(field);
                std::getline(ss, field, ','); w = std::stoi(field);
                std::getline(ss, field, ','); h = std::stoi(field);
            } else {
                x = std::stoi(field);
                std::getline(ss, field, ','); y = std::stoi(field);
                std::getline(ss, field, ','); w = std::stoi(field);
                std::getline(ss, field, ','); h = std::stoi(field);
            }

            // ⚠️ Defensive check
            if (w <= 0 || h <= 0 || x < 0 || y < 0 || w > 10000 || h > 10000) {
                std::cerr << "⚠️ Invalid bbox skipped in " << name << ": x=" << x << " y=" << y << " w=" << w << " h=" << h << "\n";
                continue;
            }

            detections.push_back({name, cv::Rect(x, y, w, h)});
        } catch (...) {
            std::cerr << "⚠️ Parsing error in line: " << line << "\n";
            continue;
        }
    }

    return detections;
}
double computeIoU(const cv::Rect& pred, const cv::Rect& truth) {
    int xA = std::max(pred.x, truth.x);
    int yA = std::max(pred.y, truth.y);
    int xB = std::min(pred.x + pred.width, truth.x + truth.width);
    int yB = std::min(pred.y + pred.height, truth.y + truth.height);

    int interArea = std::max(0, xB - xA) * std::max(0, yB - yA);
    int unionArea = pred.area() + truth.area() - interArea;

    return unionArea > 0 ? static_cast<double>(interArea) / unionArea : 0.0;
}

void evaluateFaceDetection(const std::string& predCsv, const std::string& gtCsv, double iouThreshold) {
    auto preds = loadDetectionsFromCSV(predCsv);
    auto gts = loadDetectionsFromCSV(gtCsv);

    // Raggruppa per immagine
    std::unordered_map<std::string, std::vector<cv::Rect>> gtMap, predMap;
    for (const auto& det : gts) gtMap[det.imageName].push_back(det.bbox);
    for (const auto& det : preds) predMap[det.imageName].push_back(det.bbox);

    int TP = 0, FP = 0, FN = 0;

    for (const auto& [imageName, gtRects] : gtMap) {
        std::vector<cv::Rect> predRects = predMap[imageName];
        std::vector<bool> gtMatched(gtRects.size(), false);

        for (const auto& pred : predRects) {
            double maxIoU = 0.0;
            int bestIdx = -1;
            for (size_t i = 0; i < gtRects.size(); ++i) {
                double iou = computeIoU(pred, gtRects[i]);
                if (iou > maxIoU) {
                    maxIoU = iou;
                    bestIdx = i;
                }
            }

            if (maxIoU >= iouThreshold && !gtMatched[bestIdx]) {
                TP++;
                gtMatched[bestIdx] = true;
            } else {
                FP++;
            }
        }

        for (bool matched : gtMatched) {
            if (!matched) FN++;
        }
    }

    double precision = TP + FP > 0 ? (double)TP / (TP + FP) : 0.0;
    double recall = TP + FN > 0 ? (double)TP / (TP + FN) : 0.0;
    double f1 = precision + recall > 0 ? 2 * (precision * recall) / (precision + recall) : 0.0;

    std::cout << "\nFace Detection Evaluation\n";
    std::cout << "True Positives: " << TP << "\n";
    std::cout << "False Positives: " << FP << "\n";
    std::cout << "False Negatives: " << FN << "\n";
    std::cout << "Precision: " << precision << "\n";
    std::cout << "Recall:    " << recall << "\n";
    std::cout << "F1-Score:  " << f1 << "\n";
}
