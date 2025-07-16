// Author: Mattia Cozza

#include "face_detector.hpp"

cv::Mat preprocessImage(const cv::Mat &img) {
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // CLAHE
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(3.0);
    clahe->apply(gray, gray);

    // Boost contrast
    gray.convertTo(gray, -1, 1.5, 10);

    // Reduce noise
    cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0);

    return gray;
}

std::vector<cv::Rect> detectFrontalFaces(const cv::Mat &gray,
                                         cv::CascadeClassifier &frontalCascade,
                                         double scaleFactor,
                                         int minNeighbors,
                                         const cv::Size &minSize) {
    std::vector<cv::Rect> faces;
    frontalCascade.detectMultiScale(gray, faces, scaleFactor, minNeighbors, 0 | cv::CASCADE_SCALE_IMAGE, minSize);
    return faces;
}

std::vector<cv::Rect> detectProfileFaces(const cv::Mat &gray,
                                         cv::CascadeClassifier &profileCascade,
                                         double scaleFactor,
                                         int minNeighbors,
                                         const cv::Size &minSize) {
    std::vector<cv::Rect> profileFaces, flippedFaces, allProfiles;

    // Detect right-facing profiles
    profileCascade.detectMultiScale(gray, profileFaces, scaleFactor, minNeighbors, 0 | cv::CASCADE_SCALE_IMAGE,
                                    minSize);

    // Flip the image horizontally for left-facing profiles
    cv::Mat flipped;
    cv::flip(gray, flipped, 1);
    profileCascade.detectMultiScale(flipped, flippedFaces, scaleFactor, minNeighbors, 0 | cv::CASCADE_SCALE_IMAGE,
                                    minSize);

    // Convert flipped coordinates
    for (const auto &r: flippedFaces) {
        cv::Rect corrected;
        corrected.x = gray.cols - r.x - r.width;
        corrected.y = r.y;
        corrected.width = r.width;
        corrected.height = r.height;
        profileFaces.push_back(corrected);
    }

    return profileFaces;
}

std::vector<cv::Rect> detectRotatedFaces(const cv::Mat &gray,
                                         cv::CascadeClassifier &frontalCascade,
                                         double scaleFactor,
                                         int minNeighbors,
                                         const cv::Size &minSize,
                                         const std::vector<int> &rotationAngles) {
    std::vector<cv::Rect> rotatedDetections;

    for (int angle: rotationAngles) {
        // Compute rotation matrix
        cv::Mat rotMat = cv::getRotationMatrix2D(cv::Point(gray.cols / 2, gray.rows / 2), angle, 1.0);
        cv::Mat rotated;
        cv::warpAffine(gray, rotated, rotMat, gray.size());

        // Detect frontal faces on rotated image
        std::vector<cv::Rect> faces;
        frontalCascade.detectMultiScale(rotated, faces, scaleFactor, minNeighbors, 0 | cv::CASCADE_SCALE_IMAGE,
                                        minSize);

        // Invert rotation and map bounding boxes back
        cv::Mat invRotMat;
        cv::invertAffineTransform(rotMat, invRotMat);

        for (const auto &r: faces) {
            std::vector<cv::Point2f> ptsRotated = {
                cv::Point2f(static_cast<float>(r.x), static_cast<float>(r.y)),
                cv::Point2f(static_cast<float>(r.x + r.width), static_cast<float>(r.y)),
                cv::Point2f(static_cast<float>(r.x), static_cast<float>(r.y + r.height)),
                cv::Point2f(static_cast<float>(r.x + r.width), static_cast<float>(r.y + r.height))
            };

            std::vector<cv::Point2f> ptsOriginal;
            cv::transform(ptsRotated, ptsOriginal, invRotMat);

            float minX = ptsOriginal[0].x, maxX = ptsOriginal[0].x;
            float minY = ptsOriginal[0].y, maxY = ptsOriginal[0].y;
            for (const auto &pt: ptsOriginal) {
                minX = std::min(minX, pt.x);
                maxX = std::max(maxX, pt.x);
                minY = std::min(minY, pt.y);
                maxY = std::max(maxY, pt.y);
            }

            cv::Rect bbox(cv::Point2f(minX, minY), cv::Point2f(maxX, maxY));
            if (bbox.x >= 0 && bbox.y >= 0 && bbox.x + bbox.width <= gray.cols && bbox.y + bbox.height <= gray.rows)
                rotatedDetections.push_back(bbox);
        }
    }

    return rotatedDetections;
}

std::vector<cv::Rect> mergeOverlappingBoxes(const std::vector<cv::Rect> &boxes, float iouThreshold = 0.3f) {
    std::vector<cv::Rect> merged;
    std::vector<bool> used(boxes.size(), false);

    for (size_t i = 0; i < boxes.size(); ++i) {
        if (used[i]) continue;

        cv::Rect mergedRect = boxes[i];
        used[i] = true;

        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (used[j]) continue;

            float interArea = static_cast<float>((mergedRect & boxes[j]).area());
            float unionArea = static_cast<float>(mergedRect.area() + boxes[j].area()) - interArea;
            float iou = interArea / unionArea;

            if (iou > iouThreshold) {
                // Keep the smaller bounding box
                if (boxes[j].area() < mergedRect.area()) {
                    mergedRect = boxes[j];
                }
                used[j] = true;
            }
        }

        merged.push_back(mergedRect);
    }

    return merged;
}

bool isValidFace(const cv::Rect &face, const cv::Mat &grayImage) {
    int imgW = grayImage.cols;
    int imgH = grayImage.rows;

    // Filter out very small or very large detections
    if (face.width < imgW * 0.03 || face.height < imgH * 0.03 ||
        face.width > imgW * 0.7 || face.height > imgH * 0.7) {
        return false;
    }

    return true;
}

void drawAndSaveDetections(const std::string &inputFile,
                           const std::string &outputFolder,
                           const std::vector<cv::Rect> &faces,
                           const cv::Mat &image,
                           std::ofstream &csv) {
    cv::Mat annotated = image.clone();
    std::string imageName = inputFile.substr(inputFile.find_last_of("/\\") + 1);

    for (const auto &face: faces) {
        cv::rectangle(annotated, face, cv::Scalar(0, 255, 0), 2);
        csv << imageName << "," << face.x << "," << face.y << "," << face.width << "," << face.height << "\n";
    }

    std::string outputPath = outputFolder + imageName;
    cv::imwrite(outputPath, annotated);
}

void processImage(const std::string &file,
                  const std::string &outputFolder,
                  cv::CascadeClassifier &frontalCascade,
                  cv::CascadeClassifier &profileCascade,
                  std::ofstream &csv) {
    cv::Mat img = cv::imread(file);
    if (img.empty()) {
        std::cerr << "Error loading image: " << file << std::endl;
        return;
    }

    int minDim = std::min(img.cols, img.rows);
    cv::Size minSize(static_cast<int>(minDim * 0.05), static_cast<int>(minDim * 0.05));
    constexpr double scaleFactor = 1.15;
    constexpr int minNeighbors = 5;
    const std::vector<int> rotationAngles = {-45, -30, -15, 15, 30, 45};

    cv::Mat gray = preprocessImage(img);

    auto frontal = detectFrontalFaces(gray, frontalCascade, scaleFactor, minNeighbors, minSize);
    auto profile = detectProfileFaces(gray, profileCascade, scaleFactor, minNeighbors, minSize);
    auto rotated = detectRotatedFaces(gray, frontalCascade, scaleFactor, minNeighbors, minSize, rotationAngles);

    frontal.insert(frontal.end(), profile.begin(), profile.end());
    frontal.insert(frontal.end(), rotated.begin(), rotated.end());

    auto merged = mergeOverlappingBoxes(frontal, 0.3f);

    std::vector<cv::Rect> finalFaces;
    for (const auto &face: merged) {
        if (isValidFace(face, gray))
            finalFaces.push_back(face);
    }

    drawAndSaveDetections(file, outputFolder, finalFaces, img, csv);
}
