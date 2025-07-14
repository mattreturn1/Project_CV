// Author: Mattia Cozza

#include "face_detector.hpp"

void processImage(const std::string& file,
                  const std::string& outputFolder,
                  cv::CascadeClassifier& frontalCascade,
                  cv::CascadeClassifier& profileCascade,
                  std::ofstream& csv) {
    cv::Mat img = cv::imread(file);
    if (img.empty()) {
        std::cerr << "Error loading image: " << file << std::endl;
        return;
    }

    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);  // Convert to grayscale

    // CLAHE for adaptive local contrast
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(3.0);  // A bit more aggressive
    clahe->apply(gray, gray);

    // Optional contrast boosting (alpha=1.5, beta=10)
    gray.convertTo(gray, -1, 1.5, 10);

    // Slight blur to reduce noise
    cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0);

    std::vector<cv::Rect> frontalFaces, profileFaces, allFaces;

    // Detect frontal faces
    frontalCascade.detectMultiScale(gray, frontalFaces, 1.1, 6, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(50, 50));

    // Detect profile faces
    profileCascade.detectMultiScale(gray, profileFaces, 1.1, 6, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(50, 50));

    // Combine detections
    allFaces.insert(allFaces.end(), frontalFaces.begin(), frontalFaces.end());
    allFaces.insert(allFaces.end(), profileFaces.begin(), profileFaces.end());

    // Remove overlapping detections (deduplication)
    std::vector<int> weights;
    cv::groupRectangles(allFaces, weights, 0, 0.3);  // Merge rectangles that overlap heavily

    for (const auto& face : allFaces) {
        int imageArea = img.cols * img.rows;

        // Filter out very small or very large detections
        if (face.area() < imageArea * 0.001 ||
            face.width < img.cols * 0.03 || face.height < img.rows * 0.03 ||
            face.width > img.cols * 0.6 || face.height > img.rows * 0.6)
            continue;

        // Filter based on aspect ratio (to avoid false positives)
        float aspectRatio = static_cast<float>(face.width) / static_cast<float>(face.height);
        if (aspectRatio < 0.75f || aspectRatio > 1.33f)
            continue;

        cv::rectangle(img, face, cv::Scalar(0, 255, 0), 2);  // Draw rectangle on detected face
        std::string imageName = file.substr(file.find_last_of("/\\") + 1);
        csv << imageName << "," << face.x << "," << face.y << "," << face.width << "," << face.height << "\n";
    }

    // Save the processed image with detections
    std::string outputPath = outputFolder + file.substr(file.find_last_of("/\\") + 1);
    cv::imwrite(outputPath, img);
}
