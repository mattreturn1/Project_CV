#include "face_detector.hpp"

void processImage(const std::string& file,
                  const std::string& outputFolder,
                  cv::CascadeClassifier& faceCascade,
                  std::ofstream& csv) {
    cv::Mat img = cv::imread(file);
    if (img.empty()) {
        std::cerr << "Error loading image: " << file << std::endl;
        return;
    }

    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray, gray);

    // Preprocessing
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(2.0);
    clahe->apply(gray, gray);
    cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0);

    std::vector<cv::Rect> faces;

    // Face detection
    faceCascade.detectMultiScale(gray, faces, 1.1, 6, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(50, 50));

    for (const auto& face : faces) {
        int imageArea = img.cols * img.rows;

        if (face.area() < imageArea * 0.001 ||
            face.width < img.cols * 0.03 || face.height < img.rows * 0.03 ||
            face.width > img.cols * 0.6 || face.height > img.rows * 0.6)
            continue;

        float aspectRatio = static_cast<float>(face.width) / face.height;
        if (aspectRatio < 0.75f || aspectRatio > 1.33f)
            continue;

        cv::rectangle(img, face, cv::Scalar(0, 255, 0), 2);
        std::string imageName = file.substr(file.find_last_of("/\\") + 1);
        csv << imageName << "," << face.x << "," << face.y << "," << face.width << "," << face.height << "\n";
    }

    std::string outputPath = outputFolder + file.substr(file.find_last_of("/\\") + 1);
    cv::imwrite(outputPath, img);
}
