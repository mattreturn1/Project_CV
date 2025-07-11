#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char** argv) {

    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " image" << std::endl;
    }
    cv::Mat img = cv::imread(argv[1]);

    if (img.empty()) {
        std::cout << "Error: Image not found!" << std::endl;
        return -1;
    }
    cv::imshow("Loaded Image", img);
    cv::waitKey(0);

    return 0;
}
