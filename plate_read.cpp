#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <iostream>

int main() {
    cv::Mat image = cv::imread("car.jpg");
    if (image.empty()) {
        std::cerr << "Image not found.\n";
        return 1;
    }

    // Resize for faster processing
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(800, 600));

    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);

    // Noise reduction
    cv::Mat blur;
    cv::bilateralFilter(gray, blur, 11, 17, 17);

    // Edge detection
    cv::Mat edged;
    cv::Canny(blur, edged, 30, 200);

    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edged, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    // Sort by area, descending
    std::sort(contours.begin(), contours.end(), [](const auto& a, const auto& b) {
        return cv::contourArea(a) > cv::contourArea(b);
    });

    cv::Rect plateRect;
    for (const auto& contour : contours) {
        std::vector<cv::Point> approx;
        cv::approxPolyDP(contour, approx, 0.018 * cv::arcLength(contour, true), true);
        if (approx.size() == 4) {
            plateRect = cv::boundingRect(approx);
            break;
        }
    }

    if (plateRect.area() == 0) {
        std::cerr << "Number plate not detected.\n";
        return 1;
    }

    // Crop the detected plate
    cv::Mat plate = resized(plateRect);

    // Preprocess for Tesseract
    cv::Mat plateGray;
    cv::cvtColor(plate, plateGray, cv::COLOR_BGR2GRAY);
    cv::threshold(plateGray, plateGray, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // OCR using Tesseract
    tesseract::TessBaseAPI tess;
    tess.Init(NULL, "eng", tesseract::OEM_LSTM_ONLY);
    tess.SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);
    tess.SetImage(plateGray.data, plateGray.cols, plateGray.rows, 1, plateGray.step);

    std::string text = tess.GetUTF8Text();
    std::cout << "Detected Number Plate: " << text << std::endl;

    // Optional: Show the cropped plate
    cv::imshow("Detected Plate", plate);
    cv::waitKey(0);

    return 0;
}
