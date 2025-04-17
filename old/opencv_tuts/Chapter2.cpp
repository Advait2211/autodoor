// basic functions

#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>

using namespace cv;
using namespace std;


int main(){

    string path = "./Resources/lambo.png";
    Mat img = imread(path);
    Mat imgGrey, imgBlur, imgCanny, imgDil, imgEro;

    cvtColor(img, imgGrey, COLOR_BGR2GRAY);

    GaussianBlur(img, imgBlur, Size(5, 5), 5, 0);

    Canny(imgBlur, imgCanny, 50, 150);

    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));

    dilate(imgCanny, imgDil, kernel);
    erode(imgDil, imgEro, kernel);


    imshow("Image", img);
    imshow("Grey Image", imgGrey);
    imshow("Blurred Image", imgBlur);
    imshow("Canny Image", imgCanny);
    imshow("Dilated Image", imgDil);
    imshow("Eroded Image", imgEro);
    waitKey(0);

    return 0;
}