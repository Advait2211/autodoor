#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

///////////////// Webcam //////////////////////

int main()
{

    VideoCapture cap(1);
    Mat img;

    while (true)
    {

        cap.read(img);
        imshow("Image", img);
        if(waitKey(1) == 27) break;
    }
    return 0;
}