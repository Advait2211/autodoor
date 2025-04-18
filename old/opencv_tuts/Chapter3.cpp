// resize and crop images

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{

    string path = "./Resources/lambo.png";
    
    Mat img = imread(path);

    Mat imgResize, imgCrop;

    // cout << img.size() << endl;

resize(img, imgResize, Size(), 0.5, 0.5);

Rect roi(200, 100, 300, 300);
imgCrop = img(roi);

imshow("Image", img);    
imshow("Resized Image", imgResize);    
imshow("Cropped Image", imgCrop);   
waitKey(0);
return 0;
}

// git push to maintain daily streak
