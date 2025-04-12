// photo, video and webcam capture

#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>

using namespace cv;
using namespace std;

void getImages(string path){
    Mat img = imread(path);
    if (img.empty()){
        cout << "Could not open or find the image" << endl;
        return;
    }
    imshow("Image", img);
    waitKey(0);
}

void getVideo(string path){

    VideoCapture cap(path);

    if (!cap.isOpened()){
        cout << "Could not open or find the video" << endl;
        return;
    }

    Mat img;
    while (true){

        cap.read(img);
        if (img.empty()){
            cout << "End of video" << endl;
            break;
        }

        imshow("Video1", img);
        if (waitKey(30) >= 0) break; // Wait for 30 ms or until a key is pressed
    }

    cap.release();
    destroyAllWindows();
    cout << "Video closed" << endl;
}

void getWebcam(int path){
    VideoCapture cap(path);

    if (!cap.isOpened()){
        cout << "Could not open or find the camera" << endl;
        return;
    }

    Mat img;
    while (true){

        cap.read(img);
        if (img.empty()){
            cout << "End of webcam footage" << endl;
            break;
        }

        imshow("webcame", img);
        if (waitKey(30) >= 0) break; // Wait for 30 ms or until a key is pressed
    }

    cap.release();
    destroyAllWindows();
    cout << "Webcam closed" << endl;
}

int main(){
    // getImages("./Resources/lambo.png");

    getVideo("./Resources/test_video.mp4");

    getWebcam(0);

    return 0;
}