#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main()
{
    Mat img = imread("lena.jpg");
        namedWindow("ceshi");
            imshow("ceshi", img);
            waitKey(0);
    return 0;
}
