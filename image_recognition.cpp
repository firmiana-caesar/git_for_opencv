
#include <iostream>  


#include <opencv2/highgui/highgui.hpp>    
#include <opencv2/imgproc/imgproc.hpp>    
#include <opencv2/core/core.hpp>   

using namespace cv;
using namespace std;

int main()
{
	//以下六行定义用于检测的颜色的HSV值
		int iLowH = 0;
		int iHighH = 10;
		int iLowS = 43;
		int iHighS = 255;
		int iLowV = 46;
		int iHighV = 255;
	
		//定义图片存储
		Mat imgOriginal = imread("F:\\timg.jpg");
		Mat imgHSV;
	
		//用向量存储经过HSV处理的图片
		vector<Mat> hsvSplit;
		cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV);
	
		//通道分离并处理
		split(imgHSV, hsvSplit);
		equalizeHist(hsvSplit[2], hsvSplit[2]);
		merge(hsvSplit, imgHSV);
		Mat imgThresholded;
	
		//颜色分离
		inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded);
	
		//开操作 (去除一些噪点)
		Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
		morphologyEx(imgThresholded, imgThresholded, MORPH_OPEN, element);cv::
	
		//闭操作 (连接一些连通域)
		morphologyEx(imgThresholded, imgThresholded, MORPH_CLOSE, element);
	
		//进行膨胀操作;i的值根据光照条件进行改变
		for (int i = 0; i < 3; i++)
		{
			dilate(imgThresholded, imgThresholded, element);
		}
	
	
		//显示相应的图片
		imshow("Thresholded Image", imgThresholded);


	/////////////////////////////////////////////////////////////////////  
		//void cv::drawContours(
		//  InputOutputArray     image,                                      目标图像
		//	InputArrayOfArrays     contours,                                 输入轮廓，每一个轮廓是一个点向量( ? )
		//	int     contourIdx,                                              轮廓编号
		//	const Scalar &     color,                                         
		//	int     thickness = 1,                                           
		//	int     lineType = LINE_8,                                       8/4连通线性
		//	InputArray     hierarchy = noArray(),                            可选层次结构( ? )
		//	int     maxLevel = INT_MAX,                                      最大等级( ? )
		//	Point     offset = Point()                                       轮廓偏移参数
		//)
	vector<vector<Point>> contours;
	                                                                         //轮廓查找函数
	findContours(imgThresholded,
		contours,
		CV_RETR_EXTERNAL,
		CV_CHAIN_APPROX_NONE);

							                                                 //打印轮廓数量
	cout << "轮廓数量： " << contours.size() << endl;
	vector<vector<Point>>::const_iterator itContours = contours.begin();
	for (; itContours != contours.end(); ++itContours)
	{

		cout << "Size: " << itContours->size() << endl;                      //轮廓长度
	}

	                                                                         //画出轮廓 
	Mat result(imgThresholded.size(), CV_8U, Scalar(255));
	drawContours(result, contours,                                           //定义参数和findContours参数大致对应
		-1, // draw all contours  
		Scalar(0), // in black  
		2); // with a thickness of 2  

	imshow("Contours", result);


	                                                                         //在原图像上画出轮廓  
	Mat original = imread("F:\\timg.jpg");
	drawContours(original, contours,
		-1, // draw all contours  
		Scalar(255, 255, 255),// in white  
		-1); // with a thickness of 2  

	imshow("Contours on Original", original);

	waitKey(0);

	return 0;
}
		 
