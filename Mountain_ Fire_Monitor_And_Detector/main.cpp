#include <opencv2\opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;
Mat checkRGB(Mat &original);
void drawContours(Mat &orginal, Mat input);
Ptr<BackgroundSubtractor> mog2 = createBackgroundSubtractorMOG2();
Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
Mat trackMoving(Mat original);
//tracking fire
Mat checkRGB(Mat &original)
{
	Mat dst;
	dst.create(original.size(), CV_8UC1);
	int redThre = 115;
	int saturationTh = 45;
	Mat RGB[3];
	split(original, RGB);
	for (int i = 0; i < original.rows; i++)
	{
		for (int j = 0; j < original.cols; j++)
		{
			float B = RGB[0].at<uchar>(i, j);
			float G = RGB[1].at<uchar>(i, j);
			float R = RGB[2].at<uchar>(i, j);

			int minValue = min(R, min(G, B));
			double S = (1 - 3.0 * minValue / (R + G + B));
			if (R > redThre && R >= G && G >= B && S > 0.20 && S > ((255 - R) * saturationTh / redThre))
			{
				dst.at<uchar>(i, j) = 255;
			}
			else
			{
				dst.at<uchar>(i, j) = 0;
			}
		}
	}
	dilate(dst, dst, kernel, Point(-1, -1));
	return dst;
}
//draw the contours of the fire. Locking the fire
vector<vector<Point>> contours;
vector<Vec4i> hireachy;
void drawContours(Mat &orginal, Mat input)
{
	findContours(input, contours, hireachy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));
	for (int i = 0; i < contours.size(); i++) {
		Rect selection = boundingRect(contours[i]);
		if (selection.width < 3 || selection.height < 3)
			continue;
		rectangle(orginal, selection, Scalar(0, 255, 0), 2, 8, 0);
	}
}

//tracking smoke
Mat trackMoving(Mat original)
{
	Mat fgmask;
	mog2->apply(original, fgmask);
	threshold(fgmask, fgmask, 100, 255, THRESH_BINARY);
	morphologyEx(fgmask, fgmask, MORPH_OPEN, kernel, Point(-1, -1));
	dilate(fgmask, fgmask, kernel, Point(-1, -1));
	return fgmask;
}

int main()
{
	// test image;
	Mat res = imread("E:/mf1.png");
	Mat dst = checkRGB(res);
	drawContours(res, dst);
	imshow("Fire Binary", dst);
	imshow("Fire", res);
	waitKey(0);

	//test smoke
	VideoCapture capture(0);
	capture.open("E:/smoke.avi");
	Mat frame, smoke;
	while (capture.read(frame))
	{
		smoke = trackMoving(frame);
		drawContours(frame, smoke);
		imshow("Smoke Binary", smoke);
		imshow("Smoke", frame);

		char c = waitKey(40);
		if (c == 27)
			break;
	}
	capture.release();
	waitKey(0);

}