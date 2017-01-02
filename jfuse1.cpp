#define _ITERATOR_DEBUG_LEVEL = 0

#include "opencv2/opencv.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <time.h>
#include <iostream>
#include <stdio.h>
#include <opencv/cv.h>
 
using namespace cv;
using namespace std;


int trackObject = 0;
Rect selection;
Rect trackWin;
int hsize = 16;
float hr[] = {0,180};
const float* ph = hr;
int matchesNum = 0;
Mat image;
Mat frame, hsv, hue, hist, mask, backproj,crop;

Point camCenter;
Point KFPredictCenter;
Point KFCorrectCenter;
int frameStart;
KalmanFilter KF;
Mat_<float> measurement;
string winName;


double norm_L2(const cv::Point &x, const cv::Point &y)
{
	return std::sqrt((double)((x.x-y.x)*(x.x-y.x)+(x.y-y.y)*(x.y-y.y)));
}


void initKalman(double interval)
{
	const int stateNum = 4;
	const int measureNum = 2;
	Mat statePost = (Mat_<float>(stateNum, 1) << trackWin.x+trackWin.width/2.0, trackWin.y+trackWin.height/2.0, 0,0);
	Mat transitionMatrix = (Mat_<float>(stateNum, stateNum) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1);
	KF.init(stateNum, measureNum);
	KF.transitionMatrix = transitionMatrix;
	KF.statePost = statePost;
	setIdentity(KF.measurementMatrix);
	setIdentity(KF.processNoiseCov, Scalar::all(1e-1));
	setIdentity(KF.measurementNoiseCov, Scalar::all(1e-3));
	setIdentity(KF.errorCovPost, Scalar::all(0.1));
	measurement = Mat::zeros(measureNum, 1, CV_32F);
}


Point getCurrentState()
{
	Mat statePost = KF.statePost;
	return Point(statePost.at<float>(0), statePost.at<float>(1));
}


void setCurrentTrackWindow()
{
	int cols = image.cols;
	int rows = image.rows;
	trackWin.x = KFCorrectCenter.x - trackWin.width/2;
	trackWin.y = KFCorrectCenter.y - trackWin.height/2;
 	//trackWin.x = MAX(0, trackWin.x);
 	//trackWin.x = MIN(cols, trackWin.width);
 	//trackWin.y = MAX(0, trackWin.y);
 	//trackWin.y = MIN(rows, trackWin.height);
	trackWin &= Rect(0, 0, cols, rows);
	if(trackWin.width <= 0 || trackWin.height <=0)
	{
		int width = MIN(KFCorrectCenter.x, cols-KFCorrectCenter.x)*2;
		int height = MIN(KFCorrectCenter.y, rows-KFCorrectCenter.y)*2;
		trackWin = Rect(KFCorrectCenter.x-width/2, KFCorrectCenter.y-height/2, width, height);
	}
}


void normalizeHist(Mat &hist)
{
	if(hist.dims != 3)
	{
		cout << "can only normalize hist when hist.dims = 3" << endl;
		exit(-1);
	}
	/*
	* find the min and max elements in hist
	*/
	float min = 1e10, max = 0;
	for(int i=0;i<hist.size[0];i++)
	for(int j=0;j<hist.size[1];j++)
	for(int k=0;k<hist.size[2];k++)
	{
		float tmp = hist.at<float>(i, j, k);
		if(tmp < min)
			min = tmp;
		if(tmp > max)
			max = tmp;
	}
	/*
	* normalize the hist
	*/
	hist = (hist-min)/(max-min)*255;
}


void drawTrackResult()
{
	Mat img;
	image.copyTo(img);
	circle(img, camCenter, 2, Scalar(255,0,0), 2, CV_AA); // draw camshift result
	circle(img, KFPredictCenter, 2, Scalar(0,255,0), 2, CV_AA); // draw kalman predict result
	circle(img, KFCorrectCenter, 2, Scalar(0,0,255), 2, CV_AA); // draw kalman correct result
	rectangle(img, trackWin, Scalar(0,0,255), 3, CV_AA); // draw track window
	waitKey(10);
	if (!img.empty()) 
	{
    	imshow("Tracker", img);
	}
}


int main(int argc, char* argv[])
{
 
	CascadeClassifier cascade;
	cascade.load("/home/jayashree/opencv/opencv-3.0.0-alpha/data/haarcascades/haarcascade_frontalface_alt_tree.xml" );
	const string srcRef = argv[1];
	if (argc != 2)
    {
        cout << "Not enough parameters" << endl;
        return -1;
    }
    
    VideoCapture cap(srcRef);
    // VideoCapture cap("/home/jayashree/opencv_2/opencv-2.4.9/working/cwoman.mp4");

	if(!cap.isOpened())
	{
		cerr << "open error" << endl;
		exit(-1);
	}
    // Load the cascade
    
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 500);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 500);
	namedWindow("Tracker", WINDOW_NORMAL);
    //setWindowProperty("Tracker", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
    Point lastCenter(trackWin.x,trackWin.y);
    bool isLost = false;
    double fps = cap.get(CV_CAP_PROP_FPS);
    double interval = 1.0/cap.get(CV_CAP_PROP_FPS);
    cout << "Frames per second using video.get(CAP_PROP_FPS) : " << fps << endl;
    int num_frames = cap.get(CV_CAP_PROP_FRAME_COUNT);
    time_t start, end;
    cout << "Capturing " << num_frames<< " frames" << endl ;
    initKalman(interval);
	cap.set(CV_CAP_PROP_POS_FRAMES, frameStart);
	time(&start);
    for(int i = 0; i < num_frames; i++)
    {
		cap >> frame;
		//waitKey(1000);
    	if( frame.empty() )
        	break;

    	frame.copyTo(image);
		Mat gf; 
        vector <Rect> facesBuf;
        Mat res;
    	Mat gray;
    	string text;
    	stringstream sstm;
    	int filenumber; // Number of file to be saved
    	string filename;
		cvtColor(image, gf, CV_BGR2GRAY);
        cascade.detectMultiScale(gf, facesBuf, 1.2, 4, CV_HAAR_FIND_BIGGEST_OBJECT |CV_HAAR_SCALE_IMAGE, cvSize(0, 0));
		// Set Region of Interest
    	cv::Rect ca;
		//float fsize= faces.size();
    	size_t ic = 0; // ic is index of current element

    	size_t ib = 0; // ib is index of biggest element


    	for (ic = 0; ic <facesBuf.size(); ic++) // Iterate through all current elements (detected faces)

    	{
        	int area_c = facesBuf[ic].width * facesBuf[ic].height;
			int area_b = facesBuf[ib].width * facesBuf[ib].height;

			if (area_c > area_b) 
    			ca = facesBuf[ic];
 			else 
    			ca = facesBuf[ib];

			crop = frame(ca);
			resize(crop, res, Size(128, 128), 0, 0, INTER_LINEAR); // This will be needed later while saving images
			cvtColor(crop, gray, CV_BGR2GRAY); // Convert cropped image to Grayscale

        	// Form a filename
        	filename = "";
        	stringstream ssfn;
        	ssfn << filenumber << ".png";
        	filename = ssfn.str();
        	filenumber++;
			imwrite(filename, gray);
 		}

		sstm << "Crop area size: " << ca.width << "x" << ca.height << " Filename: " << filename;
    	text = sstm.str();
		putText(frame, text, cvPoint(30, 30), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 0, 255), 1, CV_AA);
		
		Rect *faceRects = &facesBuf[0];
		selection = faceRects[0];
        cvtColor(image, hsv, CV_BGR2HSV);
        inRange(hsv, Scalar(0, 69, 53), Scalar(180, 256, 256), mask);
        int ch[] = {0, 0};
        hue.create(hsv.size(), hsv.depth());
        mixChannels(&hsv, 1, &hue, 1, ch, 1);
		Mat roi(hue, selection), maskroi(mask, selection);
        calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &ph);
        normalize(hist, hist, 0, 255, CV_MINMAX);
	    
		while(1)
		{
			cap.read(image);
			//waitKey(1000);
			if(image.empty())
				break;
			cvtColor(image, hsv, CV_BGR2HSV);
			inRange(hsv, Scalar(0, 69, 53), Scalar(180, 256, 256), mask);
			hue.create(hsv.size(), hsv.depth());
			mixChannels(&hsv, 1, &hue, 1, ch, 1);
            trackWin = selection;
            calcBackProject(&hue, 1, 0, hist, backproj, &ph);
			backproj &= mask;
			if(trackWin.width <= 0 || trackWin.height <=0)
				trackWin = Rect(0, 0, image.cols, image.rows);
        	RotatedRect box = CamShift(backproj, trackWin,TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));
        	
			camCenter = Point(box.center.x, box.center.y);
			KF.predict();
			KFPredictCenter = getCurrentState();
			/*
			* set measurement
			*/
			measurement.at<float>(0) = camCenter.x;
			measurement.at<float>(1) = camCenter.y;
			/*
			* do kalman correction
			*/
			KF.correct(measurement);
			KFCorrectCenter = getCurrentState();

			if(norm_L2(KFCorrectCenter, lastCenter)>350 || trackWin.area() < 10 || trackWin.width <= 0 || trackWin.height <=0)
			{
				isLost = true;
				Mat im;
				image.copyTo(im);
				putText(im, "Target Lost", Point(im.rows/2,im.cols/4), cv::FONT_HERSHEY_PLAIN, 2.0, Scalar(0,0,255), 2);
				imshow("result", im);
				waitKey();
			}
			
			else
			drawTrackResult();
		}
        
		if(!isLost)
			lastCenter = KFCorrectCenter;
		waitKey();
		int key = waitKey(int(interval*1000));
		if(key == 27)
			break;
		setCurrentTrackWindow();
	}
   
    time(&end);
    double seconds = difftime (end, start);
    cout << "Time taken : " << seconds << " seconds" << endl;
    fps  = num_frames / seconds;
    cout << "Estimated frames per second : " << fps << endl;
     
}
