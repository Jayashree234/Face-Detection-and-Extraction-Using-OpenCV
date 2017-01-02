#include "opencv2/opencv.hpp"
#include "opencv2/video/tracking.hpp"

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include <time.h>
#include <opencv/cv.h>

using namespace std;
using namespace cv;

// Global variables
// Copy this file from opencv/data/haarscascades to target folder
string face_cascade_name ="/home/jayashree/opencv/opencv-3.0.0-alpha/data/haarcascades/haarcascade_frontalface_alt_tree.xml" ;
CascadeClassifier face_cascade;
string window_name = "Capture - Face detection";
int filenumber; // Number of file to be saved
string filename;
Mat crop;


const float size_factor = 1.1;
const std::size_t num_buffers = 2;
const Size face_size(30, 30);
int trackObject = 0;
Rect selection;
Rect trackWindow;
int hsize = 16;
float hr[] = {0,180};
const float* ph = hr;
int matchesNum = 0;
Mat hsv, hue, hist, mask, backproj, det;


// Function detectAndDisplay
 std::vector<Rect> detectAndDisplay(Mat frame)
{
    std::vector<Rect> faces;
    Mat frame_gray;
    Mat res;
    Mat gray;
    string text;
    stringstream sstm;

    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);

// Detect faces
    face_cascade.detectMultiScale(frame_gray, faces, size_factor, num_buffers, CASCADE_SCALE_IMAGE, face_size);

// Set Region of Interest
    cv::Rect roi;
//float fsize= faces.size();
    size_t ic = 0; // ic is index of current element

    size_t ib = 0; // ib is index of biggest element


    for (ic = 0; ic <faces.size(); ic++) // Iterate through all current elements (detected faces)

    {
        int area_c = faces[ic].width * faces[ic].height;
int area_b = faces[ib].width * faces[ib].height;

if (area_c > area_b) 
    roi = faces[ic];
 else 
    roi = faces[ib];

crop = frame(roi);
//nframe = frame(roi);

	 Point pt1(faces[ic].x, faces[ic].y); // Display detected faces on main window - live stream from camera
        Point pt2((faces[ic].x + faces[ic].height), (faces[ic].y + faces[ic].width));
rectangle(frame, pt1, pt2, Scalar(100, 100, 100), 1, 8, 0);
        
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

    
// Show image
    sstm << "Crop area size: " << roi.width << "x" << roi.height << " Filename: " << filename;
    text = sstm.str();

    putText(frame, text, cvPoint(30, 30), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 0, 255), 1, CV_AA);
    //imshow("original", frame); 
	return faces;

}




void trackface(Mat image)
{

 
    if ( !trackObject )
    { 
	Mat gf; 
        vector <Rect> facesBuf;
        int detectionsNum = 0;

        cvtColor(image, gf, CV_BGR2GRAY);  
   facesBuf=detectAndDisplay(image);
detectionsNum = (int) facesBuf.size();
        Rect *faceRects = &facesBuf[0];

        
        if (detectionsNum > 0) 
            matchesNum += 1;
        else matchesNum = 0;
        if ( matchesNum == 1)
        {
            trackObject = -1;
            selection = faceRects[0];
        }   

    }

    if( trackObject )
    {
        cvtColor(image, hsv, CV_BGR2HSV);
        inRange(hsv, Scalar(0, 69, 53),
             Scalar(180, 256, 256), mask);
        int ch[] = {0, 0};
        hue.create(hsv.size(), hsv.depth());
        mixChannels(&hsv, 1, &hue, 1, ch, 1);

        if( trackObject < 0 )
        {
            Mat roi(hue, selection), maskroi(mask, selection);
            calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &ph);
            normalize(hist, hist, 0, 255, CV_MINMAX);

            trackWindow = selection;
            trackObject = 1;      
        }

        calcBackProject(&hue, 1, 0, hist, backproj, &ph);

        backproj &= mask;
        RotatedRect trackBox = CamShift(backproj, trackWindow,TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));
        if( trackWindow.area() <= 1 )
        {
            int cols = backproj.cols, rows = backproj.rows, r = (MAX(cols, rows) );
            trackWindow = Rect(trackWindow.x - r, trackWindow.y - r,trackWindow.x + r, trackWindow.y + r) & Rect(0, 0, cols, rows);
        }

       ellipse( image, trackBox, Scalar(0,255,255), 2);
    }

//destroyWindow("original");
   imshow( "Result", image );

}




// Function main
int main(int argc, char *argv[])
{

	const string srcRef = argv[1];
	if (argc != 2)
    {
        cout << "Not enough parameters" << endl;
        return -1;
    }
    VideoCapture capture(srcRef);

    if (!capture.isOpened())  // check if we succeeded
        return -1;

    // Load the cascade
    if (!face_cascade.load(face_cascade_name))
    {
        printf("--(!)Error loading\n");
        return (-1);
    }
double fps = capture.get(CV_CAP_PROP_FPS);
    
    cout << "Frames per second using video.get(CAP_PROP_FPS) : " << fps << endl;
     
 
    int num_frames = capture.get(CV_CAP_PROP_FRAME_COUNT);
cout << "Capturing " << num_frames<< " frames" << endl ;

    // Read the video stream
    Mat frame;

    for (;;)
    {
        capture >> frame;

        // Apply the classifier to the frame
        if (frame.empty())
        {
             break;
        }
        else
	trackface(frame);

        int c = waitKey(10);

        if (27 == char(c))
        {
            break;
        }
    }
}
