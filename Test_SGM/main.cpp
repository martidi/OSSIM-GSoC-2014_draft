#include <stdio.h>
#include <iostream>
#include <string>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv/cv.h>

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>

using namespace cv;
using namespace std;

char *windowDisparity = "Disparity";
char *windowDisparitySGM = "Disparity of SGM";
char *windowMatch = "TP matched";

int main(int argc, const char* argv[])
{
	cout <<"File Left:  "<< argv[1] <<"\n";
	cout <<"File Right: "<< argv[2] <<"\n";

	Mat imgLeft =  imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE );
	Mat imgRight = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE );
	
	if( !imgLeft.data || !imgRight.data )
	{ std::cout<< " --(!) Error reading images " << std::endl; return -1; }
	
	//resampling
	Mat outputLeft, outputRight;
	resize(imgLeft, outputLeft, Size(), 0.5, 0.5, INTER_AREA);
	resize(imgRight, outputRight, Size(), 0.5, 0.5, INTER_AREA);

	//computing detector
	OrbFeatureDetector detector(400);
	vector<KeyPoint> keypoints1, keypoints2;
	detector.detect(outputLeft, keypoints1);
	detector.detect(outputRight, keypoints2);
	
	// computing descriptors
	BriefDescriptorExtractor extractor;
	Mat descriptors1, descriptors2;
	extractor.compute(outputLeft, keypoints1, descriptors1);
	extractor.compute(outputRight, keypoints2, descriptors2);

	// matching descriptors
	BFMatcher matcher(NORM_L2);
	vector<DMatch> matches;
	matcher.match(descriptors1, descriptors2, matches);

	// drawing the results lower than a threshold based on distances 
	double max_dist = 0; double min_dist = 100;

	for( int i = 0; i < descriptors1.rows; i++ )
	{ double dist = matches[i].distance;
		if( dist < min_dist ) min_dist = dist;
		if( dist > max_dist ) max_dist = dist;
	}

	cout <<"Max dist:  "<< max_dist <<"\n";
	cout <<"Min dist: "<< min_dist <<"\n";

	vector< DMatch > good_matches;

	for( int i = 0; i < descriptors1.rows; i++ )
	{
		if(matches[i].distance <= max(2*min_dist, 200.50))
		{ 
			good_matches.push_back( matches[i]);

			//cout << " " << matches[i].queryIdx << " " << matches[i].trainIdx << " " << matches[i].imgIdx;

		    cout << i << " " << keypoints1[i].pt.x << " "
		              << keypoints1[i].pt.y << " "
		              << keypoints2[matches[i].trainIdx].pt.x << " "
		              << keypoints2[matches[i].trainIdx].pt.y 
		              <<endl;
	
	    }
	  // Computing parallax error
	 //matches[i].pt
	  
	}

	Mat img_matches;
	drawMatches( outputLeft, keypoints1, outputRight, keypoints2,
               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

  	namedWindow( windowMatch, CV_WINDOW_NORMAL );
	imshow( windowMatch, img_matches );
	
	// Disparity Map generation
	Mat imgDisparity16S = Mat( imgLeft.rows, imgLeft.cols, CV_16S );
	Mat imgDisparity8U  = Mat( imgLeft.rows, imgLeft.cols, CV_8UC1 );

	int ndisparities = 16*2;   /**< Range of disparity */
	int SADWindowSize = 3;    /**< Size of the block window. Must be odd */

	StereoSGBM sgbm;

	sgbm.preFilterCap = 63;
	sgbm.SADWindowSize = SADWindowSize > 0 ? SADWindowSize : 3;

	int cn = outputLeft.channels();

	sgbm.P1 = 8*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
	sgbm.P2 = 32*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
	sgbm.minDisparity = 0;
	sgbm.numberOfDisparities = ndisparities;
	sgbm.uniquenessRatio = 10;
	sgbm.speckleWindowSize = 100;
	sgbm.speckleRange = 32;
	sgbm.disp12MaxDiff = 1;
	
	Mat disp,disp8U;

	sgbm(imgLeft, imgRight, disp);

	double minVal; double maxVal;

	minMaxLoc( disp, &minVal, &maxVal );

	disp.convertTo( disp8U, CV_8UC1, 255/(maxVal - minVal));

	namedWindow( windowDisparitySGM, CV_WINDOW_NORMAL );
	imshow( windowDisparitySGM, disp8U );
	
	waitKey(0);

	return 0;
}
