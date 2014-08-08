#include <ossim/base/ossimString.h>
#include <ossim/base/ossimNotify.h>
#include <ossim/base/ossimTrace.h>
#include <ossim/base/ossimIrect.h>
#include <ossim/base/ossimRefPtr.h>
#include <ossim/base/ossimConstants.h>
#include <ossim/elevation/ossimElevManager.h>
#include <ossim/imaging/ossimImageData.h>
#include <ossim/imaging/ossimImageSource.h>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/median.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include "DisparityMap.h"

#include <opencv/highgui.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/flann/flann.hpp>
#include <opencv2/legacy/legacy.hpp>
// Note: These are purposely commented out to indicate non-use.
// #include <opencv2/nonfree/nonfree.hpp>
// #include <opencv2/nonfree/features2d.hpp>
// Note: These are purposely commented out to indicate non-use.
#include <vector>
#include <iostream>

DisparityMap::DisparityMap()
{
	
}

void DisparityMap::execute(cv::Mat master_mat, cv::Mat slave_mat)
{
	cout << "Disparity Map generation..." << endl;
	/// Disparity Map generation
	//cv::Mat imgDisparity16S = cv::Mat( master_mat.rows, master_mat.cols, CV_16S );
	//cv::Mat imgDisparity8U  = cv::Mat( master_mat.rows, master_mat.cols, CV_8UC1 );

	int ndisparities = 16*2;   
	int SADWindowSize = 3;   

	cv::StereoSGBM sgbm;

	sgbm.preFilterCap = 63;
	sgbm.SADWindowSize = SADWindowSize > 0 ? SADWindowSize : 3;

	int cn = master_mat.channels();

	sgbm.P1 = 8*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
	sgbm.P2 = 32*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
	sgbm.minDisparity = 50-8;
	sgbm.numberOfDisparities = ndisparities;
	sgbm.uniquenessRatio = 10;
	sgbm.speckleWindowSize = 100;
	sgbm.speckleRange = 32;
	sgbm.disp12MaxDiff = 1;
	
	double minVal, maxVal;
	
	cv::Mat array_disp;
	cv::Mat array_disp_8U;
       
	sgbm(master_mat, slave_mat, array_disp);
	minMaxLoc( array_disp, &minVal, &maxVal );
	array_disp.convertTo( array_disp_8U, CV_8UC1, 255/(maxVal - minVal), -minVal*255/(maxVal - minVal));   
    
	cv::namedWindow( "SGM Disparity", CV_WINDOW_NORMAL );
	cv::imshow( "SGM Disparity", array_disp_8U);

	cv::waitKey(0);
}



